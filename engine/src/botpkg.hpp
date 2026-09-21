// Bot packages: the standardised way somebody else's bot gets onto this table.
//
// Until now an outside bot arrived as a GitHub link and left as a hand-written
// bridge -- src/kv.hpp is a port of KV's search, src/kv6.hpp is a translator for
// their JSON service, and both took a person a day to write and to argue with.
// That does not scale to "my friends have bots", so this is the other half: a
// package anybody can produce, that this engine can install and seat without a
// line of new C++.
//
// A package is a .zip holding a manifest named `fishbot.json` (at the root, or
// inside a single top-level directory) and whatever the bot needs beside it --
// source, weights, data.  The manifest says how to launch it and which protocol
// it speaks; docs/BOT_PACKAGE.md is the document to hand to whoever is writing
// one.  The protocol itself is in extbot.hpp.
//
// Two properties this file exists to guarantee:
//
//   * INSTALLING NEVER EXECUTES.  Unpacking a package runs no code from it --
//     no setup hook, no import, nothing.  The uploaded bot is inert on disk
//     until somebody with the host credential seats it and deals, and the
//     dependency step (prepare) is a separate, explicitly host-triggered
//     action.  That is the whole of the trust boundary, and it is worth saying
//     out loud: a bot that plays is a program the host chose to run.
//   * NOTHING MEASURED CHANGES.  This header is included by factory.hpp's
//     `bot:` branch and by serve.hpp, and reaches no policy, no belief and no
//     rule.  A build with it produces every published number bit for bit.
#pragma once
#include "zip.hpp"
#include "minijson.hpp"
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>

namespace fish {
namespace botpkg {

// Budgets.  Generous enough for a bot that ships learned weights, small enough
// that a mistake or a prank costs a bounded amount of disk.
static constexpr size_t MAX_ZIP_BYTES   = 64u * 1024 * 1024;
static constexpr size_t MAX_UNPACKED    = 512u * 1024 * 1024;
static constexpr size_t MAX_ENTRIES     = 8192;
static constexpr size_t MAX_ENTRY_BYTES = 256u * 1024 * 1024;

// ------------------------------------------------------------ tiny fs helpers
inline bool isDirPath(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}
inline bool isFilePath(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}
inline bool mkdirp(const std::string& p) {
  if (p.empty()) return false;
  if (isDirPath(p)) return true;
  size_t slash = p.find_last_of('/');
  if (slash != std::string::npos && slash > 0 && !mkdirp(p.substr(0, slash))) return false;
  return ::mkdir(p.c_str(), 0755) == 0 || isDirPath(p);
}
inline bool writeWhole(const std::string& p, const std::string& data) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  if (!f) return false;
  f.write(data.data(), std::streamsize(data.size()));
  return bool(f);
}
inline std::string readWhole(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}
// Recursive delete, restricted by the caller to paths under the bots root.
// lstat, never stat: a prepared package contains a venv, and a venv is mostly
// symlinks to the system interpreter -- following one would delete the wrong
// tree entirely.
inline bool rmTree(const std::string& p) {
  struct stat st;
  if (::lstat(p.c_str(), &st) != 0) return true;             // already gone
  if (!S_ISDIR(st.st_mode)) return ::unlink(p.c_str()) == 0;
  DIR* d = ::opendir(p.c_str());
  if (!d) return false;
  bool ok = true;
  while (dirent* e = ::readdir(d)) {
    std::string n = e->d_name;
    if (n == "." || n == "..") continue;
    ok = rmTree(p + "/" + n) && ok;
  }
  ::closedir(d);
  return ::rmdir(p.c_str()) == 0 && ok;
}
inline std::string absolutise(const std::string& p) {
  if (!p.empty() && p[0] == '/') return p;
  char cwd[4096];
  if (!getcwd(cwd, sizeof(cwd))) return p;
  return std::string(cwd) + "/" + p;
}
inline long long nowSeconds() { return (long long)::time(nullptr); }

// PATH lookup.  execvp does this for the native bridge; kv6.hpp's bridge uses
// execl, which does not, so an uploaded package that spells its interpreter the
// way every README does -- a bare "python3" -- has to be resolved before it
// gets there.
inline std::string whichProgram(const std::string& name) {
  if (name.empty()) return name;
  if (name.find('/') != std::string::npos) return name;
  const char* path = getenv("PATH");
  std::string p = path && *path ? path : "/usr/bin:/bin:/usr/local/bin";
  size_t at = 0;
  while (at <= p.size()) {
    size_t colon = p.find(':', at);
    std::string dir = p.substr(at, (colon == std::string::npos ? p.size() : colon) - at);
    if (!dir.empty()) {
      std::string cand = dir + "/" + name;
      if (::access(cand.c_str(), X_OK) == 0 && isFilePath(cand)) return cand;
    }
    if (colon == std::string::npos) break;
    at = colon + 1;
  }
  return std::string();
}

// ----------------------------------------------------------------- manifest
struct Manifest {
  std::string format;                  // must be "fishlab-bot/1"
  std::string id;                      // optional; derived from the name if absent
  std::string name;
  std::string version;
  std::string author;
  std::string description;
  std::string protocol = "fishlab-json-v1";
  std::vector<std::string> run;        // argv, executed with cwd = the package dir
  std::map<std::string, std::string> env;
  bool venv = false;                   // python.venv
  std::string requirements;            // python.requirements
  int timeoutMs = 15000;               // per decision
  // The engine polls every seat for a declaration before every move, because
  // the rules allow declaring at any moment and a bot that only ever declares
  // on its own turn plays a materially weaker game.  It is also five extra
  // round trips per event, so a package whose author would rather have the
  // speed can decline them.
  bool pollOffTurn = true;             // poll_off_turn
};

// Identifiers are used as directory names, spec strings and query parameters, so
// the alphabet is deliberately narrow rather than escaped in four places.
inline std::string slugify(const std::string& s) {
  std::string o;
  for (char c : s) {
    char l = char(tolower((unsigned char)c));
    if ((l >= 'a' && l <= 'z') || (l >= '0' && l <= '9')) o.push_back(l);
    else if (!o.empty() && o.back() != '-') o.push_back('-');
  }
  while (!o.empty() && o.back() == '-') o.pop_back();
  if (o.size() > 32) o = o.substr(0, 32);
  while (!o.empty() && o.back() == '-') o.pop_back();
  return o;
}
inline bool validId(const std::string& s) {
  if (s.empty() || s.size() > 32) return false;
  for (char c : s) if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-')) return false;
  return s.front() != '-' && s.back() != '-';
}

inline std::string clip(std::string s, size_t n) {
  std::string o;
  for (char c : s) if ((unsigned char)c >= 0x20 && c != 0x7f) o.push_back(c);
  size_t a = o.find_first_not_of(" \t");
  size_t b = o.find_last_not_of(" \t");
  o = (a == std::string::npos) ? std::string() : o.substr(a, b - a + 1);
  return o.size() > n ? o.substr(0, n) : o;
}

inline bool parseManifest(const std::string& text, Manifest& m, std::string& err) {
  mj::Value v;
  if (!mj::parse(text, v, err)) { err = "fishbot.json is not valid JSON: " + err; return false; }
  if (!v.isObj()) { err = "fishbot.json must be a JSON object"; return false; }

  m.format = v.s("format");
  if (m.format != "fishlab-bot/1") {
    err = m.format.empty()
        ? "fishbot.json has no \"format\"; it must be \"fishlab-bot/1\""
        : "fishbot.json declares format \"" + clip(m.format, 40) + "\"; this engine speaks \"fishlab-bot/1\"";
    return false;
  }
  m.name = clip(v.s("name"), 40);
  if (m.name.empty()) { err = "fishbot.json needs a \"name\""; return false; }
  m.id = slugify(v.s("id").empty() ? m.name : v.s("id"));
  if (!validId(m.id)) { err = "the bot's name gives no usable id; add an \"id\" of a-z, 0-9 and dashes"; return false; }
  m.version     = clip(v.s("version", "0"), 20);
  m.author      = clip(v.s("author"), 40);
  m.description = clip(v.s("description"), 200);
  m.protocol    = v.s("protocol", "fishlab-json-v1");
  if (m.protocol != "fishlab-json-v1" && m.protocol != "kv-json-v1") {
    err = "unknown protocol \"" + clip(m.protocol, 40) +
          "\"; this engine speaks \"fishlab-json-v1\" (see docs/BOT_PACKAGE.md) and "
          "\"kv-json-v1\" for packages written against KV's bridge";
    return false;
  }

  const mj::Value* run = v.find("run");
  if (!run || !run->isArr() || run->arr.empty()) {
    err = "fishbot.json needs a \"run\": an argv array, e.g. [\"python3\",\"-m\",\"mybot\"]";
    return false;
  }
  if (run->arr.size() > 32) { err = "\"run\" has too many arguments"; return false; }
  for (const auto& a : run->arr) {
    if (!a.isStr() || a.str.empty()) { err = "every element of \"run\" must be a non-empty string"; return false; }
    if (a.str.size() > 300) { err = "an element of \"run\" is absurdly long"; return false; }
    for (unsigned char c : a.str) if (c < 0x20) { err = "an element of \"run\" contains a control character"; return false; }
    m.run.push_back(a.str);
  }
  // An argv[0] that walks out of the package would launch something the package
  // does not contain.  A bare name (python3) is looked up on PATH, which is the
  // normal case; a path is resolved against the package directory.
  if (m.run[0].find("..") != std::string::npos || m.run[0][0] == '/') {
    err = "\"run\" must name a program on PATH or a file inside the package, not an absolute or parent path";
    return false;
  }

  const mj::Value* env = v.find("env");
  if (env && env->isObj()) {
    for (const auto& kv : env->obj) {
      if (!kv.second.isStr()) { err = "\"env\" values must be strings"; return false; }
      if (kv.first.empty() || kv.first.find('=') != std::string::npos) { err = "bad name in \"env\""; return false; }
      if (m.env.size() >= 32) break;
      m.env[kv.first] = clip(kv.second.str, 300);
    }
  }
  const mj::Value* py = v.find("python");
  if (py && py->isObj()) {
    m.venv = py->bo("venv", false);
    m.requirements = clip(py->s("requirements"), 100);
    if (!m.requirements.empty() &&
        (m.requirements[0] == '/' || m.requirements.find("..") != std::string::npos)) {
      err = "\"python.requirements\" must be a file inside the package";
      return false;
    }
    if (!m.requirements.empty()) m.venv = true;
  }
  double t = v.n("timeout_ms", 15000);
  m.timeoutMs = int(std::max(500.0, std::min(120000.0, t)));
  m.pollOffTurn = v.bo("poll_off_turn", true);
  return true;
}

// ---------------------------------------------------------------- installed
struct Installed {
  std::string id;
  Manifest man;
  std::string home;        // <root>/<id>
  std::string dir;         // <root>/<id>/pkg -- the bot's own files, and its cwd
  std::string uploader;
  long long uploaded = 0;
  long long zipBytes = 0;
  std::string status;      // "" (never checked), "ok", "failed", "preparing"
  std::string note;        // last check or prepare message
  bool prepared = false;   // a virtualenv exists

  std::string venvPython() const { return dir + "/.venv/bin/python"; }
  // Three logs, kept apart because they answer different questions: what the
  // bot printed while playing, what pip said while installing it, and what the
  // last self-check found.
  std::string logPath() const { return home + "/bot.log"; }
  std::string preparePath() const { return home + "/prepare.log"; }
  std::string checkPath() const { return home + "/check.log"; }
  std::string zipPath() const { return home + "/package.zip"; }

  // What actually gets exec'd.  Two substitutions, both of them so that a
  // package can be written once and run on a machine it has never seen:
  //   * a prepared virtualenv's interpreter replaces a leading `python`, which
  //     is what puts numpy within reach of a bot that only declared it;
  //   * a run[0] containing a slash is resolved against the package directory,
  //     because the child chdir()s there and "./bot.sh" should mean the one in
  //     the package.
  std::vector<std::string> argv() const {
    std::vector<std::string> a = man.run;
    if (a.empty()) return a;
    const std::string& p0 = a[0];
    bool looksPython = p0 == "python" || p0 == "python3" || p0.rfind("python3.", 0) == 0;
    if (man.venv && looksPython && isFilePath(venvPython())) a[0] = venvPython();
    else if (p0.find('/') != std::string::npos) a[0] = dir + "/" + p0;
    return a;
  }
};

// The spec string that seats a package.  Two, because a package written against
// KV's dialect is played through the bridge already written for it (kv6.hpp)
// rather than through the native protocol, and the spec is where that is
// decided -- once, here, rather than in the factory and the table and the page.
inline std::string specFor(const Installed& b) {
  return (b.man.protocol == "kv-json-v1" ? std::string("kvbot:") : std::string("bot:")) + b.id;
}

// `bot:kraken`, `bot:id=kraken` and `bot:kraken,log=1` all name the same
// package.  factory.hpp's parseOpts drops a field with no '=', so the plain
// form has to be read off the spec itself.
inline std::string idFromSpec(const std::string& spec) {
  size_t colon = spec.find(':');
  if (colon == std::string::npos) return std::string();
  std::string rest = spec.substr(colon + 1);
  for (;;) {
    size_t comma = rest.find(',');
    std::string field = rest.substr(0, comma);
    if (field.rfind("id=", 0) == 0) return field.substr(3);
    if (!field.empty() && field.find('=') == std::string::npos) return field;
    if (comma == std::string::npos) return std::string();
    rest = rest.substr(comma + 1);
  }
}

// ------------------------------------------------------------------ registry
// One process-wide registry.  It is read from the game thread (factory.hpp
// builds an agent from a spec) and written from HTTP threads (an upload), so
// every accessor takes the lock and hands back a copy.
struct Registry {
  mutable std::mutex mu;     // guards `bots` and `root`
  std::mutex writeMu;        // serialises install / erase / prepare against each other
  std::string root;
  std::map<std::string, Installed> bots;
  bool scanned = false;
  long long rootStamp = 0;      // mtime of the bots directory at the last scan

  void setRoot(const std::string& r) {
    std::lock_guard<std::mutex> lk(mu);
    root = absolutise(r);
    scanned = false;
  }
  std::string rootDir() const {
    std::lock_guard<std::mutex> lk(mu);
    return root;
  }

  // Read <home>/meta.json (ours) and <dir>/fishbot.json (theirs) back into an
  // Installed.  The manifest is re-parsed from the package on every scan rather
  // than cached in meta.json, so editing a package on disk is picked up and a
  // package that stopped being valid stops being offered.
  bool loadOne(const std::string& id, Installed& out) const {
    Installed in;
    in.id = id;
    in.home = root + "/" + id;
    in.dir = in.home + "/pkg";
    std::string text = readWhole(in.dir + "/fishbot.json");
    if (text.empty()) return false;
    std::string err;
    if (!parseManifest(text, in.man, err)) return false;
    mj::Value meta;
    std::string metaText = readWhole(in.home + "/meta.json");
    if (!metaText.empty() && mj::parse(metaText, meta, err) && meta.isObj()) {
      in.uploader = meta.s("uploader");
      in.uploaded = (long long)meta.n("uploaded", 0);
      in.zipBytes = (long long)meta.n("zip_bytes", 0);
      in.status = meta.s("status");
      in.note = meta.s("note");
    }
    in.prepared = isFilePath(in.venvPython());
    out = in;
    return true;
  }

  // The directory's own mtime, which changes when a package is added or
  // removed.  It is what lets a running `fish serve` notice a bot installed
  // from the command line in another process, for one stat() per poll.
  long long rootMtimeLocked() const {
    struct stat st;
    if (root.empty() || ::stat(root.c_str(), &st) != 0) return 0;
    return (long long)st.st_mtime;
  }

  void rescanLocked() {
    bots.clear();
    scanned = true;
    rootStamp = rootMtimeLocked();
    if (root.empty()) return;
    DIR* d = ::opendir(root.c_str());
    if (!d) return;
    while (dirent* e = ::readdir(d)) {
      std::string n = e->d_name;
      if (n.empty() || n[0] == '.') continue;
      if (!validId(n) || !isDirPath(root + "/" + n)) continue;
      Installed in;
      if (loadOne(n, in)) bots[n] = in;
    }
    ::closedir(d);
  }
  void rescan() {
    std::lock_guard<std::mutex> lk(mu);
    rescanLocked();
  }
  void ensureScanned() {
    std::lock_guard<std::mutex> lk(mu);
    if (!scanned || rootMtimeLocked() != rootStamp) rescanLocked();
  }

  bool get(const std::string& id, Installed& out) {
    ensureScanned();
    std::lock_guard<std::mutex> lk(mu);
    auto it = bots.find(id);
    if (it == bots.end()) return false;
    out = it->second;
    return true;
  }
  bool has(const std::string& id) {
    Installed x;
    return get(id, x);
  }
  std::vector<Installed> all() {
    ensureScanned();
    std::lock_guard<std::mutex> lk(mu);
    std::vector<Installed> v;
    for (const auto& kv : bots) v.push_back(kv.second);
    return v;
  }

  void setStatus(const std::string& id, const std::string& status, const std::string& note) {
    Installed in;
    if (!get(id, in)) return;
    std::string meta = "{\"uploader\":" + mj::escape(in.uploader) +
                       ",\"uploaded\":" + std::to_string(in.uploaded) +
                       ",\"zip_bytes\":" + std::to_string(in.zipBytes) +
                       ",\"status\":" + mj::escape(status) +
                       ",\"note\":" + mj::escape(clip(note, 400)) + "}\n";
    writeWhole(in.home + "/meta.json", meta);
    std::lock_guard<std::mutex> lk(mu);
    auto it = bots.find(id);
    if (it != bots.end()) { it->second.status = status; it->second.note = clip(note, 400); }
  }

  // ------------------------------------------------------------- install
  // `replace` is required to overwrite an id that already exists, so that two
  // people uploading bots with the same name is a question rather than a
  // silent loss.  Returns the installed id.
  bool install(const std::string& zipBytes, const std::string& uploader, bool replace,
               std::string& id, std::string& err) {
    // try_lock rather than lock: prepare() holds this for as long as pip takes,
    // and an HTTP thread that blocks for two minutes is a connection the client
    // has already given up on.  Saying "busy" is the honest answer.
    std::unique_lock<std::mutex> wk(writeMu, std::try_to_lock);
    if (!wk.owns_lock()) { err = "the bot library is busy -- a dependency install is running"; return false; }
    ensureScanned();
    if (zipBytes.size() > MAX_ZIP_BYTES) {
      err = "that package is larger than " + std::to_string(MAX_ZIP_BYTES / (1024 * 1024)) + " MB";
      return false;
    }
    if (root.empty()) { err = "no bots directory is configured"; return false; }
    if (!mkdirp(root)) { err = "could not create the bots directory " + root; return false; }

    std::vector<zipf::Entry> entries;
    if (!zipf::list(zipBytes, entries, err)) return false;
    if (entries.size() > MAX_ENTRIES) { err = "that package has too many files in it"; return false; }

    size_t total = 0;
    for (const auto& e : entries) {
      if (!zipf::safeName(e.name, err)) return false;
      if (e.symlink) { err = "the package contains a symbolic link (" + e.name + "); packages must be self-contained"; return false; }
      if (e.usize > MAX_ENTRY_BYTES) { err = "the file " + e.name + " is too large"; return false; }
      total += size_t(e.usize);
      if (total > MAX_UNPACKED) { err = "the package unpacks to more than 512 MB"; return false; }
    }

    // The manifest is either at the root or one directory down -- the shape you
    // get from `zip -r mybot.zip mybot`, which is what most people will do.
    std::string prefix;
    bool found = false;
    for (const auto& e : entries) if (e.name == "fishbot.json") { found = true; break; }
    if (!found) {
      for (const auto& e : entries) {
        size_t slash = e.name.find('/');
        if (slash == std::string::npos) continue;
        if (e.name.substr(slash + 1) != "fishbot.json") continue;
        std::string pre = e.name.substr(0, slash + 1);
        if (found && pre != prefix) { err = "the package has more than one fishbot.json"; return false; }
        prefix = pre;
        found = true;
      }
    }
    if (!found) {
      err = "no fishbot.json in that zip -- a package needs its manifest at the root of "
            "the archive, or inside a single top-level directory (see docs/BOT_PACKAGE.md)";
      return false;
    }

    std::string manText;
    for (const auto& e : entries) {
      if (e.name != prefix + "fishbot.json") continue;
      if (!zipf::extract(zipBytes, e, MAX_ENTRY_BYTES, manText, err)) return false;
      break;
    }
    Manifest man;
    if (!parseManifest(manText, man, err)) return false;

    id = man.id;
    const std::string home = root + "/" + id;
    if (isDirPath(home) && !replace) {
      err = "a bot called '" + id + "' is already installed";
      return false;
    }

    // Unpack beside the destination and rename into place, so a failure halfway
    // through leaves the installed copy untouched rather than half-replaced.
    const std::string stage = root + "/.staging-" + id + "-" + std::to_string(::getpid()) +
                              "-" + std::to_string(nowSeconds());
    rmTree(stage);
    if (!mkdirp(stage + "/pkg")) { err = "could not write to " + root; return false; }
    struct Cleanup {
      std::string p;
      bool armed = true;
      ~Cleanup() { if (armed) rmTree(p); }
    } cleanup{stage, true};

    for (const auto& e : entries) {
      if (e.name.rfind(prefix, 0) != 0) continue;          // outside the package root
      std::string rel = e.name.substr(prefix.size());
      if (rel.empty()) continue;
      std::string dest = stage + "/pkg/" + rel;
      if (e.dir) { if (!mkdirp(dest)) { err = "could not create " + rel; return false; } continue; }
      size_t slash = dest.find_last_of('/');
      if (slash != std::string::npos && !mkdirp(dest.substr(0, slash))) {
        err = "could not create the directory for " + rel;
        return false;
      }
      std::string data;
      if (!zipf::extract(zipBytes, e, MAX_ENTRY_BYTES, data, err)) return false;
      if (!writeWhole(dest, data)) { err = "could not write " + rel; return false; }
      // Executability is the one permission bit that matters here -- a package
      // whose entry point is a script or a compiled binary has to be runnable --
      // so it is carried across when the archive recorded it.
      if (e.executable) ::chmod(dest.c_str(), 0755);
    }
    // ...and granted to the entry point regardless, because plenty of zips do
    // not record modes at all, and "run" naming a file in the package is a
    // statement that the file is meant to be run.  "./play" and "play" are the
    // same entry.
    if (!man.run.empty()) {
      std::string prog = man.run[0];
      if (prog.rfind("./", 0) == 0) prog = prog.substr(2);
      const std::string progPath = stage + "/pkg/" + prog;
      if (prog.find('/') != std::string::npos || isFilePath(progPath))
        if (isFilePath(progPath)) ::chmod(progPath.c_str(), 0755);
    }
    // A "run" that names a file in the package has to name one that is there.
    // Left to run time this arrives as "the bot process exited without
    // answering", which sends its author looking in the wrong place entirely.
    if (!man.run.empty() && man.run[0].find('/') != std::string::npos) {
      std::string prog = man.run[0];
      if (prog.rfind("./", 0) == 0) prog = prog.substr(2);
      if (!isFilePath(stage + "/pkg/" + prog)) {
        err = "\"run\" starts " + man.run[0] + ", which is not in the package";
        return false;
      }
    }
    if (!writeWhole(stage + "/package.zip", zipBytes)) { err = "could not keep a copy of the package"; return false; }
    std::string meta = "{\"uploader\":" + mj::escape(clip(uploader, 40)) +
                       ",\"uploaded\":" + std::to_string(nowSeconds()) +
                       ",\"zip_bytes\":" + std::to_string(zipBytes.size()) +
                       ",\"status\":\"\",\"note\":\"\"}\n";
    writeWhole(stage + "/meta.json", meta);

    const std::string old = home + ".old-" + std::to_string(::getpid());
    bool hadOld = isDirPath(home);
    if (hadOld && ::rename(home.c_str(), old.c_str()) != 0) {
      err = "could not replace the existing '" + id + "'";
      return false;
    }
    if (::rename(stage.c_str(), home.c_str()) != 0) {
      if (hadOld) ::rename(old.c_str(), home.c_str());
      err = "could not install into " + home;
      return false;
    }
    cleanup.armed = false;
    if (hadOld) rmTree(old);

    rescan();
    return true;
  }

  bool erase(const std::string& id, std::string& err) {
    std::unique_lock<std::mutex> wk(writeMu, std::try_to_lock);
    if (!wk.owns_lock()) { err = "the bot library is busy -- a dependency install is running"; return false; }
    if (!validId(id)) { err = "no such bot"; return false; }
    Installed in;
    if (!get(id, in)) { err = "no such bot"; return false; }
    if (!rmTree(in.home)) { err = "could not remove " + in.home; return false; }
    rescan();
    return true;
  }

  // --------------------------------------------------------------- prepare
  // Creates a virtualenv inside the package and installs its requirements.
  // This RUNS CODE from the package's dependency list, so it is host-only and
  // never automatic; the caller is serve.hpp's host-gated route.  The whole
  // transcript goes to bot.log and to the returned string, because "pip failed"
  // without the output is a support ticket rather than an answer.
  bool prepare(const std::string& id, std::string& log, std::string& err) {
    std::lock_guard<std::mutex> wk(writeMu);
    Installed in;
    if (!get(id, in)) { err = "no such bot"; return false; }
    if (!in.man.venv) { err = "this package does not ask for a virtualenv"; return false; }
    std::string req = in.man.requirements.empty() ? std::string("requirements.txt") : in.man.requirements;
    bool haveReq = isFilePath(in.dir + "/" + req);

    setStatus(id, "preparing", "creating a virtualenv");
    log.clear();
    // Every path here goes through the shell, and the bots root is wherever the
    // host started the server -- "GitHub/fish optimization/engine/bots" on this
    // machine -- so quoting is load-bearing rather than decorative.
    auto shq = [](const std::string& a) {
      std::string q = "'";
      for (char c : a) { if (c == '\'') q += "'\\''"; else q.push_back(c); }
      return q + "'";
    };
    auto run = [&](const std::vector<std::string>& argv, const std::string& cwd) -> int {
      std::string cmd;
      for (const auto& a : argv) cmd += shq(a) + " ";
      // 2>&1 into a pipe: the interesting half of a pip failure is on stderr.
      std::string full = "cd " + shq(cwd) + " && " + cmd + "2>&1";
      log += "$ " + cmd + "\n";
      FILE* f = ::popen(full.c_str(), "r");
      if (!f) { log += "(could not start it)\n"; return -1; }
      char buf[4096];
      while (fgets(buf, sizeof(buf), f)) {
        log += buf;
        if (log.size() > 200000) { log += "\n(output truncated)\n"; break; }
      }
      int rc = ::pclose(f);
      return WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
    };

    std::string py = in.man.run.empty() ? std::string("python3") : in.man.run[0];
    bool looksPython = py == "python" || py == "python3" || py.rfind("python3.", 0) == 0;
    if (!looksPython) py = "python3";
    rmTree(in.dir + "/.venv");
    int rc = run({py, "-m", "venv", ".venv"}, in.dir);
    if (rc != 0) {
      err = "could not create a virtualenv (is python3 installed?)";
      writeWhole(in.preparePath(), log);
      setStatus(id, "failed", err);
      return false;
    }
    if (haveReq) {
      rc = run({".venv/bin/python", "-m", "pip", "install", "--disable-pip-version-check",
                "-q", "-r", req}, in.dir);
      if (rc != 0) {
        err = "pip could not install " + req;
        writeWhole(in.preparePath(), log);
        setStatus(id, "failed", err);
        return false;
      }
    } else {
      log += "(no " + req + " in the package; the virtualenv is empty)\n";
    }
    writeWhole(in.preparePath(), log);
    setStatus(id, "", haveReq ? "dependencies installed" : "virtualenv created");
    rescan();
    return true;
  }

  // ------------------------------------------------------------------ json
  std::string json() {
    std::vector<Installed> v = all();
    std::string o = "[";
    for (size_t i = 0; i < v.size(); i++) {
      const Installed& b = v[i];
      if (i) o += ",";
      o += "{\"id\":" + mj::escape(b.id) +
           ",\"spec\":" + mj::escape(specFor(b)) +
           ",\"name\":" + mj::escape(b.man.name) +
           ",\"version\":" + mj::escape(b.man.version) +
           ",\"author\":" + mj::escape(b.man.author) +
           ",\"description\":" + mj::escape(b.man.description) +
           ",\"protocol\":" + mj::escape(b.man.protocol) +
           ",\"uploader\":" + mj::escape(b.uploader) +
           ",\"uploaded\":" + std::to_string(b.uploaded) +
           ",\"bytes\":" + std::to_string(b.zipBytes) +
           ",\"needsVenv\":" + (b.man.venv ? "true" : "false") +
           ",\"prepared\":" + (b.prepared ? "true" : "false") +
           ",\"status\":" + mj::escape(b.status) +
           ",\"note\":" + mj::escape(b.note) + "}";
    }
    return o + "]";
  }
};

inline Registry& registry() {
  static Registry r;
  // A magic static, so the default root is chosen exactly once even though the
  // first caller may be an HTTP thread and the second the game thread.
  static const bool inited = [] {
    const char* env = getenv("FISH_BOTS_DIR");
    r.setRoot(env && *env ? env : "bots");
    return true;
  }();
  (void)inited;
  return r;
}

}  // namespace botpkg
}  // namespace fish
