// Interactive table: one live game, played by any mix of humans and bots.
//
// The game runs on its own thread inside the ordinary Game driver; the HTTP
// thread only reads a public snapshot and writes into the human seats' inboxes.
// Everything shared is guarded by HumanIO::mu.
#pragma once
#include "factory.hpp"
#include "human.hpp"
#include "httpd.hpp"
#include <chrono>
#include <thread>
#include <sstream>

namespace fish {

struct SeatCfg {
  std::string spec = "v06";
  std::string name;          // display name; distinct names are what make a
  bool human = false;        // table of six bots readable at all
};

inline bool knownPolicy(const std::string& spec) {
  std::string base;
  parseOpts(spec, base);
  // An uploaded package is seatable exactly when it is installed and the spec
  // names the dialect its manifest declares.  This is the whitelist that keeps
  // /api/table from being able to construct anything the host did not put on
  // disk, so it is a registry lookup and not a pattern match.
  if (base == "bot" || base == "kvbot") {
    botpkg::Installed b;
    std::string id = botpkg::idFromSpec(spec);
    if (id.empty() || !botpkg::registry().get(id, b)) return false;
    return botpkg::specFor(b).rfind(base + ":", 0) == 0;
  }
  static const char* ok[] = {"v07", "fishbot_v07", "v06", "fishbot_v06", "v05", "fishbot_v05",
                             "v04", "fishbot_v04", "v03", "fishbot_v03", "v02", "fishbot_v02",
                             "random", "hunter", "diversifier", "detective", "lockout", "bluffer",
                             "silent", "feint", "withholder",
                             // External engines ported into this one (src/kv.hpp).
                             "kv", "kvsearch", "kv6", "kvfishbot", "kraken"};
  for (const char* n : ok) if (base == n) return true;
  return false;
}

inline std::string policyLabel(const std::string& spec) {
  std::string base;
  auto o = parseOpts(spec, base);
  std::string name;
  // A belief mode other than the shipped default is the one option worth
  // spelling out on the table, because it is a different agent to play against
  // rather than a tuning detail; the rest of the spec stays in the tooltip.
  auto beliefSuffix = [&](const char* dflt) {
    auto it = o.find("belief");
    if (it == o.end() || it->second == dflt) return std::string();
    return "-" + std::string(1, char(toupper((unsigned char)it->second[0]))) + it->second.substr(1);
  };
  if (base == "v06" || base == "fishbot_v06") {
    name = "FishBot v0.6";
    if (optI(o, "s1", 0))
      // det=16,cand=6 is the F-mid operating point -- the strongest v0.6
      // configuration measured -- and a table of six needs the two search
      // variants distinguishable at a glance.
      name = optI(o, "det", 12) >= 16 ? "FishBot v0.6-Search-Max" : "FishBot v0.6-Search";
    else if (optI(o, "legacy", 0)) name = "FishBot v0.6-legacy";
    name += beliefSuffix("fast");
  }
  // SESTINA v1.0 is the deployed name for the v0.7 configuration.  The SPEC
  // stays `v07`: every experiment script, frozen artifact and paper reference
  // is written against that string, and this is a label on a seat, not an
  // identifier.  engine/web/index.html carries the same name in its engine list.
  else if (base == "v07" || base == "fishbot_v07") name = "SESTINA v1.0";
  else if (base == "v05" || base == "fishbot_v05") name = "FishBot v0.5" + beliefSuffix("fast");
  else if (base == "v04" || base == "fishbot_v04") {
    auto it = o.find("belief");
    name = "FishBot v0.4";
    if (it != o.end() && it->second == "block") name = "FishBot v0.4-Block";
    else if (it != o.end() && it->second != "fast") name = "FishBot v0.4-" + it->second;
  }
  else if (base == "v03" || base == "fishbot_v03") name = "FishBot v0.3";
  else if (base == "v02" || base == "fishbot_v02") name = "FishBot v0.2";
  else if (base == "kv" || base == "kvsearch") {
    // KV's sampled-world search (src/kv.hpp).  The shipped settings are 96
    // particles / 48 determinizations; anything larger is his `fish analyze`
    // operating point and is worth distinguishing on the table.
    name = "KV Search";
    if (optI(o, "part", 96) > 96 || optI(o, "det", 48) > 48) name = "KV Search-Deep";
  }
  else if (base == "kv6" || base == "kvfishbot") name = "KV's FishBot v0.6";
  else if (base == "kraken") name = "KRAKEN v1.0";
  // An uploaded package names itself.  Its own manifest is the only authority
  // on what to call it, so the label is read from the registry rather than
  // guessed from the id -- "kv-fishbot-v08" on the felt would be this project
  // naming somebody else's bot for them.
  else if (base == "bot" || base == "kvbot") {
    botpkg::Installed b;
    std::string id = botpkg::idFromSpec(spec);
    if (botpkg::registry().get(id, b)) {
      name = b.man.name;
      // Only when the name does not already carry it: plenty of packages call
      // themselves "FishAI v1.0" and then set version "1.0", and "FishAI v1.0
      // 1.0" on a seat is this engine talking over its author.
      if (!b.man.version.empty() && b.man.version != "0" &&
          name.find(b.man.version) == std::string::npos)
        name += " " + b.man.version;
    } else name = id.empty() ? "Uploaded bot" : id;
  }
  else if (base.empty()) name = "Detective";
  else { name = base; name[0] = char(toupper((unsigned char)name[0])); }
  return name;
}

// ------------------------------------------------------------------ JSON out
inline std::string jesc(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    switch (c) {
      case '"':  o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n";  break;
      case '\r': o += "\\r";  break;
      case '\t': o += "\\t";  break;
      default:
        if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
        else o += c;
    }
  }
  return o + "\"";
}

struct Snap {
  bool running = false, finished = false, hitLimit = false, everStarted = false;
  int turn = 0, dealer = 0;
  int score[2] = {0, 0};
  int handCount[NPLAY] = {0,0,0,0,0,0};
  bool setActive[NSET] = {false,false,false,false,false,false,false,false,false};
  int setWinner[NSET] = {2,2,2,2,2,2,2,2,2};
  int nEvents = 0, nAsks = 0;
  std::vector<Event> hist;
  uint64_t hand[NPLAY] = {0,0,0,0,0,0};
  uint64_t dealt[NPLAY] = {0,0,0,0,0,0};
  bool reveal = false;
  std::string label[NPLAY];   // engine label, e.g. "FishBot v0.4"
  std::string name[NPLAY];    // display name, e.g. "Nova"
  std::string spec[NPLAY];
  bool isHuman[NPLAY] = {false,false,false,false,false,false};
  uint64_t seed = 0;
  int deckSets = 9;
  // Why the last game stopped, when it stopped because a seated bot broke its
  // side of the protocol.  Shown on the table rather than only in the server's
  // terminal, because the person who has to fix it is usually one of the
  // players and not the host.
  std::string fault;
};

struct Table {
  HumanIO io;
  Snap snap;
  std::thread th;
  SeatCfg seats[NPLAY];
  Rules rules;
  uint64_t seed = 0;

  static const char* defaultName(int p) {
    static const char* n[NPLAY] = {"Ari", "Vega", "Nova", "Orion", "Lyra", "Rigel"};
    return n[p];
  }
  // "You" is only ever a default for a seat somebody is actually playing.
  std::string resolvedName(int p) const {
    if (!seats[p].name.empty()) return seats[p].name;
    return seats[p].human ? std::string("You") : std::string(defaultName(p));
  }

  Table() {
    for (int p = 0; p < NPLAY; p++) { seats[p].spec = "v06"; seats[p].name = defaultName(p); }
    seats[0].human = true;
    seats[0].name = "You";
    publishConfig();
  }

  // Display names, de-duplicated.  Recomputed from the base names on every
  // publish rather than edited in place, because a name that is disambiguated
  // in place grows another suffix every time somebody joins.
  void displayNames(std::string out[NPLAY]) const {
    for (int p = 0; p < NPLAY; p++) out[p] = resolvedName(p);
    for (int p = 0; p < NPLAY; p++)
      for (int q = 0; q < p; q++)
        if (out[q] == out[p]) { out[p] += " " + std::to_string(p + 1); break; }
  }

  // Copies the seat configuration into the snapshot so that stateJson never
  // has to touch `seats` -- the game thread owns those, the HTTP thread reads
  // only the snapshot.  Callers must not hold io.mu.
  void publishConfig() {
    std::string disp[NPLAY];
    displayNames(disp);
    std::lock_guard<std::mutex> lk(io.mu);
    for (int p = 0; p < NPLAY; p++) {
      snap.spec[p] = seats[p].spec;
      snap.isHuman[p] = seats[p].human;
      snap.label[p] = seats[p].human ? std::string("Human") : policyLabel(seats[p].spec);
      snap.name[p] = disp[p];
    }
    io.bump();
  }

  // ------------------------------------------------------------- lifecycle
  void stop() {
    {
      std::lock_guard<std::mutex> lk(io.mu);
      io.abandon = true;
      io.paused = false;
      io.cv.notify_all();
    }
    if (th.joinable()) th.join();
    std::lock_guard<std::mutex> lk(io.mu);
    io.abandon = false;
    snap.running = false;
    io.bump();
  }

  void start() {
    stop();
    {
      std::lock_guard<std::mutex> lk(io.mu);
      snap = Snap{};
      snap.running = true;
      snap.everStarted = true;
      snap.seed = seed;
      snap.deckSets = rules.deckSets;
      std::string disp[NPLAY];
      displayNames(disp);
      for (int p = 0; p < NPLAY; p++) {
        snap.spec[p] = seats[p].spec;
        snap.isHuman[p] = seats[p].human;
        snap.label[p] = seats[p].human ? std::string("Human") : policyLabel(seats[p].spec);
        snap.name[p] = disp[p];
        snap.handCount[p] = rules.deckSets;
      }
      snap.hist.clear();
      for (int s = 0; s < NSET; s++) { snap.setActive[s] = (s < rules.deckSets); snap.setWinner[s] = 2; }
      for (int p = 0; p < NPLAY; p++) io.slot[p] = HumanSlot{};
      io.bump();
    }
    th = std::thread([this] { gameLoop(); });
  }

  // ------------------------------------------------------------ game thread

  // Who acts next.  Game::run moves the turn *after* the event is emitted, so
  // pub.turn still names the previous mover while an observer is running; this
  // applies the transition the driver is about to apply.
  static int nextMover(const Game& g) {
    int t = g.g.pub.turn;
    if (!g.g.pub.history.empty()) {
      const Event& e = g.g.pub.history.back();
      if ((e.kind == Kind::Ask && !e.success) || e.kind == Kind::Pass) t = e.target;
    }
    return (t >= 0 && t < NPLAY) ? t : g.g.pub.turn;
  }

  void capture(const Game& g) {
    std::lock_guard<std::mutex> lk(io.mu);
    // Display correction only: agents still observe pub.turn exactly as before.
    snap.turn = nextMover(g);
    snap.dealer = g.g.dealer;
    snap.score[0] = g.g.pub.score[0];
    snap.score[1] = g.g.pub.score[1];
    for (int p = 0; p < NPLAY; p++) {
      snap.handCount[p] = g.g.pub.handCount[p];
      // Every seat's live hand is captured; who may SEE one is decided at
      // emission (stateJson): a human's only by its own credential, a bot's
      // only via the host's all-bots x-ray (serve.hpp).
      snap.hand[p] = g.agents[p]->k.myHand;
    }
    for (int s = 0; s < NSET; s++) { snap.setActive[s] = g.g.pub.setActive[s]; snap.setWinner[s] = g.g.setWinner[s]; }
    snap.nEvents = g.g.pub.nEvents;
    snap.nAsks = g.res.asks;
    snap.hist = g.g.pub.history;
    io.bump();
  }

  // Paces the bots so a human can follow them, and honours pause / single step.
  //
  // The delay is the time a bot takes to move, so it is charged by *who acts
  // next*, not by who just moved:
  //
  //   * you ask and miss  -> a bot is next, so it replies after the delay;
  //   * you ask and hit   -> you are next, so you may ask again at once;
  //   * a bot asks you and misses -> you are next, so you are prompted at once;
  //   * a bot asks you and hits   -> it keeps the turn, so its next ask is paced.
  //
  // Once the turn is yours there is nothing to wait for: you set your own tempo
  // and can read the board for as long as you like before acting.
  //
  // The exception is the forced endgame, where nobody is "to move" in the asking
  // sense and every event is a declaration worth watching, so the delay stands.
  //
  // Pause and single-step still apply to every seat -- pausing means the table
  // is frozen, whoever moved last.
  void paceGate(const Game& g) {
    bool humanNext = false;
    if (g.g.pub.teamAlive(0) && g.g.pub.teamAlive(1))
      humanNext = seats[nextMover(g)].human;
    std::unique_lock<std::mutex> lk(io.mu);
    for (;;) {
      if (io.abandon) throw Abandoned{};
      if (!io.paused) break;
      if (io.stepBudget > 0) { io.stepBudget--; return; }
      io.cv.wait(lk);
    }
    if (humanNext) return;
    // The delay is measured against paceMs as it stands *now*, re-read on every
    // wakeup, so dragging the pace slider down from 20s to 1s takes effect at
    // once instead of after the deadline the old value had already fixed.
    auto start = std::chrono::steady_clock::now();
    for (;;) {
      if (io.abandon) throw Abandoned{};
      if (io.paused) return;                 // pausing supersedes the delay
      // A pending human declaration ends the delay at once: the rules allow
      // declaring at any moment, and the pace is presentation, not turn
      // structure.  The declaration poll at the top of the loop announces it.
      for (int p = 0; p < NPLAY; p++)
        if (io.slot[p].human && io.slot[p].haveDecl) return;
      long long ms = io.paceMs;
      long long el = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - start).count();
      if (el >= ms) return;
      io.cv.wait_for(lk, std::chrono::milliseconds(ms - el));
    }
  }

  void gameLoop() {
    try {
      std::unique_ptr<Agent> owned[NPLAY];
      Agent* ptr[NPLAY];
      for (int p = 0; p < NPLAY; p++) {
        if (seats[p].human) {
          auto h = std::make_unique<HumanAgent>();
          h->io = &io;
          h->label = resolvedName(p);
          owned[p] = std::move(h);
        } else {
          owned[p] = makeAgent(seats[p].spec);
        }
        ptr[p] = owned[p].get();
      }
      Game game;
      game.observer = [this](const Game& g) { capture(g); paceGate(g); };
      // setup() is idempotent for a fixed seed; running it here just gives the
      // browser the opening position before run() blocks on the first decision.
      game.setup(seed, rules, ptr);
      capture(game);
      GameResult r = game.run(seed, rules, ptr);
      std::lock_guard<std::mutex> lk(io.mu);
      snap.finished = true;
      snap.running = false;
      snap.reveal = true;
      snap.hitLimit = r.hitLimit;
      snap.score[0] = r.score[0];
      snap.score[1] = r.score[1];
      snap.nAsks = r.asks;
      snap.nEvents = game.g.pub.nEvents;
      snap.hist = game.g.pub.history;
      for (int p = 0; p < NPLAY; p++) { snap.dealt[p] = game.g.dealt[p]; snap.handCount[p] = game.g.pub.handCount[p]; }
      for (int s = 0; s < NSET; s++) { snap.setActive[s] = false; snap.setWinner[s] = game.g.setWinner[s]; }
      for (int p = 0; p < NPLAY; p++) io.slot[p].need = Need::None;
      io.bump();
    } catch (const BotFault& f) {
      std::lock_guard<std::mutex> lk(io.mu);
      snap.running = false;
      snap.fault = f.text;
      // Whoever was waiting on a prompt is no longer being asked for anything.
      for (int p = 0; p < NPLAY; p++) io.slot[p].need = Need::None;
      fprintf(stderr, "fish serve: %s\n", f.what());
      io.bump();
    } catch (const Abandoned&) {
      std::lock_guard<std::mutex> lk(io.mu);
      snap.running = false;
      io.bump();
    }
  }

  // ------------------------------------------------------------------- JSON
  static void jarr(std::ostringstream& os, const uint64_t mask) {
    os << "[";
    bool first = true;
    for (int c = 0; c < NCARD; c++) if (mask & bit(c)) { if (!first) os << ","; os << c; first = false; }
    os << "]";
  }

  // `extra` is a pre-rendered ",\"k\":v" run appended before the closing brace.
  // Who is asking is a question for the lobby, not for the game, so those
  // fields are composed by the caller rather than reaching in here.
  std::string stateJson(int viewSeat, const std::string& extra = std::string(), bool xray = false) {
    std::lock_guard<std::mutex> lk(io.mu);
    std::ostringstream os;
    os << "{\"rev\":" << io.rev
       << ",\"running\":" << (snap.running ? "true" : "false")
       << ",\"finished\":" << (snap.finished ? "true" : "false")
       << ",\"everStarted\":" << (snap.everStarted ? "true" : "false")
       << ",\"hitLimit\":" << (snap.hitLimit ? "true" : "false")
       << ",\"fault\":" << (snap.fault.empty() ? std::string("null") : jesc(snap.fault))
       << ",\"paused\":" << (io.paused ? "true" : "false")
       << ",\"pace\":" << io.paceMs
       << ",\"seed\":" << snap.seed
       << ",\"deckSets\":" << snap.deckSets
       << ",\"turn\":" << snap.turn
       << ",\"dealer\":" << snap.dealer
       << ",\"nAsks\":" << snap.nAsks
       << ",\"nEvents\":" << snap.nEvents
       << ",\"score\":[" << snap.score[0] << "," << snap.score[1] << "]";

    os << ",\"handCount\":[";
    for (int p = 0; p < NPLAY; p++) { if (p) os << ","; os << snap.handCount[p]; }
    os << "],\"setActive\":[";
    for (int s = 0; s < NSET; s++) { if (s) os << ","; os << (snap.setActive[s] ? "true" : "false"); }
    os << "],\"setWinner\":[";
    for (int s = 0; s < NSET; s++) { if (s) os << ","; os << snap.setWinner[s]; }
    os << "]";

    os << ",\"seats\":[";
    for (int p = 0; p < NPLAY; p++) {
      if (p) os << ",";
      os << "{\"i\":" << p << ",\"team\":" << teamOf(p)
         << ",\"name\":" << jesc(snap.name[p])
         << ",\"label\":" << jesc(snap.label[p])
         << ",\"spec\":" << jesc(snap.spec[p])
         << ",\"human\":" << (snap.isHuman[p] ? "true" : "false") << "}";
    }
    os << "]";

    os << ",\"history\":[";
    for (size_t i = 0; i < snap.hist.size(); i++) {
      const Event& e = snap.hist[i];
      if (i) os << ",";
      const char* kind = e.kind == Kind::Ask ? "ask"
                       : e.kind == Kind::Declare ? "declare"
                       : e.kind == Kind::ForcedDeclare ? "forced"
                       : e.kind == Kind::Pass ? "pass" : "end";
      os << "{\"k\":\"" << kind << "\",\"a\":" << int(e.actor)
         << ",\"t\":" << int(e.target) << ",\"c\":" << int(e.card)
         << ",\"s\":" << int(e.set) << ",\"ok\":" << (e.success ? "true" : "false");
      if (e.kind == Kind::Declare || e.kind == Kind::ForcedDeclare) {
        os << ",\"o\":[";
        for (int j = 0; j < SETSZ; j++) { if (j) os << ","; os << int(e.decl.owner[j]); }
        os << "]";
      }
      os << ",\"n\":[";
      for (int p = 0; p < NPLAY; p++) { if (p) os << ","; os << int(e.handCount[p]); }
      os << "]}";
    }
    os << "]";

    // Spectator x-ray: every live hand, emitted only when serve.hpp has
    // established that the requester is the host and no seat is a person's.
    if (xray) {
      os << ",\"hands\":[";
      for (int p = 0; p < NPLAY; p++) { if (p) os << ","; jarr(os, snap.hand[p]); }
      os << "]";
    }

    if (snap.reveal) {
      os << ",\"deal\":[";
      for (int p = 0; p < NPLAY; p++) { if (p) os << ","; jarr(os, snap.dealt[p]); }
      os << "]";
    }

    if (viewSeat >= 0 && viewSeat < NPLAY && snap.isHuman[viewSeat]) {
      const HumanSlot& sl = io.slot[viewSeat];
      os << ",\"you\":{\"seat\":" << viewSeat << ",\"hand\":";
      jarr(os, snap.hand[viewSeat]);
      os << ",\"need\":\"" << needName(sl.need) << "\""
         << ",\"needSet\":" << sl.needSet
         << ",\"deferAll\":" << (sl.forcedDeferAll ? "true" : "false")
         << ",\"note\":" << jesc(sl.note)
         << ",\"cands\":[";
      for (int i = 0; i < sl.nCands; i++) { if (i) os << ","; os << sl.cands[i]; }
      os << "]";
      if (sl.haveDecl) {
        os << ",\"queued\":{\"set\":" << int(sl.decl.set) << ",\"owner\":[";
        for (int j = 0; j < SETSZ; j++) { if (j) os << ","; os << int(sl.decl.owner[j]); }
        os << "]}";
      } else os << ",\"queued\":null";
      os << "}";
    } else os << ",\"you\":null";

    os << extra;
    os << "}";
    return os.str();
  }
};

} // namespace fish
