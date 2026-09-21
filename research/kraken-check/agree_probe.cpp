// Decision-agreement probe (standalone; the repository's engine is not modified).
// Native SESTINA plays in FishLab's arbiter under KV's dialect (declare on turn only); at every
// on-turn decision the same position is ALSO serialised exactly the way KV's bridge
// (fish4/dylan_v07.py) serialises it and fed to KV's shim binary, once with the CURRENT hand
// (their rev-2 behaviour) and once with the hand as DEALT. The shim's answers are compared with
// the native agent's, and hypothetical declarations are scored against the true hands.
//   clang++ -std=c++20 -O2 -w -I engine/src -o agree_probe agree_probe.cpp -pthread
//   (from engine/)  ../research/kraken-check/agree_probe --deals=60 --opp=kraken --spec=<allparamsSpec> --shim=<fish_v07_decide> --tmp=<dir>
#include "fish.hpp"
#include "game.hpp"
#include "factory.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <memory>
#include <unistd.h>
using namespace fish;

static std::string SPEC_FULL, SHIM, TMP;

struct Stats {
  long onTurn = 0, innerDecl = 0, curDecl = 0, dealtDecl = 0;
  long curAgreeDecl = 0, dealtAgreeDecl = 0;
  long curOnly = 0, curOnlyWrong = 0, curOnlyOwnWrong = 0;
  long dealtOnly = 0, dealtOnlyWrong = 0, dealtOnlyOwnWrong = 0;
  long innerOnlyCur = 0, innerOnlyDealt = 0;
  long bothDiffCur = 0, bothDiffDealt = 0;
  long innerDeclWrong = 0, curDeclWrong = 0, dealtDeclWrong = 0;
  long askCmp = 0, askAgreeCur = 0, askAgreeDealt = 0;
  long first = 0, firstAgreeCur = 0, firstAgreeDealt = 0;
  long shimErrors = 0;
};

static std::string trim(std::string s) { while (!s.empty() && (s.back()=='\n' || s.back()=='\r' || s.back()==' ')) s.pop_back(); return s; }

struct Probe : Agent {
  std::unique_ptr<Agent> inner;
  Game* game = nullptr;
  Stats* st = nullptr;
  uint64_t dealt = 0, sd = 1;
  std::vector<std::string> ev;
  int score[2] = {0, 0};
  int cachedEvents = -1; std::string curOut, dealtOut; bool firstPending = true;

  const char* name() const override { return inner->name(); }
  void reset(int s, uint64_t hand, const Rules& r, uint64_t seed) override {
    Agent::reset(s, hand, r, seed); inner->reset(s, hand, r, seed);
    // The shim reads SEED as a uint64, so the native agent's exact seed is sent.  (KV's bridge masks
    // its own seed to 63 bits; that convention is theirs and is not what is under test here.)
    dealt = hand; sd = seed;
    ev.clear(); score[0] = score[1] = 0; cachedEvents = -1; firstPending = true;
  }
  void resetWithKnowledge(int s, uint64_t hand, const Rules& r, uint64_t seed, const Knowledge& k0) override {
    reset(s, hand, r, seed); k = k0; inner->resetWithKnowledge(s, hand, r, seed, k0);
  }
  void observe(const Event& e) override {
    Agent::observe(e); inner->observe(e);
    char buf[256]; std::string hc;
    for (int p = 0; p < NPLAY; p++) hc += " " + std::to_string(int(e.handCount[p]));
    if (e.kind == Kind::Ask) {
      snprintf(buf, sizeof buf, "EV ASK %d %d %d %d", int(e.actor), int(e.target), int(e.card), e.success ? 1 : 0);
      ev.push_back(std::string(buf) + hc);
    } else if (e.kind == Kind::Declare || e.kind == Kind::ForcedDeclare) {
      int team = teamOf(e.actor); int winner = e.success ? team : 1 - team; score[winner]++;
      snprintf(buf, sizeof buf, "EV DECL %d %d %d", int(e.actor), int(e.decl.set), e.success ? 1 : 0);
      std::string o; for (int i = 0; i < SETSZ; i++) o += " " + std::to_string(int(e.decl.owner[i]));
      ev.push_back(std::string(buf) + o + hc + " " + std::to_string(score[0]) + " " + std::to_string(score[1]));
    } else if (e.kind == Kind::Pass) {
      snprintf(buf, sizeof buf, "EV PASS %d %d", int(e.actor), int(e.target));
      ev.push_back(std::string(buf) + hc);
    }
  }
  void observeResolution(int set, const uint8_t* o) override { inner->observeResolution(set, o); }
  std::string callShim(uint64_t hand, const PublicState& pub) {
    std::string in = "SPEC " + SPEC_FULL + "\nRULES 9 0 1\nSEAT " + std::to_string(seat) + "\nHAND " +
                     std::to_string(hand) + "\nSEED " + std::to_string(sd) + "\n";
    for (auto& l : ev) in += l + "\n";
    in += "TURN " + std::to_string(pub.turn) + "\nDECIDE TURN\n";
    FILE* f = fopen(TMP.c_str(), "w"); fwrite(in.data(), 1, in.size(), f); fclose(f);
    std::string cmd = "'" + SHIM + "' < '" + TMP + "' 2>/dev/null";
    FILE* p = popen(cmd.c_str(), "r"); char buf[512]; std::string out;
    if (p) { if (fgets(buf, sizeof buf, p)) out = buf; pclose(p); }
    out = trim(out);
    if (out.empty()) st->shimErrors++;
    return out;
  }
  static bool parseDecl(const std::string& s, Declaration& d) {
    if (s.rfind("DECL ", 0) != 0) return false;
    int set, o[6]; if (sscanf(s.c_str(), "DECL %d %d %d %d %d %d %d", &set, &o[0], &o[1], &o[2], &o[3], &o[4], &o[5]) != 7) return false;
    d.set = uint8_t(set); for (int i = 0; i < 6; i++) d.owner[i] = uint8_t(o[i]); return true;
  }
  static bool parseAsk(const std::string& s, int& target, int& card) {
    return s.rfind("ASK ", 0) == 0 && sscanf(s.c_str(), "ASK %d %d", &target, &card) == 2;
  }
  void scoreDecl(const Declaration& d, bool& right, bool& ownWrong) {
    right = true; ownWrong = false;
    for (int i = 0; i < SETSZ; i++) {
      int c = cardOf(d.set, i); int holder = -1;
      for (int p = 0; p < NPLAY; p++) if (game->g.hand[p] & bit(c)) holder = p;
      if (holder != int(d.owner[i])) right = false;
      if (holder >= 0 && teamOf(holder) != teamOf(seat)) ownWrong = true;
    }
  }
  static bool sameDecl(const Declaration& a, const Declaration& b) {
    if (a.set != b.set) return false; for (int i = 0; i < SETSZ; i++) if (a.owner[i] != b.owner[i]) return false; return true;
  }
  bool proposeDeclaration(const PublicState& pub, Declaration& d, double& conf) override {
    bool r = inner->proposeDeclaration(pub, d, conf);
    if (pub.turn != seat) return r;
    st->onTurn++;
    curOut = callShim(k.myHand, pub); dealtOut = callShim(dealt, pub); cachedEvents = pub.nEvents;
    Declaration dc{}, dd{}; bool cIs = parseDecl(curOut, dc), dIs = parseDecl(dealtOut, dd);
    if (r) { st->innerDecl++; bool rt, ow; scoreDecl(d, rt, ow); if (!rt) st->innerDeclWrong++; }
    if (cIs) { st->curDecl++; bool rt, ow; scoreDecl(dc, rt, ow); if (!rt) st->curDeclWrong++; }
    if (dIs) { st->dealtDecl++; bool rt, ow; scoreDecl(dd, rt, ow); if (!rt) st->dealtDeclWrong++; }
    auto cmp = [&](bool sIs, const Declaration& sd_, long& agree, long& only, long& onlyWrong, long& onlyOwn, long& innerOnly, long& bothDiff) {
      if (r && sIs) { if (sameDecl(d, sd_)) agree++; else bothDiff++; }
      else if (!r && !sIs) agree++;
      else if (sIs) { only++; bool rt, ow; scoreDecl(sd_, rt, ow); if (!rt) onlyWrong++; if (ow) onlyOwn++; }
      else innerOnly++;
    };
    cmp(cIs, dc, st->curAgreeDecl, st->curOnly, st->curOnlyWrong, st->curOnlyOwnWrong, st->innerOnlyCur, st->bothDiffCur);
    cmp(dIs, dd, st->dealtAgreeDecl, st->dealtOnly, st->dealtOnlyWrong, st->dealtOnlyOwnWrong, st->innerOnlyDealt, st->bothDiffDealt);
    if (firstPending && r) {
      firstPending = false; st->first++;
      if (cIs && sameDecl(d, dc)) st->firstAgreeCur++; else fprintf(stderr, "FIRST-DISAGREE(cur) seat %d decl native=set%d shim=%s events=%zu\n", seat, int(d.set), curOut.c_str(), ev.size());
      if (dIs && sameDecl(d, dd)) st->firstAgreeDealt++; else fprintf(stderr, "FIRST-DISAGREE(dealt) seat %d decl native=set%d shim=%s events=%zu\n", seat, int(d.set), dealtOut.c_str(), ev.size());
    }
    return r;
  }
  AskMove chooseAsk(const PublicState& pub) override {
    AskMove m = inner->chooseAsk(pub);
    if (cachedEvents == pub.nEvents && pub.turn == seat) {
      int t, c; st->askCmp++;
      bool ac = parseAsk(curOut, t, c) && t == m.target && c == m.card; if (ac) st->askAgreeCur++;
      bool ad = parseAsk(dealtOut, t, c) && t == m.target && c == m.card; if (ad) st->askAgreeDealt++;
      if (firstPending) { firstPending = false; st->first++;
        if (ac) st->firstAgreeCur++; else fprintf(stderr, "FIRST-DISAGREE(cur) seat %d ask native=%d/%d shim=%s events=%zu\n", seat, int(m.target), int(m.card), curOut.c_str(), ev.size());
        if (ad) st->firstAgreeDealt++; else fprintf(stderr, "FIRST-DISAGREE(dealt) seat %d ask native=%d/%d shim=%s events=%zu\n", seat, int(m.target), int(m.card), dealtOut.c_str(), ev.size()); }
    }
    return m;
  }
  int choosePassTarget(const PublicState& pub, const int* cand, int n) override { return inner->choosePassTarget(pub, cand, n); }
  bool willingForced(const PublicState& pub, int set, Declaration& d, double& conf, double th) override { return inner->willingForced(pub, set, d, conf, th); }
  void bestGuess(const PublicState& pub, int set, Declaration& d, double& conf) override { inner->bestGuess(pub, set, d, conf); }
  const DecisionInfo* lastDecision() const override { return inner->lastDecision(); }
  double lastAskForecast() const override { return inner->lastAskForecast(); }
};

static std::string arg(int argc, char** argv, const char* key, const char* dflt) {
  std::string k = std::string("--") + key + "=";
  for (int i = 1; i < argc; i++) { std::string a = argv[i]; if (a.rfind(k, 0) == 0) return a.substr(k.size()); }
  return dflt;
}

int main(int argc, char** argv) {
  int deals = atoi(arg(argc, argv, "deals", "50").c_str());
  uint64_t seed = strtoull(arg(argc, argv, "seed", "31337").c_str(), nullptr, 10);
  std::string opp = arg(argc, argv, "opp", "kraken");
  SPEC_FULL = arg(argc, argv, "spec", ""); SHIM = arg(argc, argv, "shim", "");
  TMP = arg(argc, argv, "tmp", "/tmp") + "/agree_probe_" + std::to_string(getpid()) + ".txt";
  Rules rules; rules.outOfTurnDeclare = false; rules.cardlessMayDeclare = true;
  Stats st;
  // --fresh=1 rebuilds the native agents for every game, as FishLab's zero-tolerance gate mode does,
  // so that no per-agent state can survive from one deal to the next (SESTINA's report, section 12.6).
  bool fresh = arg(argc, argv, "fresh", "0") == "1";
  std::unique_ptr<Probe> A[3]; std::unique_ptr<Agent> B[3];
  for (int i = 0; i < 3; i++) { A[i] = std::make_unique<Probe>(); A[i]->inner = makeAgent(SPEC_FULL); A[i]->st = &st; B[i] = makeAgent(opp); }
  Game game;
  for (int i = 0; i < 3; i++) A[i]->game = &game;
  long setsA = 0, setsB = 0, games = 0, winsA = 0, declA = 0, declAok = 0;
  for (int i = 0; i < deals; i++) {
    uint64_t s = mixSeed(seed, uint64_t(i) * 2654435761ull + 1);
    for (int rot = 0; rot < 2; rot++) {
      if (fresh) for (int q = 0; q < 3; q++) A[q]->inner = makeAgent(SPEC_FULL);
      Agent* ag[NPLAY];
      for (int p = 0; p < NPLAY; p++) ag[p] = (teamOf(p) == rot) ? (Agent*)A[p / 2].get() : B[p / 2].get();
      game.rotation = 0;
      GameResult r = game.run(s, rules, ag);
      int teamA = rot, teamB = 1 - rot;
      setsA += r.score[teamA]; setsB += r.score[teamB]; games++; if (r.winner == teamA) winsA++;
      declA += r.decls[teamA]; declAok += r.correctDecls[teamA];
    }
    if ((i + 1) % 10 == 0) fprintf(stderr, "deal %d/%d  onTurn=%ld curOnly=%ld dealtOnly=%ld\n", i + 1, deals, st.onTurn, st.curOnly, st.dealtOnly);
  }
  printf("{\"games\":%ld,\"setsNative\":%ld,\"setsOpp\":%ld,\"winsNative\":%ld,\"declNative\":%ld,\"declNativeOk\":%ld,"
         "\"onTurnDecisions\":%ld,\"innerDecl\":%ld,\"innerDeclWrong\":%ld,"
         "\"cur\":{\"decl\":%ld,\"declWrong\":%ld,\"agreeDecl\":%ld,\"shimOnly\":%ld,\"shimOnlyWrong\":%ld,\"shimOnlyOwnershipWrong\":%ld,\"nativeOnly\":%ld,\"bothDiffer\":%ld,\"askAgree\":%ld},"
         "\"dealt\":{\"decl\":%ld,\"declWrong\":%ld,\"agreeDecl\":%ld,\"shimOnly\":%ld,\"shimOnlyWrong\":%ld,\"shimOnlyOwnershipWrong\":%ld,\"nativeOnly\":%ld,\"bothDiffer\":%ld,\"askAgree\":%ld},"
         "\"askCompared\":%ld,\"firstDecisions\":%ld,\"firstAgreeCur\":%ld,\"firstAgreeDealt\":%ld,\"shimErrors\":%ld}\n",
         games, setsA, setsB, winsA, declA, declAok, st.onTurn, st.innerDecl, st.innerDeclWrong,
         st.curDecl, st.curDeclWrong, st.curAgreeDecl, st.curOnly, st.curOnlyWrong, st.curOnlyOwnWrong, st.innerOnlyCur, st.bothDiffCur, st.askAgreeCur,
         st.dealtDecl, st.dealtDeclWrong, st.dealtAgreeDecl, st.dealtOnly, st.dealtOnlyWrong, st.dealtOnlyOwnWrong, st.innerOnlyDealt, st.bothDiffDealt, st.askAgreeDealt,
         st.askCmp, st.first, st.firstAgreeCur, st.firstAgreeDealt, st.shimErrors);
  unlink(TMP.c_str());
  return 0;
}
