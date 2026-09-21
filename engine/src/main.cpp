#include "arena.hpp"
#include "tuner.hpp"
#include "blockdp.hpp"
#include "oracle.hpp"
// The interactive table pulls in the HTTP server, the lobby and the tunnel.
// Those are another workstream's files and are edited concurrently; the v0.7
// instrument batteries must stay buildable while they are mid-flight, so the
// dependency is guarded.  Default behaviour is unchanged -- `-DFISH_NO_SERVE`
// is opt-in and only ever used by the phase-1 battery's own build.
#ifndef FISH_NO_SERVE
#include "serve.hpp"
#endif
// The bot library itself is not part of the server: `fish bots` manages it from
// the command line, and `fish match --a=bot:<id>` measures one, both in a build
// with -DFISH_NO_SERVE.
#include "botcheck.hpp"
// The inverse of the bot bridge: this engine's own policies, published over the
// same protocol an uploaded package speaks.
#include "botserve.hpp"
#include "diag.hpp"
#include "probe_deadlock.hpp"
#include "probe_vdeadlock.hpp"
#include "probe_coordination.hpp"
#include "probe_turnxfer.hpp"
#include "probe_deception_run.hpp"   // appended: P3
#include "probe_valuefn.hpp"
#include "probe_forcedendgame.hpp"
#include "probe_human.hpp"           // appended: P5
#include "probe_policy.hpp"          // appended: P4
#include "probe_literature.hpp"      // appended: P-lit
#include "probe_verifyforced.hpp"    // appended: adversarial check of the forced ladder
#include "probe_passverify.hpp"     // appended: adversarial verify (turn-transfer)
#include "probe_polreview.hpp"      // appended: adversarial verify (P4 policy review)
#include "probe_vpolicy.hpp"        // appended: adversarial verify (P4/D1 forcing horizon)
#include "probe_declcard.hpp"       // appended: adversarial verify (declareByValue card delta)
#include "probe_v06.hpp"            // v0.6 diagnostics: ties, belief-as-predictor
#include "v07_probe.hpp"            // v0.7 phase-1 instrument drivers
#include "v07_side.hpp"             // v0.7 phase-3 mechanical side-channel gate
#include "v07_learn_run.hpp"        // v0.7 phase-3 K5: the amortised (learned) policy
#include "v07_leaffit.hpp"          // v0.7 phase-3 K1: fitting the search leaf
#include <chrono>
#include <fstream>
#include <iostream>

using namespace fish;

static std::string argVal(int argc, char** argv, const char* key, const char* dflt) {
  std::string k = std::string("--") + key + "=";
  for (int i = 1; i < argc; i++) { std::string a = argv[i];
    if (a.rfind(k, 0) == 0) return a.substr(k.size()); }
  return dflt;
}
static bool argFlag(int argc, char** argv, const char* key) {
  std::string k = std::string("--") + key;
  for (int i = 1; i < argc; i++) if (std::string(argv[i]) == k) return true;
  return false;
}

static Rules rulesFrom(int argc, char** argv) {
  Rules r;
  { std::string arb = argVal(argc, argv, "arb", "low");
    r.declArbitration = arb == "high" ? 1 : (arb == "turn" ? 2 : 0); }
  if (argFlag(argc, argv, "legacy")) {
    r.outOfTurnDeclare = false;
    r.cardlessMayDeclare = false;
    r.maxAsks = 360;
    r.nForcedTh = 2; r.forcedTh[0] = 0.38; r.forcedTh[1] = -1.0;
    for (int i = 2; i < 8; i++) r.forcedTh[i] = -1.0;
  }
  r.deckSets = atoi(argVal(argc, argv, "sets", "9").c_str());
  r.maxAsks = atoi(argVal(argc, argv, "maxasks", std::to_string(r.maxAsks).c_str()).c_str());
  if (argFlag(argc, argv, "no-out-of-turn")) r.outOfTurnDeclare = false;
  if (argFlag(argc, argv, "no-cardless-declare")) r.cardlessMayDeclare = false;
  return r;
}

static void printMatch(const MatchStats& st, const std::string& a, const std::string& b, bool json, std::ostream& os,
                       int rotations = 2, bool mirror = false) {
  int n = st.games * rotations;
  double lo, hi; wilson(st.winsA, n, lo, hi);
  double wr = n ? double(st.winsA) / n : 0;
  // The rotations of one deal are a single correlated cluster, so the interval
  // that is reported has to resample deals, not games.
  double bm, blo, bhi;
  clusterBootstrap(st.paired, rotations, bm, blo, bhi);
  auto acc = [](long long c, long long t) { return t ? double(c) / double(t) : 0.0; };
  if (json) {
    os << "{\"a\":\"" << a << "\",\"b\":\"" << b << "\",\"deals\":" << st.games
       << ",\"games\":" << n << ",\"winRateA\":" << wr
       << ",\"ci\":[" << blo << "," << bhi << "]"
       << ",\"wilsonCI\":[" << lo << "," << hi << "]"
       << ",\"meanSetsA\":" << (st.games ? double(st.sets[0]) / n : 0)
       << ",\"meanSetsB\":" << (st.games ? double(st.sets[1]) / n : 0)
       << ",\"askAccA\":" << acc(st.hits[0], st.asks[0])
       << ",\"askAccB\":" << acc(st.hits[1], st.asks[1])
       << ",\"declAccA\":" << acc(st.declCorrect[0], st.decl[0])
       << ",\"declAccB\":" << acc(st.declCorrect[1], st.decl[1])
       // v0.7 phase 3 (K4).  The DENOMINATORS of the per-decision objectives.
       // Ledger L5's whole arithmetic is decisions-per-game, and no artifact in
       // the corpus has ever printed the ask count, so the effective sample of a
       // per-decision estimator could not be checked against its binomial
       // baseline.  Printed here, plus the allocation-error share (L1's class).
       << ",\"asksPerGameA\":" << (n ? double(st.asks[0]) / n : 0)
       << ",\"asksPerGameB\":" << (n ? double(st.asks[1]) / n : 0)
       << ",\"nAsksA\":" << st.asks[0] << ",\"nDeclA\":" << st.decl[0]
       << ",\"nAsksB\":" << st.asks[1] << ",\"nDeclB\":" << st.decl[1]
       << ",\"allocErrRateA\":" << acc(st.declAllocErr[0], st.decl[0])
       << ",\"allocErrRateB\":" << acc(st.declAllocErr[1], st.decl[1])
       << ",\"declPerGameA\":" << (n ? double(st.decl[0]) / n : 0)
       << ",\"declPerGameB\":" << (n ? double(st.decl[1]) / n : 0)
       << ",\"forcedAccA\":" << acc(st.fdeclCorrect[0], st.fdecl[0])
       << ",\"forcedAccB\":" << acc(st.fdeclCorrect[1], st.fdecl[1])
       << ",\"forcedPerGameA\":" << (n ? double(st.fdecl[0]) / n : 0)
       << ",\"outOfTurnA\":" << (n ? double(st.outOfTurn[0]) / n : 0)
       << ",\"outOfTurnB\":" << (n ? double(st.outOfTurn[1]) / n : 0)
       << ",\"lockHoldA\":" << (st.lockedDecls[0] ? double(st.lockHeld[0]) / st.lockedDecls[0] : 0)
       << ",\"lockHoldB\":" << (st.lockedDecls[1] ? double(st.lockHeld[1]) / st.lockedDecls[1] : 0)
       << ",\"eventsPerGame\":" << (n ? double(st.events) / n : 0)
       << ",\"limitHitRate\":" << (n ? double(st.limitHits) / n : 0)
       << ",\"auditViolations\":" << st.auditViolations
       << ",\"auditChecks\":" << st.auditChecks
       << ",\"seconds\":" << st.seconds
       << ",\"gamesPerSec\":" << st.gamesPerSec(rotations)
       << ",\"threads\":" << st.threadsUsed
       << "," << powerJson(powerLine(n, st.games, rotations, mirror)) << "}";
  } else {
    os << a << " vs " << b << "\n";
    os << "  win rate      " << 100 * wr << "%  [" << 100 * lo << ", " << 100 * hi << "]  n=" << n << "\n";
    os << "  mean sets     " << double(st.sets[0]) / n << " - " << double(st.sets[1]) / n << "\n";
    os << "  ask accuracy  " << 100 * acc(st.hits[0], st.asks[0]) << "% / " << 100 * acc(st.hits[1], st.asks[1]) << "%\n";
    os << "  declarations  " << double(st.decl[0]) / n << "/game at " << 100 * acc(st.declCorrect[0], st.decl[0]) << "%"
       << "   opp " << double(st.decl[1]) / n << "/game at " << 100 * acc(st.declCorrect[1], st.decl[1]) << "%\n";
    os << "  forced decls  " << double(st.fdecl[0]) / n << "/game at " << 100 * acc(st.fdeclCorrect[0], st.fdecl[0]) << "%\n";
    os << "  out-of-turn   " << double(st.outOfTurn[0]) / n << " / " << double(st.outOfTurn[1]) / n << " per game\n";
    os << "  lock hold     " << (st.lockedDecls[0] ? double(st.lockHeld[0]) / st.lockedDecls[0] : 0)
       << " / " << (st.lockedDecls[1] ? double(st.lockHeld[1]) / st.lockedDecls[1] : 0) << " events before cashing\n";
    os << "  events/game   " << double(st.events) / n << "   limit hits " << 100.0 * st.limitHits / n << "%\n";
    if (st.auditChecks) os << "  audit         " << st.auditViolations << " violations in " << st.auditChecks << " checks\n";
    os << "  elapsed       " << st.seconds << "s  (" << (n / std::max(1e-9, st.seconds))
       << " games/s on " << st.threadsUsed << " threads)\n";
    os << powerText(powerLine(n, st.games, rotations, mirror)) << "\n";
  }
}

#include "kv_parity.hpp"

int main(int argc, char** argv) {
  std::string cmd = argc > 1 ? argv[1] : "help";
  int threads = atoi(argVal(argc, argv, "threads", "0").c_str());

  // Test-only: see src/kv_parity.hpp.  Constructs no game.
  if (cmd == "kvparity") {
    return kvsearch::parityMain(argVal(argc, argv, "file", "").c_str());
  }

  if (cmd == "match") {
    MatchConfig mc;
    mc.specA = argVal(argc, argv, "a", "v04");
    mc.specB = argVal(argc, argv, "b", "v03");
    mc.partnersA = argVal(argc, argv, "partners", "");
    mc.partnersB = argVal(argc, argv, "partnersb", "");
    mc.games = atoi(argVal(argc, argv, "games", "1000").c_str());
    mc.seed = strtoull(argVal(argc, argv, "seed", "20260821").c_str(), nullptr, 10);
mc.freshAgents = argFlag(argc, argv, "freshagents");  // diagnostic, see MatchConfig
        mc.rules = rulesFrom(argc, argv);
    mc.audit = argFlag(argc, argv, "audit");
    mc.threads = threads;
    mc.rotations = atoi(argVal(argc, argv, "rotations", "2").c_str());
    { std::string sh = argVal(argc, argv, "shard", "");
      if (!sh.empty()) { auto sl = sh.find('/');
        if (sl != std::string::npos) { mc.shard = atoi(sh.substr(0, sl).c_str());
                                       mc.shards = std::max(1, atoi(sh.substr(sl + 1).c_str())); } } }
    mc.correlated = argFlag(argc, argv, "correlated");
    MatchStats st = runMatch(mc);
    bool json = argFlag(argc, argv, "json");
    // v0.7 phase 4.  A cell is a mirror only if BOTH TEAMS are the same three
    // policies.  `specA == specB` alone flagged `--a=v06 --partners=v03 --b=v06`
    // -- a one-seat deviation column running at 31.7% -- as a mirror, and printed
    // its interval as [0,0].  Any consumer that skips a "mirror" cell as carrying
    // no information silently dropped a real measurement.
    printMatch(st, mc.specA, mc.specB, json, std::cout, mc.rotations,
               mc.specA == mc.specB && mc.partnersA == mc.partnersB);
    double m, lo, hi;
    clusterBootstrap(st.paired, mc.rotations, m, lo, hi);
    if (!json) std::cout << "  cluster boot  " << 100 * m << "% [" << 100 * lo << ", " << 100 * hi << "]\n";
    std::cout << std::endl;
    return 0;
  }

  if (cmd == "verify") {
    int games = atoi(argVal(argc, argv, "games", "200").c_str());
    const char* pool[] = {"v04", "v03", "v02", "lockout", "detective", "hunter",
                          "diversifier", "bluffer", "random"};
    int np = 9;
    long long viol = 0, checks = 0, limit = 0, bad = 0;
    for (int i = 0; i < np; i++) for (int j = 0; j < np; j++) {
      MatchConfig mc;
      mc.specA = pool[i]; mc.specB = pool[j];
      mc.games = std::max(1, games / (np * np));
      mc.seed = 991 + i * 31 + j;
      mc.rules = rulesFrom(argc, argv);
      mc.audit = true; mc.threads = threads;
      MatchStats st = runMatch(mc);
      viol += st.auditViolations; checks += st.auditChecks; limit += st.limitHits;
      if (st.sets[0] + st.sets[1] != (long long)st.games * 2 * mc.rules.deckSets) bad++;
    }
    std::cout << "audit violations: " << viol << " / " << checks << " checks\n";
    std::cout << "set-conservation failures: " << bad << "\n";
    std::cout << "action-limit games: " << limit << "\n";
    // determinism
    MatchConfig mc; mc.specA = "v04"; mc.specB = "v03"; mc.games = 50; mc.seed = 7; mc.threads = 1;
    mc.rules = rulesFrom(argc, argv);
    MatchStats s1 = runMatch(mc), s2 = runMatch(mc);
    std::cout << "determinism: " << ((s1.winsA == s2.winsA && s1.events == s2.events) ? "PASS" : "FAIL") << "\n";
    std::cout << ((viol == 0 && bad == 0) ? "VERIFY PASS" : "VERIFY FAIL") << std::endl;
    return viol == 0 && bad == 0 ? 0 : 1;
  }

  if (cmd == "matrix") {
    std::string list = argVal(argc, argv, "policies", "v04,v03,v02,lockout,detective,diversifier,hunter,bluffer,random");
    std::vector<std::string> ps;
    { std::stringstream ss(list); std::string it; while (std::getline(ss, it, ',')) ps.push_back(it); }
    int games = atoi(argVal(argc, argv, "games", "300").c_str());
    uint64_t seed = strtoull(argVal(argc, argv, "seed", "5150").c_str(), nullptr, 10);
    Rules r = rulesFrom(argc, argv);
    std::cout << "{\"policies\":[";
    for (size_t i = 0; i < ps.size(); i++) std::cout << (i ? "," : "") << "\"" << ps[i] << "\"";
    std::cout << "],\"cells\":[";
    bool first = true;
    for (size_t i = 0; i < ps.size(); i++) for (size_t j = 0; j < ps.size(); j++) {
      if (i == j) continue;
      MatchConfig mc; mc.specA = ps[i]; mc.specB = ps[j]; mc.games = games;
      mc.seed = seed + i * 1000 + j; mc.rules = r; mc.threads = threads;
      mc.rotations = atoi(argVal(argc, argv, "rotations", "2").c_str());
      MatchStats st = runMatch(mc);
      if (!first) std::cout << ",";
      first = false;
      printMatch(st, ps[i], ps[j], true, std::cout, mc.rotations, ps[i] == ps[j]);
      std::cout.flush();
    }
    std::cout << "]}" << std::endl;
    return 0;
  }

  if (cmd == "tune") {
    TuneSpec sp;
    std::string panel = argVal(argc, argv, "panel", "v03,lockout,detective,diversifier");
    // A panel member may itself be a spec with options, and the spec grammar
    // separates options with commas -- so an exploitability panel of ONE target
    // ("v06:hcap=decl,hstr=0.15") silently became a panel of two.  When the
    // string contains a semicolon it is the separator, and '+' inside a member
    // is restored to ','.  Both are backwards compatible with every existing
    // call in engine/*.sh.
    { char sep = panel.find(';') != std::string::npos ? ';' : ',';
      std::stringstream ss(panel); std::string it;
      while (std::getline(ss, it, sep)) {
        if (it.empty()) continue;
        for (auto& ch : it) if (ch == '+') ch = ',';
        sp.panel.push_back(it);
      } }
    sp.gamesPerOpponent = atoi(argVal(argc, argv, "games", "250").c_str());
    sp.population = atoi(argVal(argc, argv, "pop", "24").c_str());
    sp.elite = atoi(argVal(argc, argv, "elite", "6").c_str());
    sp.generations = atoi(argVal(argc, argv, "gens", "40").c_str());
    sp.beta = atof(argVal(argc, argv, "beta", "10").c_str());
    sp.sigma0 = atof(argVal(argc, argv, "sigma", "0.6").c_str());
    sp.rotations = atoi(argVal(argc, argv, "rotations", "2").c_str());
    sp.sigmaRel = atof(argVal(argc, argv, "sigmarel", "0").c_str());
    sp.paired = argFlag(argc, argv, "paired");
    { std::string ob = argVal(argc, argv, "obj", "softmin");
      sp.objective = ob == "min"           ? TuneObjective::Min
                   : ob == "mean"          ? TuneObjective::Mean
                   : ob == "regret"        ? TuneObjective::Regret
                   : ob == "minimaxregret" ? TuneObjective::MinimaxRegret
                                           : TuneObjective::SoftMin; }
    { std::string sh = argVal(argc, argv, "shard", "");
      if (!sh.empty()) { auto sl = sh.find('/');
        if (sl != std::string::npos) { sp.shard = atoi(sh.substr(0, sl).c_str());
                                       sp.shards = std::max(1, atoi(sh.substr(sl + 1).c_str())); } } }
    sp.seed = strtoull(argVal(argc, argv, "seed", "424242").c_str(), nullptr, 10);
    // v0.7 phase 2.  --kpi selects the quantity the CEM climbs.  The default is
    // the incumbent behaviour (win rate); the others name a per-decision failure
    // mode of the TARGET (tuner.hpp, TuneKpi) so that a battery of exploiter
    // searches does not share one search bias.
    sp.kpi = kpiFromName(argVal(argc, argv, "kpi", "win"));
    // v0.7 P2.  --partners fits the one-seat deviation column: the candidate
    // occupies ONE seat of the adversary team and the other two carry whatever
    // is named here (normally the target itself).  --correlated draws the A2
    // ex-ante signal during fitting.
    { std::string pt = argVal(argc, argv, "partners", "");
      for (auto& ch : pt) if (ch == '+') ch = ',';
      sp.partners = pt; }
    sp.correlated = argFlag(argc, argv, "correlated");
    sp.rules = rulesFrom(argc, argv);
    sp.threads = threads;
    sp.baseSpec = argVal(argc, argv, "base", "v04");
    std::vector<double> mu;
    std::string init = argVal(argc, argv, "init", "");
    if (!init.empty()) { std::stringstream ss(init); std::string t; while (std::getline(ss, t, '|')) mu.push_back(atof(t.c_str())); }
    else if (argFlag(argc, argv, "fromv6")
             || sp.baseSpec.rfind("v07", 0) == 0) {
      // v0.7 P-3 fix.  The v0.6 exploitability probe fitted a responder from
      // v0.5's 34-coordinate family, seeded from v0.5's weights, against a
      // 37-coordinate v0.6 target on a different scoring path
      // (SUBOPTIMALITY-LEDGER.md P-3).  An exploiter should start from the
      // incumbent it is attacking, in the incumbent's own class.
      for (int i = 0; i < NV6PARAM; i++) mu.push_back(V6PARAMS[i]);
    }
    else if (sp.baseSpec.rfind("v05", 0) == 0 || sp.baseSpec.rfind("v06", 0) == 0) {
      V05Config d; for (int i = 0; i < NFEAT; i++) mu.push_back(d.w[i]);
    }
    else { V04Config d; for (int i = 0; i < NFEAT; i++) mu.push_back(d.w[i]); }
    // The v0.7 responder class extends the vector by NR7 coordinates; at zero
    // they make it v0.6 bit for bit, so starting the fit there is starting it at
    // the incumbent.
    if (sp.baseSpec.rfind("v07", 0) == 0 && mu.size() == size_t(NV6PARAM))
      for (int i = 0; i < NR7; i++) mu.push_back(0.0);
    if (argFlag(argc, argv, "full") && mu.size() == NFEAT
        && (sp.baseSpec.rfind("v05", 0) == 0 || sp.baseSpec.rfind("v06", 0) == 0)) {
      V05Config d;
      mu.push_back(d.declThreshold); mu.push_back(d.lockedAllocThresh);
      mu.push_back(d.askFloor); mu.push_back(double(d.patiencePool));
      mu.push_back(d.oppCardFloor); mu.push_back(d.valueWeight);
      mu.push_back(d.linearWeight); mu.push_back(d.minTeamProb);
      mu.push_back(d.declareMargin);
      mu.push_back(d.priorTheta); mu.push_back(d.priorPhi);
      mu.push_back(double(d.searchTopK)); mu.push_back(d.chainWeight); mu.push_back(d.threatWeight);
      if (sp.baseSpec.rfind("v06", 0) == 0) {   // the three v0.6 ask terms
        mu.push_back(0.0); mu.push_back(0.0); mu.push_back(0.0);
      }
    }
    else if (argFlag(argc, argv, "full") && mu.size() == NFEAT) {
      V04Config d;
      mu.push_back(d.declThreshold); mu.push_back(d.lockedAllocThresh);
      mu.push_back(d.askFloor); mu.push_back(double(d.patiencePool));
      mu.push_back(d.oppCardFloor); mu.push_back(d.valueWeight);
      mu.push_back(d.linearWeight); mu.push_back(d.minTeamProb);
      mu.push_back(d.declareMargin);
      mu.push_back(d.priorTheta); mu.push_back(d.priorPhi);
      mu.push_back(double(d.searchTopK)); mu.push_back(d.chainWeight); mu.push_back(d.threatWeight);
    }
    sp.lo.assign(mu.size(), -12.0); sp.hi.assign(mu.size(), 20.0);
    sp.lo[0] = 0.0; sp.hi[0] = 30.0;
    if (mu.size() > NFEAT) {
      const double plo[14] = {0.55, 0.55, 0.0,  0.0, 0.0,  0.0, 0.0,  0.05, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0};
      const double phi[14] = {0.999, 0.99999, 0.6, 20.0, 12.0, 40.0, 3.0, 0.95, 0.05, 1.5, 0.6, 14.0, 12.0, 12.0};
      for (int i = 0; i < 14 && NFEAT + i < (int)mu.size(); i++) { sp.lo[NFEAT + i] = plo[i]; sp.hi[NFEAT + i] = phi[i]; }
      // v0.6's three extra ask terms: wVoid >= 0 (creating a void is an asset),
      // wTeamHas <= 0 (asking for a card our own team probably holds is waste),
      // wLastLive free (the forced endgame it walks into can cut either way).
      const double q6lo[3] = { 0.0, -12.0, -12.0 };
      const double q6hi[3] = { 12.0,   0.0,  12.0 };
      for (int i = 0; i < 3 && NFEAT + 14 + i < (int)mu.size(); i++) {
        sp.lo[NFEAT + 14 + i] = q6lo[i]; sp.hi[NFEAT + 14 + i] = q6hi[i];
      }
      // The v0.7 responder coordinates are left unsigned: every one of them is a
      // hypothesis about how the TARGET can be exploited, and fixing its sign in
      // advance would be assuming the answer.  `deadDonation` in particular is
      // the coordinate that lets a linear score price a deliberate miss for the
      // first time, and which way it should point is exactly what is unknown.
      for (int i = 0; i < NR7 && NFEAT + 17 + i < (int)mu.size(); i++) {
        sp.lo[NFEAT + 17 + i] = -12.0; sp.hi[NFEAT + 17 + i] = 12.0;
      }
    }
    { std::string sigPer = argVal(argc, argv, "sigmaparams", "");
      if (!sigPer.empty()) { std::stringstream ss(sigPer); std::string t;
        while (std::getline(ss, t, '|')) sp.sigmaVec.push_back(atof(t.c_str())); } }
    std::string outPath = argVal(argc, argv, "out", "");
    FILE* out = outPath.empty() ? stdout : fopen(outPath.c_str(), "w");
    std::vector<double> w = tune(sp, mu, out);
    fprintf(out, "{\"weights\":\"");
    for (size_t i = 0; i < w.size(); i++) fprintf(out, "%s%.5f", i ? "|" : "", w[i]);
    fprintf(out, "\"}\n");
    if (out != stdout) fclose(out);
    printf("weights=");
    for (size_t i = 0; i < w.size(); i++) printf("%s%.5f", i ? "|" : "", w[i]);
    printf("\n");
    return 0;
  }

  if (cmd == "selftest") {
    // Validate the exact belief engines against each other and against exact
    // rejection sampling from the constraint-satisfying posterior.
    int games = atoi(argVal(argc, argv, "games", "60").c_str());
    Rules r = rulesFrom(argc, argv);
    double maxCardDiff = 0, maxDisjDiff = 0, maxSinkDiff = 0, sumSink = 0; long long nSink = 0;
    long long checks = 0, blockFail = 0;
    Rng rng(4242);
    for (int gi = 0; gi < games; gi++) {
      std::unique_ptr<Agent> A[3], B[3];
      for (int i = 0; i < 3; i++) { A[i] = makeAgent("v04"); B[i] = makeAgent("v03"); }
      Agent* ag[NPLAY];
      for (int p = 0; p < NPLAY; p++) ag[p] = (p % 2 == 0) ? A[p / 2].get() : B[p / 2].get();
      Game game;
      game.setup(mixSeed(777, gi), r, ag);
      // replay the game, auditing beliefs at every event
      int guard = 0;
      while (game.g.pub.activeSets() && guard++ < 300) {
        for (int p = 0; p < NPLAY; p++) {
          const Knowledge& kk = ag[p]->k;
          if (!kk.unresolved) continue;
          BlockDP bd;
          if (!bd.build(kk)) { blockFail++; continue; }
          static double mb[NCARD][NPLAY], mc[NCARD][NPLAY];
          for (int c = 0; c < NCARD; c++) for (int q = 0; q < NPLAY; q++) { mb[c][q] = 0; mc[c][q] = 0; }
          bd.marginals(mb);
          DealDP dd;
          bool ddok = dd.build(kk);
          if (ddok) dd.marginals(mc);
          Belief sk; sk.sinkhornDisj(kk, 4, 8);
          // exact rejection sample as ground truth for the disjunctive posterior
          static double ms[NCARD][NPLAY];
          for (int c = 0; c < NCARD; c++) for (int q = 0; q < NPLAY; q++) ms[c][q] = 0;
          int drawn = 0;
          if (ddok) {
            std::array<uint8_t, NCARD> buf{};
            for (int c = 0; c < NCARD; c++) buf[c] = kk.owner[c];
            for (int t = 0; t < 40000 && drawn < 4000; t++) {
              dd.sample(rng, buf.data());
              if (!Belief::satisfies(kk, buf.data())) continue;
              uint64_t u = kk.unresolved;
              while (u) { int c = __builtin_ctzll(u); u &= u - 1; ms[c][buf[c]] += 1; }
              drawn++;
            }
          }
          uint64_t u = kk.unresolved;
          while (u) {
            int c = __builtin_ctzll(u); u &= u - 1;
            for (int q = 0; q < NPLAY; q++) {
              if (kk.disj.empty() && ddok) maxCardDiff = std::max(maxCardDiff, std::fabs(mb[c][q] - mc[c][q]));
              if (drawn >= 1500) {
                double gt = ms[c][q] / drawn;
                maxDisjDiff = std::max(maxDisjDiff, std::fabs(mb[c][q] - gt));
                double sd = std::fabs(sk.marg[c][q] - mb[c][q]);
                maxSinkDiff = std::max(maxSinkDiff, sd);
                sumSink += sd; nSink++;
              }
            }
            checks++;
          }
          break;   // one observer per event is enough
        }
        // advance one action
        if (!game.g.pub.teamAlive(0) || !game.g.pub.teamAlive(1)) break;
        if (!game.g.pub.handCount[game.g.turn]) break;
        AskMove mv = ag[game.g.turn]->chooseAsk(game.g.pub);
        if (!legalAsk(game.g, game.g.turn, mv.card, mv.target)) break;
        int actor = game.g.turn, target = mv.target, card = mv.card;
        bool success = (game.g.hand[target] & bit(card)) != 0;
        if (success) { game.g.hand[target] &= ~bit(card); game.g.hand[actor] |= bit(card); }
        game.g.pub.handCount[actor] = uint8_t(popcount64(game.g.hand[actor]));
        game.g.pub.handCount[target] = uint8_t(popcount64(game.g.hand[target]));
        Event e{}; e.kind = Kind::Ask; e.actor = uint8_t(actor); e.target = uint8_t(target);
        e.card = uint8_t(card); e.set = uint8_t(setOf(card)); e.success = success;
        game.emit(e);
        if (!success) { game.g.turn = target; game.g.pub.turn = target; }
      }
    }
    printf("checks                    %lld\n", checks);
    printf("block build failures      %lld\n", blockFail);
    printf("block vs card DP (no C5)  max abs diff %.3e\n", maxCardDiff);
    printf("block vs exact sampling   max abs diff %.3e\n", maxDisjDiff);
    printf("sinkhorn vs block  max %.4f  mean %.5f\n", maxSinkDiff, nSink ? sumSink / nSink : 0.0);
    printf("%s\n", (maxCardDiff < 1e-9 && maxDisjDiff < 0.05 && blockFail == 0) ? "SELFTEST PASS" : "SELFTEST CHECK");
    printf("note: allocation probabilities are validated by `fish oracle`, not here\n");
    return 0;
  }


  if (cmd == "oracle") {
    // Deterministic brute-force validation of the exact block engine.  Every
    // quantity the engine claims to compute exactly -- Z, per-card marginals,
    // team-ownership probabilities and full allocation probabilities -- is
    // compared against exhaustive enumeration of the posterior on small
    // reachable states, and the sampler's frequencies are compared against the
    // exact marginals.  Coverage is reported alongside the result: states whose
    // posterior is too large to enumerate are counted, not silently dropped.
    int games      = atoi(argVal(argc, argv, "games", "200").c_str());
    long long maxD = atoll(argVal(argc, argv, "maxdeals", "200000").c_str());
    int samples    = atoi(argVal(argc, argv, "samples", "3000").c_str());
    uint64_t seed  = strtoull(argVal(argc, argv, "seed", "20260822").c_str(), nullptr, 10);
    std::string aSpec = argVal(argc, argv, "a", "v04");
    std::string bSpec = argVal(argc, argv, "b", "v03");
    Rules r = rulesFrom(argc, argv);
    OracleStats st;
    Rng rng(seed ? seed : 1);
    static BruteForce bf;
    std::unique_ptr<Agent> A[3], B[3];
    for (int i = 0; i < 3; i++) { A[i] = makeAgent(aSpec); B[i] = makeAgent(bSpec); }
    Agent* ag[NPLAY];
    for (int p = 0; p < NPLAY; p++) ag[p] = (p % 2 == 0) ? A[p / 2].get() : B[p / 2].get();
    Game game;
    game.observer = [&](const Game& gm) {
      for (int p = 0; p < NPLAY; p++) {
        const Knowledge& kk = gm.agents[p]->k;
        if (!kk.unresolved) continue;
        DealDP dd;
        if (!dd.build(kk)) continue;
        if (!(dd.N > 0) || dd.N > double(maxD)) { st.statesSkipped++; continue; }
        if (!bf.enumerate(kk, maxD * 64)) { st.statesSkipped++; continue; }
        BlockDP bd;
        if (!bd.build(kk)) { st.buildFailures++; continue; }
        oracleCheckState(kk, bd, bf, st, rng, samples);
      }
    };
    for (int gi = 0; gi < games; gi++) game.run(mixSeed(seed, gi), r, ag);
    printf("games replayed             %d  (%s vs %s)\n", games, aSpec.c_str(), bSpec.c_str());
    printf("states enumerated          %lld  (with a live C5 certificate: %lld)\n",
           st.statesChecked, st.statesWithC5);
    printf("states skipped (too large) %lld\n", st.statesSkipped);
    printf("block build failures       %lld\n", st.buildFailures);
    printf("consistent deals counted   %lld\n", st.deals);
    printf("partition function Z       max rel diff %.3e\n", st.maxZRelDiff);
    printf("per-card marginals         max abs diff %.3e over %lld checks\n",
           st.maxMarginalDiff, st.marginalChecks);
    printf("team-ownership prob        max abs diff %.3e over %lld checks\n",
           st.maxTeamDiff, st.teamChecks);
    printf("named allocation prob      max abs diff %.3e over %lld checks\n",
           st.maxAllocDiff, st.allocChecks);
    printf("equal-prob corollary       max within-class spread %.3e\n", st.maxCountVectorSpread);
    printf("bestTeamAllocation         %lld checks, %lld inconsistent, %lld not argmax, max diff %.3e\n",
           st.bestAllocChecks, st.bestAllocInconsistent, st.bestAllocNotArgmax, st.maxBestAllocDiff);
    printf("sampler vs exact marginals max abs diff %.4f over %lld draws\n",
           st.maxSampleDiff, st.sampleDraws);
    printf("%s\n", st.pass() ? "ORACLE PASS" : "ORACLE FAIL");
    return st.pass() ? 0 : 1;
  }

  if (cmd == "gateaudit") {
    // Corpus-wide false-negative audit of the declaration pre-gates.  The
    // shipped policy screens half-suits with two cheap scores before running the
    // full posterior query; neither screen is proved to be an upper bound on the
    // full evaluation, so this re-runs the complete evaluation on every half-suit
    // the screens reject and counts how often it would have declared one.
    std::string a = argVal(argc, argv, "a", "v04:mgate=0.008,gateaudit=1");
    int deals = atoi(argVal(argc, argv, "games", "300").c_str());
    uint64_t seed = strtoull(argVal(argc, argv, "seed", "90210").c_str(), nullptr, 10);
    int rots = atoi(argVal(argc, argv, "rotations", "6").c_str());
    std::string panel = argVal(argc, argv, "panel", "v03,lockout,detective,v02,diversifier,hunter,bluffer,random");
    std::vector<std::string> opps;
    { std::stringstream ss(panel); std::string it; while (std::getline(ss, it, ',')) opps.push_back(it); }
    Rules r = rulesFrom(argc, argv);
    for (const auto& opp : opps) {
      MatchConfig mc;
      mc.specA = a; mc.specB = opp; mc.games = deals; mc.rotations = rots;
      mc.seed = seed; mc.rules = r; mc.threads = threads;
      runMatch(mc);
    }
    auto& G = gateAudit();
    long long seen = G.setsSeen.load(), rej = G.setsRejected.load(), fn = G.falseNegatives.load();
    printf("opponents                  %zu\n", opps.size());
    printf("declaration opportunities  %lld\n", G.opportunities.load());
    printf("(opportunity, half-suit)   %lld\n", seen);
    printf("rejected by a cheap gate   %lld  (%.3f%% of pairs)\n",
           rej, seen ? 100.0 * double(rej) / double(seen) : 0.0);
    printf("false negatives            %lld  (%.5f%% of rejections)\n",
           fn, rej ? 100.0 * double(fn) / double(rej) : 0.0);
    printf("declarations, gated        %lld\n", G.gatedDeclares.load());
    printf("declarations, ungated      %lld\n", G.ungatedDeclares.load());
    printf("opportunities where the chosen action differs  %lld\n", G.actionsChanged.load());
    printf("%s\n", (fn == 0 && G.actionsChanged.load() == 0) ? "GATEAUDIT PASS (no false negative observed)"
                                                             : "GATEAUDIT: FALSE NEGATIVES PRESENT");
    return 0;
  }

  if (cmd == "calibrate") {
    std::string a = argVal(argc, argv, "a", "v04");
    std::string b = argVal(argc, argv, "b", "v03");
    int deals = atoi(argVal(argc, argv, "games", "300").c_str());
    Rules r = rulesFrom(argc, argv);
    CalibSink sink = collectCalibration(a, b, deals, strtoull(argVal(argc, argv, "seed", "8181").c_str(), nullptr, 10), r);
    Reliability ra = reliability(sink.ask), rd = reliability(sink.decl);
    auto dump = [&](const char* label, const Reliability& x) {
      printf("%s n=%d brier=%.5f logloss=%.5f ece=%.5f meanPred=%.4f meanObs=%.4f\n",
             label, x.n, x.brier, x.logloss, x.ece, x.meanPred, x.meanObs);
      for (int i = 0; i < 10; i++) if (x.binN[i])
        printf("   [%.1f,%.1f) n=%6d pred=%.4f obs=%.4f\n", i / 10.0, (i + 1) / 10.0, x.binN[i], x.binPred[i], x.binObs[i]);
    };
    dump("ask ", ra);
    dump("decl", rd);
    return 0;
  }

  if (cmd == "ablate") {
    // Paired ablations: every variant plays the SAME deals against the SAME
    // panel, so the difference from the reference is a matched comparison.
    std::string ref = argVal(argc, argv, "ref", "v04");
    std::string variants = argVal(argc, argv, "variants", "");
    std::string panel = argVal(argc, argv, "panel", "v03,lockout,detective");
    int deals = atoi(argVal(argc, argv, "games", "500").c_str());
    int rot = atoi(argVal(argc, argv, "rotations", "2").c_str());
    uint64_t seed = strtoull(argVal(argc, argv, "seed", "606060").c_str(), nullptr, 10);
    Rules r = rulesFrom(argc, argv);
    std::vector<std::string> vs, ps;
    { std::stringstream ss(variants); std::string it; while (std::getline(ss, it, ';')) if (!it.empty()) vs.push_back(it); }
    { std::stringstream ss(panel); std::string it; while (std::getline(ss, it, ',')) ps.push_back(it); }
    auto runAll = [&](const std::string& spec) {
      std::vector<uint8_t> all;
      int wins = 0, tot = 0;
      std::vector<double> per;
      for (size_t i = 0; i < ps.size(); i++) {
        MatchConfig mc; mc.specA = spec; mc.specB = ps[i]; mc.games = deals;
        mc.seed = mixSeed(seed, i * 7919 + 3); mc.rules = r; mc.threads = threads; mc.rotations = rot;
        MatchStats st = runMatch(mc);
        all.insert(all.end(), st.paired.begin(), st.paired.end());
        wins += st.winsA; tot += st.games * rot;
        per.push_back(double(st.winsA) / double(st.games * rot));
      }
      return std::make_tuple(all, double(wins) / tot, per);
    };
    auto [refPaired, refWr, refPer] = runAll(ref);
    printf("{\"reference\":\"%s\",\"winRate\":%.5f,\"perOpponent\":[", ref.c_str(), refWr);
    for (size_t i = 0; i < refPer.size(); i++) printf("%s%.5f", i ? "," : "", refPer[i]);
    printf("],\"panel\":[");
    for (size_t i = 0; i < ps.size(); i++) printf("%s\"%s\"", i ? "," : "", ps[i].c_str());
    printf("],\"variants\":[\n");
    for (size_t v = 0; v < vs.size(); v++) {
      auto [vp, vwr, vper] = runAll(vs[v]);
      double m, lo, hi;
      pairedBootstrap(refPaired, vp, m, lo, hi, 31337, 20000, double(rot));
      printf("  {\"spec\":\"%s\",\"winRate\":%.5f,\"deltaFromRef\":%.5f,\"ci\":[%.5f,%.5f],\"perOpponent\":[",
             vs[v].c_str(), vwr, m, lo, hi);
      for (size_t i = 0; i < vper.size(); i++) printf("%s%.5f", i ? "," : "", vper[i]);
      printf("]}%s\n", v + 1 < vs.size() ? "," : "");
      fflush(stdout);
    }
    printf("]}\n");
    return 0;
  }

  if (cmd == "fitvalue") {
    // Ridge regression of the final set differential on public-belief-state
    // features, gathered from self-play decision points.
    std::string a = argVal(argc, argv, "a", "v04:value=0");
    std::string b = argVal(argc, argv, "b", "v03");
    int deals = atoi(argVal(argc, argv, "games", "400").c_str());
    double lambda = atof(argVal(argc, argv, "lambda", "1e-3").c_str());
    Rules r = rulesFrom(argc, argv);
    uint64_t seed = strtoull(argVal(argc, argv, "seed", "31415").c_str(), nullptr, 10);
    int nThreads = threads > 0 ? threads : int(std::thread::hardware_concurrency());
    std::vector<ValueSink> sinks(nThreads);
    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; t++) pool.emplace_back([&, t]() {
      std::unique_ptr<Agent> A[3], B[3];
      for (int i = 0; i < 3; i++) { A[i] = makeAgent(a); B[i] = makeAgent(b); }
      Game game;
      for (int i = t; i < deals; i += nThreads) {
        uint64_t s2 = mixSeed(seed, uint64_t(i) * 2654435761ull + 1);
        for (int orient = 0; orient < 2; orient++) {
          Agent* ag[NPLAY];
          for (int p = 0; p < NPLAY; p++) ag[p] = (teamOf(p) == orient) ? A[p / 2].get() : B[p / 2].get();
          game.vsink = &sinks[t];
          game.run(s2, r, ag);
        }
      }
    });
    for (auto& th : pool) th.join();
    int D = NVFEAT;
    std::vector<double> XtX(D * D, 0.0), Xty(D, 0.0);
    long long rows = 0;
    for (auto& sk : sinks) for (size_t i = 0; i < sk.y.size(); i++) {
      const auto& x = sk.X[i];
      for (int u = 0; u < D; u++) {
        Xty[u] += double(x[u]) * sk.y[i];
        for (int v = 0; v < D; v++) XtX[u * D + v] += double(x[u]) * double(x[v]);
      }
      rows++;
    }
    for (int u = 0; u < D; u++) XtX[u * D + u] += lambda * rows;
    // Gaussian elimination
    std::vector<double> M(XtX), rhs(Xty), w(D, 0.0);
    for (int c = 0; c < D; c++) {
      int piv = c;
      for (int rr = c + 1; rr < D; rr++) if (std::fabs(M[rr * D + c]) > std::fabs(M[piv * D + c])) piv = rr;
      if (std::fabs(M[piv * D + c]) < 1e-12) continue;
      if (piv != c) { for (int j = 0; j < D; j++) std::swap(M[c * D + j], M[piv * D + j]); std::swap(rhs[c], rhs[piv]); }
      for (int rr = 0; rr < D; rr++) {
        if (rr == c) continue;
        double fac = M[rr * D + c] / M[c * D + c];
        if (fac == 0) continue;
        for (int j = 0; j < D; j++) M[rr * D + j] -= fac * M[c * D + j];
        rhs[rr] -= fac * rhs[c];
      }
    }
    for (int c = 0; c < D; c++) w[c] = std::fabs(M[c * D + c]) > 1e-12 ? rhs[c] / M[c * D + c] : 0.0;
    double sse = 0, sst = 0, ybar = 0; long long nAll = 0;
    for (auto& sk : sinks) for (size_t i = 0; i < sk.y.size(); i++) { ybar += sk.y[i]; nAll++; }
    ybar /= std::max(1LL, nAll);
    for (auto& sk : sinks) for (size_t i = 0; i < sk.y.size(); i++) {
      double pred = 0;
      for (int u = 0; u < D; u++) pred += w[u] * sk.X[i][u];
      sse += (pred - sk.y[i]) * (pred - sk.y[i]);
      sst += (sk.y[i] - ybar) * (sk.y[i] - ybar);
    }
    fprintf(stderr, "rows=%lld  R2=%.4f  rmse=%.4f\n", rows, 1.0 - sse / std::max(1e-9, sst), std::sqrt(sse / std::max(1LL, nAll)));
    printf("vweights=");
    for (int u = 0; u < D; u++) printf("%s%.6f", u ? "|" : "", w[u]);
    printf("\n");
    return 0;
  }

  // ------------------------------------------------------------------ v0.7 K1
  // Sample the states the truncated search's own depth cut produces, record the
  // realised continuation value under blueprint continuation, and fit a linear
  // leaf on them.  See engine/src/v07_leaffit.hpp for why the between-candidate
  // score is the one that decides and the overall score is a distraction.
  if (cmd == "v7leaffit") {
    if (argFlag(argc, argv, "help")) {
      std::cout <<
        "usage: fish7 v7leaffit --a=<search spec> [--b=v06] --games=N --seed=<FIT bank>\n"
        "                       [--oos=<eval bank> --oosgames=M] [--stride=K] [--ridge=L]\n"
        "                       [--threads=T] [--rotations=2] [--json] [--out=FILE]\n"
        "\n"
        "Runs --a against --b with the leaf sampler armed.  Every depth cut writes its\n"
        "13-feature row; the rollout then keeps playing to the end under blueprint\n"
        "continuation and the final signed half-suit differential is the target.  The\n"
        "policy is NOT perturbed: playOut still returns the leaf value it would have\n"
        "returned, and the determinization RNG is untouched.\n"
        "Fit on the reserve bank (7030004); evaluate out of sample with --oos.\n";
      return 0;
    }
    auto sample = [&](uint64_t bank, int games, std::vector<fish::LeafSample>& out) {
      fish::leafResetStores();
      fish::g_leafSampling = true;
      fish::g_leafStride = atoi(argVal(argc, argv, "stride", "1").c_str());
      MatchConfig mc;
      mc.specA = argVal(argc, argv, "a", "v06:s1=1,det=12,cand=4,kappa=2.5,rbelief=indep,depth=12,maxq=26");
      mc.specB = argVal(argc, argv, "b", "v06");
      mc.games = games;
      mc.seed = bank;
      mc.rules = rulesFrom(argc, argv);
      mc.threads = threads;
      mc.rotations = atoi(argVal(argc, argv, "rotations", "2").c_str());
      MatchStats st = runMatch(mc);
      (void)st;
      fish::g_leafSampling = false;
      fish::leafDrain(out);
      fish::leafResetStores();
    };
    std::vector<fish::LeafSample> fitS, oosS;
    uint64_t fitBank = strtoull(argVal(argc, argv, "seed", "7030004").c_str(), nullptr, 10);
    int fitGames = atoi(argVal(argc, argv, "games", "300").c_str());
    sample(fitBank, fitGames, fitS);
    uint64_t oosBank = strtoull(argVal(argc, argv, "oos", "0").c_str(), nullptr, 10);
    if (oosBank) sample(oosBank, atoi(argVal(argc, argv, "oosgames", "150").c_str()), oosS);

    using namespace fish::leaffit;
    double lam = atof(argVal(argc, argv, "ridge", "1e-4").c_str());
    auto XofS = [](const std::vector<fish::LeafSample>& S, std::vector<const double*>& X, std::vector<double>& y) {
      X.clear(); y.clear(); X.reserve(S.size()); y.reserve(S.size());
      for (const auto& s : S) { X.push_back(s.f); y.push_back(s.y); }
    };
    std::vector<const double*> Xf, Xo; std::vector<double> yf, yo;
    XofS(fitS, Xf, yf); XofS(oosS, Xo, yo);
    Grouped gf = groupByDecisionDet(fitS), go = groupByDecisionDet(oosS);

    // The three reference weight vectors, expressed in the same 13-slot basis.
    double wMat[fish::NLEAF] = {0,0,0,0,0,0,0,0,0,0,0,0,1};   // MaterialLeaf: f[12]
    double wScr[fish::NLEAF] = {0,1,0,0,0,0,0,0,0,0,0,0,0};   // ScoreLeaf: f[1]
    bool allCols[fish::NLEAF]; for (int j = 0; j < fish::NLEAF; j++) allCols[j] = true;
    allCols[12] = false;   // f12 = f1 + lambda*f2 exactly; admitting it is collinear
    bool sc2[fish::NLEAF] = {false,true,true,false,false,false,false,false,false,false,false,false,false};

    Fit lvl = ridgeFit(Xf, yf, lam, true, allCols);  lvl.tag = "level";
    Fit dff = diffFit(fitS, gf, lam, allCols);
    Fit sc  = ridgeFit(Xf, yf, lam, true, sc2);      sc.tag  = "sc2";   // free ratio of f1:f2
    // Calibrate the difference-optimal direction to the terminal-return scale.
    { R2 r = scoreLevel(Xf, yf, dff.w);
      double a = r.slope;
      if (std::isfinite(a) && std::fabs(a) > 1e-9) {
        double mp = 0, my = 0;
        for (size_t i = 0; i < Xf.size(); i++) { mp += predict(dff.w, Xf[i]); my += yf[i]; }
        mp /= double(Xf.size()); my /= double(Xf.size());
        for (int j = 1; j < fish::NLEAF; j++) dff.w[j] *= a;
        dff.w[0] = my - a * mp;
      } }

    struct Row { const char* name; const double* w; };
    Row rows[] = { {"material(f12)", wMat}, {"score(f1)", wScr}, {"sc2(f1,f2 free)", sc.w},
                   {"linear-level", lvl.w}, {"linear-diff", dff.w} };
    double cf[fish::NLEAF]; constancy(fitS, gf, cf);

    bool json = argFlag(argc, argv, "json");
    std::ostringstream J;
    J << "{\"cmd\":\"v7leaffit\",\"a\":\"" << argVal(argc, argv, "a", "") << "\""
      << ",\"fitBank\":" << fitBank << ",\"fitGames\":" << fitGames
      << ",\"oosBank\":" << oosBank
      << ",\"nFit\":" << fitS.size() << ",\"nFitGroups\":" << gf.spans.size()
      << ",\"nOos\":" << oosS.size() << ",\"nOosGroups\":" << go.spans.size()
      << ",\"ridge\":" << lam << ",\"rows\":[";
    if (!json) {
      printf("v7leaffit  a=%s  fit bank %llu (%d games)  n=%zu leaves, %zu (decision,det) groups\n",
             argVal(argc, argv, "a", "").c_str(), (unsigned long long)fitBank, fitGames,
             fitS.size(), gf.spans.size());
      if (oosBank) printf("           oos bank %llu  n=%zu leaves, %zu groups\n",
                          (unsigned long long)oosBank, oosS.size(), go.spans.size());
      printf("\nconstancy across candidates within a (decision,determinization) group:\n");
      for (int j = 0; j < fish::NLEAF; j++)
        printf("   f%-2d %-16s constant in %6.2f%% of groups\n", j, fish::leafFeatureName(j), 100 * cf[j]);
      printf("\n%-18s %8s %8s %8s | %8s %8s %8s | oos %8s %8s\n",
             "evaluator", "R2.lvl", "corr2.l", "slope", "R2.btw", "corr2.b", "slope.b", "corr2.l", "corr2.b");
    }
    bool first = true;
    for (const Row& rw : rows) {
      R2 L = scoreLevel(Xf, yf, rw.w), B = scoreBetween(fitS, gf, rw.w);
      R2 OL{}, OB{};
      if (oosBank) { OL = scoreLevel(Xo, yo, rw.w); OB = scoreBetween(oosS, go, rw.w); }
      if (!json)
        printf("%-18s %8.5f %8.5f %8.4f | %8.5f %8.5f %8.4f | %12.5f %8.5f\n",
               rw.name, L.r2, L.corr2, L.slope, B.r2, B.corr2, B.slope, OL.corr2, OB.corr2);
      J << (first ? "" : ",") << "{\"name\":\"" << rw.name << "\""
        << ",\"r2Level\":" << L.r2 << ",\"corr2Level\":" << L.corr2 << ",\"slopeLevel\":" << L.slope
        << ",\"r2Between\":" << B.r2 << ",\"corr2Between\":" << B.corr2 << ",\"slopeBetween\":" << B.slope
        << ",\"oosCorr2Level\":" << OL.corr2 << ",\"oosCorr2Between\":" << OB.corr2
        << ",\"oosR2Between\":" << OB.r2
        << ",\"w\":\"" << specOf(rw.w) << "\"}";
      first = false;
    }
    J << "],\"constancy\":[";
    for (int j = 0; j < fish::NLEAF; j++) J << (j ? "," : "") << cf[j];
    J << "],\"linearLevel\":\"linear@" << specOf(lvl.w) << "\""
      << ",\"linearDiff\":\"linear@" << specOf(dff.w) << "\""
      << ",\"sc2\":\"linear@" << specOf(sc.w) << "\"}";
    if (!json) {
      printf("\nleafeval=linear@%s        (level fit)\n", specOf(lvl.w).c_str());
      printf("leafeval=linear@%s        (diff fit, level-calibrated)\n", specOf(dff.w).c_str());
      printf("leafeval=linear@%s        (f1,f2 free ratio)\n", specOf(sc.w).c_str());
    } else {
      std::cout << J.str() << "\n";
    }
    std::string out = argVal(argc, argv, "out", "");
    if (!out.empty()) { std::ofstream f(out, std::ios::app); f << J.str() << "\n"; }
    return 0;
  }

  if (cmd == "bench") {
    MatchConfig mc; mc.specA = argVal(argc, argv, "a", "v04"); mc.specB = argVal(argc, argv, "b", "v04");
    mc.games = atoi(argVal(argc, argv, "games", "200").c_str());
    mc.rules = rulesFrom(argc, argv); mc.threads = threads;
    MatchStats st = runMatch(mc);
    std::cout << (st.games * 2 / st.seconds) << " games/s over " << st.games * 2 << " games\n";
    return 0;
  }

  if (cmd == "v6probe") {
    V6ProbeConfig pc;
    pc.specA = argVal(argc, argv, "a", "v05");
    pc.specB = argVal(argc, argv, "b", "v05");
    pc.games = atoi(argVal(argc, argv, "games", "120").c_str());
    pc.seed  = strtoull(argVal(argc, argv, "seed", "31").c_str(), nullptr, 10);
    pc.rules = rulesFrom(argc, argv);
    pc.threads = threads;
    std::string mode = argVal(argc, argv, "mode", "ties");
    std::cout << "v6probe " << mode << ": A=" << pc.specA << " B=" << pc.specB
              << " games=" << pc.games << " seed=" << pc.seed << "\n\n";
    if (mode == "ties") { V6TieStats st = runV6Ties(pc); printV6Ties(st, std::cout); return 0; }
    if (mode == "belief") {
      std::vector<std::pair<double,double>> tp;
      std::string grid = argVal(argc, argv, "theta", "0,0.2,0.3,0.44458,0.6,0.8,1.1,1.5,2.0");
      double phi = atof(argVal(argc, argv, "phi", "0.12198").c_str());
      std::stringstream ss(grid); std::string t;
      while (std::getline(ss, t, ',')) tp.push_back({atof(t.c_str()), phi});
      auto rows = runV6Belief(pc, tp);
      printV6Belief(rows, std::cout);
      return 0;
    }
    if (mode == "search") {
      V6SearchStats st = runV6Search(pc);
      printV6Search(st, pc.games, std::cout);
      return 0;
    }
    fprintf(stderr, "fish v6probe: unknown --mode=%s (ties|belief|search)\n", mode.c_str());
    return 2;
  }

  // ---- v0.7 T1: throughput, on ONE basis --------------------------------
  // Replaces E9.  The v0.6 record states the search costs "three orders of
  // magnitude"; that figure divides an all-threads number (303.4 g/s) by a
  // single-thread one (0.144 g/s) and is not a same-basis ratio, which the
  // corpus's own final audit already flagged (V2-final-audit.md SF-6) and the
  // ledger re-measured at 300-420x (SUBOPTIMALITY-LEDGER.md section 0.1).  This
  // command reports BOTH bases for every configuration and the ratio within
  // each, so the mistake is not available.
  if (cmd == "v7through") {
    std::string specs = argVal(argc, argv, "specs", "v06");
    std::string opp = argVal(argc, argv, "b", "v06");
    int games = atoi(argVal(argc, argv, "games", "200").c_str());
    int games1 = atoi(argVal(argc, argv, "games1", "0").c_str());
    int rot = atoi(argVal(argc, argv, "rotations", "2").c_str());
    uint64_t seed = strtoull(argVal(argc, argv, "seed", "7010001").c_str(), nullptr, 10);
    bool json = argFlag(argc, argv, "json");
    bool mirror = argFlag(argc, argv, "mirror");
    Rules rules = rulesFrom(argc, argv);
    std::vector<std::string> list;
    { std::stringstream ss(specs); std::string it; while (std::getline(ss, it, ';')) if (!it.empty()) list.push_back(it); }
    double ref = 0, ref1 = 0;
    if (!json) printf("%-52s %11s %11s %9s %9s\n", "configuration", "g/s (all)", "g/s (1 thr)", "x v06", "x v06 1t");
    for (size_t i = 0; i < list.size(); i++) {
      MatchConfig mc; mc.specA = list[i];
      // E9 timed every policy against ITSELF; the ledger's F-search and F-mid
      // rows time a search arm against `v05`/`v06`.  Those are different
      // quantities -- a mirror pays the configuration's cost on both sides --
      // and mixing them is how the corpus arrived at a cost ratio that is not a
      // same-basis ratio.  `--mirror` selects the first; `--b` selects the
      // second; the block heading records which.
      mc.specB = mirror ? list[i] : opp;
      mc.games = games;
      mc.rotations = rot; mc.seed = seed; mc.rules = rules; mc.threads = threads;
      MatchStats st = runMatch(mc);
      double gps = st.gamesPerSec(rot);
      MatchConfig m1 = mc; m1.threads = 1;
      m1.games = games1 > 0 ? games1 : std::max(1, games / 8);
      MatchStats s1 = runMatch(m1);
      double gps1 = s1.gamesPerSec(rot);
      if (i == 0) { ref = gps; ref1 = gps1; }
      if (json) {
        printf("{\"spec\":\"%s\",\"opp\":\"%s\",\"seed\":%llu,\"deals\":%d,\"rotations\":%d,"
               "\"mirror\":%s,\"gamesPerSecAll\":%.4f,\"threads\":%d,\"gamesPerSecOne\":%.4f,\"deals1\":%d,"
               "\"relAll\":%.4f,\"relOne\":%.4f,\"winRateA\":%.4f,\"eventsPerGame\":%.2f,%s}\n",
               list[i].c_str(), mc.specB.c_str(), (unsigned long long)seed, games, rot,
               mirror ? "true" : "false", gps, st.threadsUsed, gps1, m1.games,
               ref > 0 ? gps / ref : 0.0, ref1 > 0 ? gps1 / ref1 : 0.0,
               double(st.winsA) / double(std::max(1, st.games * rot)),
               double(st.events) / double(std::max(1, st.games * rot)),
               powerJson(powerLine(st.games * rot, st.games, rot, mc.specA == mc.specB)).c_str());
      } else {
        printf("%-52s %11.3f %11.3f %9.3f %9.3f\n", list[i].c_str(), gps, gps1,
               ref > 0 ? gps / ref : 0.0, ref1 > 0 ? gps1 / ref1 : 0.0);
      }
      fflush(stdout);
    }
    return 0;
  }

  // ---- v0.7 D1: per-decision records and per-decision scoring -------------
  if (cmd == "v7decide") {
    MatchConfig mc;
    mc.specA = argVal(argc, argv, "a", "v06");
    mc.specB = argVal(argc, argv, "b", "v06");
    mc.partnersA = argVal(argc, argv, "partners", "");
    mc.partnersB = argVal(argc, argv, "partnersb", "");
    mc.games = atoi(argVal(argc, argv, "games", "200").c_str());
    mc.rotations = atoi(argVal(argc, argv, "rotations", "2").c_str());
    mc.seed = strtoull(argVal(argc, argv, "seed", "7011001").c_str(), nullptr, 10);
    mc.rules = rulesFrom(argc, argv);
    mc.threads = threads;
    mc.captureDecisions = true;
    mc.captureTeamAOnly = !argFlag(argc, argv, "bothteams");
    // v0.7 phase 2.  --capture=a|b|both chooses whose decisions are recorded.
    // `b` is the TARGET arm: characterising an exploiter means measuring what
    // the target does wrong against it, which the phase-1 form could not do.
    { std::string ca = argVal(argc, argv, "capture", "a");
      mc.captureArm = ca == "b" ? 1 : (ca == "both" ? 2 : 0);
      if (mc.captureArm == 2) mc.captureTeamAOnly = false; }
    MatchStats st = runMatch(mc);
    std::string dump = argVal(argc, argv, "dump", "");
    if (!dump.empty()) {
      FILE* f = fopen(dump.c_str(), "w");
      if (f) {
        fprintf(f, "deal,rot,event,seat,team,kind,card,target,set,hit,ownLocked,oppLocked,"
                   "truthHolder,dead,gateBound,searched,changed,nCand,nTie,nFeasible,p,margin,score,pAlloc,unresolved,urgent,pressure,urgWhy,"
                   "l1have,l1flat,l1jSame,jointHit,exactHit,l1n,l1nCV,l1nAlloc,l1pMap,l1jTop,l1jSecond\n");
        for (const auto& r : st.decisions)
          fprintf(f, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%d,%d,%d,%d,"
                     "%d,%d,%d,%d,%d,%d,%d,%.1f,%.9f,%.9f,%.9f\n",
                  r.deal, r.rot, r.event, r.seat, r.team, r.kind, r.card, r.target, r.set, r.hit,
                  r.ownLocked, r.oppLocked, r.truthHolder, r.dead, r.gateBound, r.searched, r.changed,
                  r.nCand, r.nTie, r.nFeasible, r.p, r.margin, r.score, r.pAlloc, r.unresolved,
                  r.urgent, r.pressure, r.urgWhy,
                  r.l1have, r.l1flat, r.l1jSame, r.jointHit, r.exactHit, r.l1n, r.l1nCV,
                  r.l1nAlloc, r.l1pMap, r.l1jTop, r.l1jSecond);
        fclose(f);
      }
    }
    v07::DecSummary S = v07::summariseDecisions(st.decisions, mc.games);
    bool json = argFlag(argc, argv, "json");
    if (json) {
      printf("{\"probe\":\"v7decide\",\"a\":\"%s\",\"b\":\"%s\",\"capture\":\"%s\",\"seed\":%llu,\"deals\":%d,"
             "\"rotations\":%d,\"records\":%lld,\"gamesPerSec\":%.3f,\"metrics\":{",
             mc.specA.c_str(), mc.specB.c_str(),
             mc.captureArm == 1 ? "b" : (mc.captureArm == 2 ? "both" : "a"),
             (unsigned long long)mc.seed, mc.games,
             mc.rotations, S.rows, st.gamesPerSec(mc.rotations));
      for (size_t i = 0; i < S.m.size(); i++) {
        double m, lo, hi;
        v07::clusterRatioCI(S.m[i].perDealNum, S.m[i].perDealDen, m, lo, hi);
        printf("%s\"%s\":{\"rate\":%.6f,\"ci\":[%.6f,%.6f],\"n\":%.0f,\"halfWidth98\":%.4f}",
               i ? "," : "", S.m[i].name, m, lo, hi, S.m[i].den, halfWidth98(S.m[i].den));
      }
      printf("}}\n");
    } else {
      printf("v7decide  %s vs %s  seed %llu  %d deals x %d  -> %lld decision records  (%.1f games/s)\n",
             mc.specA.c_str(), mc.specB.c_str(), (unsigned long long)mc.seed, mc.games,
             mc.rotations, S.rows, st.gamesPerSec(mc.rotations));
      printf("  %-22s %10s  %-22s %10s %12s\n", "metric", "rate", "95% CI (deal-clustered)", "decisions", "98/sqrt(n)");
      for (size_t i = 0; i < S.m.size(); i++) {
        double m, lo, hi;
        v07::clusterRatioCI(S.m[i].perDealNum, S.m[i].perDealDen, m, lo, hi);
        char ci[64]; snprintf(ci, sizeof(ci), "[%.4f, %.4f]", lo, hi);
        printf("  %-22s %10.5f  %-22s %10.0f %11.3f\n", S.m[i].name, m, ci, S.m[i].den, halfWidth98(S.m[i].den));
      }
    }
    return 0;
  }

  // ---- v0.7: the reserved-seed registry ----------------------------------
  if (cmd == "seeds") {
    SeedCheck c = checkSeeds();
    bool json = argFlag(argc, argv, "json");
    if (json) {
      printf("{\"registry\":[");
      const auto& R = seedRegistry();
      for (size_t i = 0; i < R.size(); i++)
        printf("%s{\"seed\":%llu,\"role\":\"%s\",\"study\":\"%s\",\"battery\":\"%s\",\"unsealPhase\":%d}",
               i ? "," : "", (unsigned long long)R[i].seed, roleName(R[i].role), R[i].study,
               R[i].battery, R[i].unsealPhase);
      printf("],\"violations\":%d}\n", c.violations);
    } else {
      printf("%-10s %-12s %-6s %-4s %s\n", "seed", "role", "study", "unseal", "battery");
      for (const auto& e : seedRegistry())
        printf("%-10llu %-12s %-6s %-6d %s\n", (unsigned long long)e.seed, roleName(e.role),
               e.study, e.unsealPhase, e.battery);
      printf("\n%s", c.report.empty() ? "no violations\n" : c.report.c_str());
      printf("%d violation(s).  FISH_UNSEAL_PHASE=%d\n", c.violations, unsealPhase());
    }
    std::string req = argVal(argc, argv, "require", "");
    if (!req.empty()) {
      std::stringstream ss(req); std::string t; int bad = 0;
      while (std::getline(ss, t, ',')) {
        uint64_t sd = strtoull(t.c_str(), nullptr, 10);
        std::string why;
        if (!seedRegistered(sd)) { printf("UNREGISTERED seed %llu\n", (unsigned long long)sd); bad++; }
        else if (!seedUsable(sd, why)) { printf("%s\n", why.c_str()); bad++; }
      }
      if (bad) return 3;
    }
    return c.violations && argFlag(argc, argv, "strict") ? 4 : 0;
  }

  // ---- v0.7 W1: transcript-inversion bit measurement ----------------------
  if (cmd == "v7bits") {
    v07::BitProbeConfig bc;
    bc.specA = argVal(argc, argv, "a", "v06");
    bc.specB = argVal(argc, argv, "b", "v06");
    bc.model = argVal(argc, argv, "model", "");
    bc.games = atoi(argVal(argc, argv, "games", "100").c_str());
    bc.rotations = atoi(argVal(argc, argv, "rotations", "2").c_str());
    bc.nDet = atoi(argVal(argc, argv, "det", "64").c_str());
    bc.fromEvent = atoi(argVal(argc, argv, "from", "0").c_str());
    bc.maxQ = atoi(argVal(argc, argv, "maxq", "0").c_str());
    bc.everyK = atoi(argVal(argc, argv, "every", "1").c_str());
    bc.gain = atof(argVal(argc, argv, "gain", "1.0").c_str());
    bc.alpha = atof(argVal(argc, argv, "alpha", "0.5").c_str());
    bc.kappa = atof(argVal(argc, argv, "kappa", "3.0").c_str());
    bc.stepClip = atof(argVal(argc, argv, "stepclip", "1.25").c_str());
    bc.clip = atof(argVal(argc, argv, "clip", "5.0").c_str());
    bc.mode = atoi(argVal(argc, argv, "mode", "0").c_str());
    bc.focus = atoi(argVal(argc, argv, "focus", "1").c_str());
    bc.theta = atof(argVal(argc, argv, "theta", "0.44458").c_str());
    bc.phi = atof(argVal(argc, argv, "phi", "0.12198").c_str());
    bc.thetaInv = atof(argVal(argc, argv, "thetainv", "-1").c_str());
    bc.phiInv = atof(argVal(argc, argv, "phiinv", "-1").c_str());
    bc.seed = strtoull(argVal(argc, argv, "seed", "7012001").c_str(), nullptr, 10);
    bc.rules = rulesFrom(argc, argv);
    bc.threads = threads;
    v07::BitResult R = v07::runBitProbe(bc);
    double mean = R.inverted ? R.bitsSum / R.inverted : 0;
    double var = R.inverted > 1 ? (R.bitsSq / R.inverted - mean * mean) : 0;
    double se = R.inverted > 1 ? std::sqrt(std::max(0.0, var) / double(R.inverted)) : 0;
    bool json = argFlag(argc, argv, "json");
    if (json) {
      printf("{\"probe\":\"v7bits\",\"a\":\"%s\",\"b\":\"%s\",\"model\":\"%s\",\"seed\":%llu,"
             "\"games\":%d,\"rotations\":%d,\"det\":%d,\"asks\":%lld,\"inverted\":%lld,\"skipped\":%lld,"
             "\"bitsPerAsk\":%.4f,\"bitsSE\":%.4f,\"meanConsistentFrac\":%.4f,"
             "\"predN\":%lld,\"natsBase\":%.5f,\"natsInv\":%.5f,\"argmaxBase\":%.5f,\"argmaxInv\":%.5f,"
             "\"oracleCalls\":%.0f,\"seconds\":%.2f,\"gamesPerSec\":%.3f}\n",
             bc.specA.c_str(), bc.specB.c_str(), (bc.model.empty() ? bc.specB : bc.model).c_str(),
             (unsigned long long)bc.seed, bc.games, bc.rotations, bc.nDet,
             R.asks, R.inverted, R.skipped, mean, se, R.inverted ? R.qSum / R.inverted : 0.0,
             R.predN, R.predN ? R.natBase / R.predN : 0.0, R.predN ? R.natInv / R.predN : 0.0,
             R.predN ? double(R.hitBase) / R.predN : 0.0, R.predN ? double(R.hitInv) / R.predN : 0.0,
             R.oracleCalls, R.seconds, double(bc.games * bc.rotations) / std::max(1e-9, R.seconds));
    } else {
      printf("v7bits  observer=%s  target=%s  model=%s  seed=%llu  n=%d deals x %d\n",
             bc.specA.c_str(), bc.specB.c_str(), (bc.model.empty() ? bc.specB : bc.model).c_str(),
             (unsigned long long)bc.seed, bc.games, bc.rotations);
      printf("  target asks seen        %lld   inverted %lld   skipped %lld   det=%d\n",
             R.asks, R.inverted, R.skipped, bc.nDet);
      printf("  contraction             %.4f bits/ask  (SE %.4f)  mean surviving fraction %.4f\n",
             mean, se, R.inverted ? R.qSum / R.inverted : 0.0);
      printf("  by event index          ");
      for (int b = 0; b < 6; b++)
        printf("[%d-%d) %.2f  ", b * 20, b * 20 + 20, R.bucketN[b] ? R.bucketBits[b] / R.bucketN[b] : 0.0);
      printf("\n");
      printf("  marginals over %lld unresolved cards, scored against ground truth:\n", R.predN);
      printf("    certificates + fitted policy prior   %.5f nats   argmax %.4f\n",
             R.predN ? R.natBase / R.predN : 0.0, R.predN ? double(R.hitBase) / R.predN : 0.0);
      printf("    ... plus measured transcript inversion %.5f nats   argmax %.4f\n",
             R.predN ? R.natInv / R.predN : 0.0, R.predN ? double(R.hitInv) / R.predN : 0.0);
      printf("  oracle calls            %.0f    elapsed %.2fs  (%.2f games/s)\n",
             R.oracleCalls, R.seconds, double(bc.games * bc.rotations) / std::max(1e-9, R.seconds));
    }
    return 0;
  }

  // ---- v0.7 P2: the bank commitment digest ---------------------------------
  // A "bank" in this corpus is a seed plus a size, and the deals are generated
  // rather than stored, so sealing a bank physically means committing to a digest
  // of the material it generates.  This emits one: for each deal index the arena
  // would play, the six dealt hands and the dealer, folded into a 64-bit
  // rolling hash.  It plays no games and constructs no policy, which is what
  // lets phase 2 compute the digest of a SEALED bank without learning anything
  // about how any policy performs on it -- the only phase-2 contact with the
  // holdout, recorded as such in RESEARCH-LOG.md.
  // ---------------------------------------------------------------- v0.7 K0
  // The mechanical side-channel certification gate.  THREAT-MODEL.md 6.4
  // specifies S1-S6 and records that none of them reads only existing
  // artifacts; this is the harness for the four pass/fail ones.  See
  // engine/src/v07_side.hpp for what each test does and, more importantly, for
  // what each cannot see.
  if (cmd == "v7side") {
    if (argFlag(argc, argv, "help")) {
      std::cout <<
        "usage: fish7 v7side --a=<spec> [--b=<opponent, default v06>] --games=N --seed=<bank>\n"
        "                    [--threads=T] [--rotations=2] [--tests=s3,s4,s5,s6]\n"
        "                    [--s3nodes=3] [--s5nodes=6] [--s5draws=8] [--json] [--out=FILE]\n"
        "\n"
        "Seats THREE copies of --a as one team against --b and certifies the team\n"
        "against THREAT-MODEL.md section 8's definition of an illegal side channel.\n"
        "  S3  listening substitution   -- rule-equivalent public action swapped inside\n"
        "                                  the bit-for-bit tie group; teammate response\n"
        "                                  rate against an OPPOSING-seat control.\n"
        "  S4  stream independence      -- T10: a per-seat stream drawn independently of\n"
        "                                  the deal.  Deterministic policies must give an\n"
        "                                  identical transcript; stochastic ones must show\n"
        "                                  no paired win-rate or ask-hit-rate movement.\n"
        "  S5  posterior invariance     -- exact posterior resample (DealDP + satisfies)\n"
        "                                  of the other five hands; P(hit|truth) against\n"
        "                                  P(hit|posterior).\n"
        "  S6  seat isolation           -- every decision of every certified seat, over all\n"
        "                                  four decision types of 6.2 PLUS bestGuess, rebuilt\n"
        "                                  from (own hand, public stream, reset seed) alone on\n"
        "                                  a fresh thread and required to match.\n"
        "positive controls: --a=v07x:cheat=seed | v07x:cheat=shared | v07x:cheat=conv\n";
      return 0;
    }
    v07side::SideConfig sc;
    sc.specA = argVal(argc, argv, "a", "v06");
    sc.specB = argVal(argc, argv, "b", "v06");
    sc.games = atoi(argVal(argc, argv, "games", "200").c_str());
    sc.rotations = atoi(argVal(argc, argv, "rotations", "2").c_str());
    sc.seed = strtoull(argVal(argc, argv, "seed", "7030001").c_str(), nullptr, 10);
    sc.threads = threads;
    sc.rules = rulesFrom(argc, argv);
    sc.s3nodes = atoi(argVal(argc, argv, "s3nodes", "3").c_str());
    sc.s5nodes = atoi(argVal(argc, argv, "s5nodes", "6").c_str());
    sc.s5draws = atoi(argVal(argc, argv, "s5draws", "8").c_str());
    sc.reconInline = argFlag(argc, argv, "reconinline");   // diagnostic, see SideConfig
    sc.freshAgents = argFlag(argc, argv, "freshagents");  // diagnostic, see SideConfig
    { std::string t = argVal(argc, argv, "tests", "");
      if (!t.empty()) {
        sc.s3 = t.find("s3") != std::string::npos;
        sc.s4 = t.find("s4") != std::string::npos;
        sc.s5 = t.find("s5") != std::string::npos;
        sc.s6 = t.find("s6") != std::string::npos;
      } }
    if (!mixSeedRoundTrip()) {
      // The gate's own premise: E-1 asserts mixSeed(.,b) is a bijection.  If the
      // inverse ever stops round-tripping, the seed cheat is not a cheat and the
      // S4/S5 calibration is void, so refuse rather than report a hollow PASS.
      fprintf(stderr, "fish: mixSeed inverse failed its round trip -- S4/S5 calibration is void\n");
      return 6;
    }
    v07side::SideStats T = v07side::runSide(sc);
    v07side::GateReport G = v07side::judge(sc, T);
    if (argFlag(argc, argv, "json")) { v07side::jsonSide(sc, T, G, std::cout); std::cout << "\n"; }
    else                             { v07side::printSide(sc, T, G, std::cout); }
    std::string out = argVal(argc, argv, "out", "");
    if (!out.empty()) {
      std::ofstream f(out, std::ios::app);
      v07side::jsonSide(sc, T, G, f); f << "\n";
    }
    return G.allPass ? 0 : 1;
  }

  if (cmd == "bankdigest") {
    uint64_t seed = strtoull(argVal(argc, argv, "seed", "0").c_str(), nullptr, 10);
    int deals = atoi(argVal(argc, argv, "deals", "24000").c_str());
    Rules r = rulesFrom(argc, argv);
    { std::string why;
      if (!seedUsable(seed, why)) { fprintf(stderr, "fish: %s\n", why.c_str()); return 5; } }
    uint64_t h = 0xcbf29ce484222325ull;
    auto mix = [&](uint64_t v) { h ^= v; h *= 0x100000001b3ull; h ^= h >> 29; };
    GameState g{};
    for (int i = 0; i < deals; i++) {
      uint64_t s = mixSeed(seed, uint64_t(i) * 2654435761ull + 1);
      dealCards(g, s, r.deckSets);
      for (int p = 0; p < NPLAY; p++) mix(g.dealt[p]);
      mix(uint64_t(g.dealer));
    }
    printf("{\"probe\":\"bankdigest\",\"seed\":%llu,\"deals\":%d,\"deckSets\":%d,\"digest\":\"%016llx\"}\n",
           (unsigned long long)seed, deals, r.deckSets, (unsigned long long)h);
    return 0;
  }

  // ---- v0.7 phase 3, candidate K5: the learned component -----------------
  if (cmd == "v7learn") {
    std::string mode = argVal(argc, argv, "mode", "capture");
    if (mode == "capture") {
      v07learn::CapConfig cc;
      cc.spec = argVal(argc, argv, "a", cc.spec.c_str());
      cc.opp  = argVal(argc, argv, "b", "");
      cc.seed = strtoull(argVal(argc, argv, "seed", "7030004").c_str(), nullptr, 10);
      cc.games = atoi(argVal(argc, argv, "games", "400").c_str());
      cc.rotations = atoi(argVal(argc, argv, "rotations", "2").c_str());
      cc.threads = threads;
      cc.out = argVal(argc, argv, "out", "");
      return v07learn::runCapture(cc);
    }
    fprintf(stderr, "usage: fish7 v7learn --mode=capture --a=<search spec> --seed=<training bank>\n"
                    "                     --games=N [--rotations=2] [--threads=T] --out=FILE\n");
    return 2;
  }

  if (cmd == "pathology") {
    PathologyConfig pc;
    pc.specA = argVal(argc, argv, "a", "v04");
    pc.specB = argVal(argc, argv, "b", "v04");
    pc.games = atoi(argVal(argc, argv, "games", "200").c_str());
    pc.rotations = atoi(argVal(argc, argv, "rotations", "2").c_str());
    pc.seed = strtoull(argVal(argc, argv, "seed", "31").c_str(), nullptr, 10);
    pc.rules = rulesFrom(argc, argv);
    pc.threads = threads;
    PathologyStats st = runPathology(pc);
    std::cout << pc.specA << " vs " << pc.specB << "\n";
    printPathology(st, std::cout);
    return 0;
  }

  if (cmd == "blockalias") {
    Rules r = rulesFrom(argc, argv);
    AliasReport rp = blockAliasCheck(argVal(argc, argv, "a", "v04"),
                                     atoi(argVal(argc, argv, "games", "20").c_str()),
                                     strtoull(argVal(argc, argv, "seed", "31").c_str(), nullptr, 10), r);
    std::cout << "same BlockDP object, same query, before/after a second build() elsewhere\n";
    std::cout << "  checks " << rp.checks
              << "   QUERY mismatches " << rp.mismatches
              << "   worst |delta| " << rp.worst
              << "   (raw shared-pool field reads that differ: " << rp.rawMismatches << ")\n";
    return 0;
  }

  if (cmd == "forcedprobe") {
    FEConfig fc;
    fc.specA = argVal(argc, argv, "a", "v04");
    fc.specB = argVal(argc, argv, "b", "v04");
    fc.games = atoi(argVal(argc, argv, "games", "300").c_str());
    fc.rotations = atoi(argVal(argc, argv, "rotations", "2").c_str());
    fc.seed = strtoull(argVal(argc, argv, "seed", "31").c_str(), nullptr, 10);
    fc.rules = rulesFrom(argc, argv);
    fc.threads = threads;
    FEStats st = runForcedProbe(fc);
    std::cout << fc.specA << " vs " << fc.specB << "  seed=" << fc.seed << "\n";
    printForcedProbe(st, std::cout);
    std::string csv = argVal(argc, argv, "csv", "");
    if (!csv.empty()) {
      std::ofstream f(csv);
      f << "game,rot,ordinal,nactive,set,declarer,declhand,rung,th,conf,correct,matched,"
           "violmask,violcap,violdisj,maxover,zeropost,pnamed,pbest,ptrue,bestistruth,nunknown,"
           "ncapseats,nnamedseats,ntrueseats,margspread,walloc,wok,wfeas,wtruth,named,truth\n";
      for (const auto& d : st.decls) {
        f << d.gameId << "," << d.rot << "," << d.ordinal << "," << d.nActiveAtDecl << ","
          << d.set << "," << d.declarer << "," << d.declHand << "," << d.rung << ","
          << d.th << "," << d.conf << "," << int(d.correct) << "," << int(d.predictedMatched) << ","
          << int(d.violMask) << "," << int(d.violCap) << "," << int(d.violDisj) << "," << d.maxOver << ","
          << int(d.zeroPost) << "," << d.pNamed << "," << d.pBest << "," << d.pTrue << ","
          << int(d.bestIsTruth) << "," << d.nUnknown << "," << d.nCapSeats << ","
          << d.nNamedSeats << "," << d.nTrueSeats << "," << d.margSpread << ","
          << d.wAlloc << "," << int(d.wOk) << "," << int(d.wFeasible) << "," << int(d.wIsTruth) << ",";
        for (int i = 0; i < SETSZ; i++) f << d.named[i];
        f << ",";
        for (int i = 0; i < SETSZ; i++) f << d.truth[i];
        f << "\n";
      }
      fprintf(stderr, "wrote %s (%zu rows)\n", csv.c_str(), st.decls.size());
    }
    return 0;
  }

  if (cmd == "coord") {
    using namespace fish::probecoord;
    CoordConfig cc;
    cc.specA = argVal(argc, argv, "a", "v04");
    cc.specB = argVal(argc, argv, "b", "v04");
    cc.games = atoi(argVal(argc, argv, "games", "300").c_str());
    cc.rotations = atoi(argVal(argc, argv, "rotations", "2").c_str());
    cc.seed = strtoull(argVal(argc, argv, "seed", "31").c_str(), nullptr, 10);
    cc.rules = rulesFrom(argc, argv);
    cc.threads = threads;
    std::string pol = argVal(argc, argv, "pass", "unilateral");
    cc.cc.policy = pol == "oracle" ? PassPolicy::Oracle
                 : pol == "ladder" ? PassPolicy::Ladder
                 : pol == "low"    ? PassPolicy::LowSeat
                 : pol == "cards"  ? PassPolicy::MostCards
                 : PassPolicy::Unilateral;
    if (argFlag(argc, argv, "both-teams")) cc.cc.policyTeam = -2;
    cc.cc.measure = !argFlag(argc, argv, "no-measure");
    cc.cc.leak = argFlag(argc, argv, "leak");
    { std::string L = argVal(argc, argv, "rungs", "");
      if (!L.empty()) { std::stringstream ss(L); std::string tok; int i = 0;
        while (std::getline(ss, tok, '|') && i < MAXRUNG) cc.cc.rung[i++] = atof(tok.c_str());
        cc.cc.nRung = i; } }
    { std::string L = argVal(argc, argv, "forcedth", "");
      if (!L.empty()) { std::stringstream ss(L); std::string tok; int i = 0;
        while (std::getline(ss, tok, '|') && i < 8) cc.rules.forcedTh[i++] = atof(tok.c_str());
        cc.rules.nForcedTh = i; } }
    std::cout << cc.specA << " (pass=" << pol << ") vs " << cc.specB << " (pass=unilateral)\n";
    std::vector<int> perGame;
    std::string dump = argVal(argc, argv, "dump", "");
    if (!dump.empty()) { perGame.assign(size_t(cc.games) * cc.rotations, 0); cc.perGameA = &perGame; }
    CoordStats st = runCoord(cc);
    if (!dump.empty()) { std::ofstream f(dump); for (size_t i = 0; i < perGame.size(); i++) f << perGame[i] << "\n"; }
    printCoord(st, cc, std::cout);
    return 0;
  }

  if (cmd == "deadlock") {
    DeadlockCfg dc;
    dc.spec = argVal(argc, argv, "spec", "v04");
    dc.games = atoi(argVal(argc, argv, "games", "60").c_str());
    dc.rotations = atoi(argVal(argc, argv, "rotations", "2").c_str());
    dc.seed = strtoull(argVal(argc, argv, "seed", "31").c_str(), nullptr, 10);
    dc.minEvents = atoi(argVal(argc, argv, "minev", "300").c_str());
    dc.dump = atoi(argVal(argc, argv, "dump", "4").c_str());
    dc.stride = atoi(argVal(argc, argv, "stride", "40").c_str());
    dc.maxStates = atoi(argVal(argc, argv, "states", "3").c_str());
    dc.h2h = atoi(argVal(argc, argv, "h2h", "0").c_str());
    dc.rules = rulesFrom(argc, argv);
    runDeadlockProbe(dc, std::cout);
    return 0;
  }

  // ---- publish one of our policies over the bot protocol ------------------
  // `fish botserve --policy=<spec>` is what a packaged FishLab bot runs; see
  // docs/BOT_PACKAGE.md and scripts/package-bot.sh.
  if (cmd == "botserve") {
    std::string spec = argVal(argc, argv, "policy", "");
    if (spec.empty()) {
      fprintf(stderr, "usage: fish botserve --policy=<spec> [--name=NAME] [--botversion=V]\n");
      return 2;
    }
    return fish::botserve::run(spec, argVal(argc, argv, "name", "FishLab"),
                               argVal(argc, argv, "botversion", "1.0"));
  }

  // ---- uploaded bot packages ---------------------------------------------
  // The same library `fish serve` offers on the setup screen, from the command
  // line: docs/BOT_PACKAGE.md is the format, and a package installed either way
  // is seatable and measurable as `bot:<id>`.
  if (cmd == "bots") {
    std::vector<std::string> pos;
    for (int i = 2; i < argc; i++) if (strncmp(argv[i], "--", 2) != 0) pos.push_back(argv[i]);
    const std::string sub = pos.empty() ? std::string("list") : pos[0];
    const std::string arg = pos.size() > 1 ? pos[1] : std::string();
    std::string dir = argVal(argc, argv, "dir", "");
    if (!dir.empty()) fish::botpkg::registry().setRoot(dir);
    auto& reg = fish::botpkg::registry();
    reg.rescan();

    if (sub == "list") {
      auto all = reg.all();
      printf("bots in %s\n", reg.rootDir().c_str());
      if (all.empty()) printf("  (none installed -- fish bots add <package.zip>)\n");
      for (const auto& b : all)
        printf("  %-20s %-24s %-16s %s%s\n", fish::botpkg::specFor(b).c_str(),
               b.man.name.c_str(), b.man.protocol.c_str(),
               b.man.venv ? (b.prepared ? "[prepared] " : "[needs prepare] ") : "",
               b.note.c_str());
      return 0;
    }
    if (sub == "add") {
      if (arg.empty()) { fprintf(stderr, "usage: fish bots add <package.zip> [--replace]\n"); return 2; }
      std::string zip = fish::botpkg::readWhole(arg);
      if (zip.empty()) { fprintf(stderr, "fish bots: cannot read %s\n", arg.c_str()); return 2; }
      std::string id, why;
      if (!reg.install(zip, argVal(argc, argv, "who", "the host"),
                       argFlag(argc, argv, "replace"), id, why)) {
        fprintf(stderr, "fish bots: %s\n", why.c_str());
        return 1;
      }
      fish::botpkg::Installed b;
      reg.get(id, b);
      printf("installed '%s' as %s\n", b.man.name.c_str(), fish::botpkg::specFor(b).c_str());
      if (b.man.venv && !b.prepared) printf("  it wants a virtualenv: fish bots prepare %s\n", id.c_str());
      printf("  check it: fish bots check %s\n", id.c_str());
      return 0;
    }
    if (sub == "remove") {
      std::string why;
      if (!reg.erase(arg, why)) { fprintf(stderr, "fish bots: %s\n", why.c_str()); return 1; }
      printf("removed %s\n", arg.c_str());
      return 0;
    }
    if (sub == "prepare") {
      std::string log, why;
      if (!reg.prepare(arg, log, why)) {
        fprintf(stderr, "%s\nfish bots: %s\n", log.c_str(), why.c_str());
        return 1;
      }
      printf("%s\nprepared %s\n", log.c_str(), arg.c_str());
      return 0;
    }
    if (sub == "check") {
      fish::botpkg::Installed b;
      if (!reg.get(arg, b)) { fprintf(stderr, "fish bots: no bot called '%s'\n", arg.c_str()); return 1; }
      std::string report, why;
      bool ok = fish::botcheck::run(b, report, why);
      printf("%s\n", report.c_str());
      return ok ? 0 : 1;
    }
    fprintf(stderr, "usage: fish bots <list|add|check|prepare|remove> [arg] [--dir=DIR]\n");
    return 2;
  }

#ifndef FISH_NO_SERVE
  if (cmd == "serve") {
    ServeOptions o;
    o.port         = atoi(argVal(argc, argv, "port", "8173").c_str());
    o.webDir       = argVal(argc, argv, "web", "");
    o.bindAll      = argFlag(argc, argv, "lan");
    o.publicTunnel = argFlag(argc, argv, "public");
    o.forceAuth    = argFlag(argc, argv, "auth");
    o.tunnel       = argVal(argc, argv, "tunnel", "auto");
    o.invite       = argVal(argc, argv, "invite", "");
    o.botsDir      = argVal(argc, argv, "bots", "");
    o.lockBots     = argFlag(argc, argv, "lock-bots");
    return runServe(o, argv[0]);
  }

#endif
  if (cmd == "dumpvalue") {
    DumpConfig dc;
    dc.specA = argVal(argc, argv, "a", "v04");
    dc.specB = argVal(argc, argv, "b", "v04");
    dc.games = atoi(argVal(argc, argv, "games", "400").c_str());
    dc.rotations = atoi(argVal(argc, argv, "rotations", "2").c_str());
    dc.seed = strtoull(argVal(argc, argv, "seed", "31415").c_str(), nullptr, 10);
    dc.rules = rulesFrom(argc, argv);
    dc.threads = threads;
    dc.out = argVal(argc, argv, "out", "rows.csv");
    DumpRows rows;
    runDump(dc, rows, NVFEAT);
    int rc = writeDump(dc, rows, NVFEAT);
    fprintf(stderr, "rows=%zu games=%d out=%s\n", rows.y.size(), dc.games * dc.rotations, dc.out.c_str());
    return rc;
  }

  // --- appended (P3 deception probe) ---------------------------------------
  if (cmd == "deceit") {
    DeceitConfig dc;
    dc.measured  = argVal(argc, argv, "m", "v04");
    dc.deceptive = argVal(argc, argv, "d", "silent");
    dc.control   = argVal(argc, argv, "ctrl", "v04");
    dc.games     = atoi(argVal(argc, argv, "games", "200").c_str());
    dc.seed      = strtoull(argVal(argc, argv, "seed", "4242").c_str(), nullptr, 10);
    dc.stride    = atoi(argVal(argc, argv, "stride", "1").c_str());
    dc.rules     = rulesFrom(argc, argv);
    dc.threads   = threads;
    { std::string ds = argVal(argc, argv, "dseats", "1");
      int mask = 0;
      std::stringstream ss(ds); std::string tok;
      while (std::getline(ss, tok, ',')) if (!tok.empty()) mask |= 1 << atoi(tok.c_str());
      dc.dseatMask = mask; }
    decCost().reset();
    DeceitStats st = runDeceit(dc);
    std::cout << "measured " << dc.measured << "  vs  seats{";
    for (int p = 0; p < NPLAY; p++) if (dc.dseatMask & (1 << p)) std::cout << p << " ";
    std::cout << "}=" << dc.deceptive << ", other opponents=" << dc.control << "\n";
    printDeceit(st, std::cout);
    return 0;
  }

  // --- appended (P5 human-strategy channel probe) ---------------------------
  if (cmd == "humanchan") {
    HumanChanConfig pc;
    pc.specA = argVal(argc, argv, "a", "v04");
    pc.specB = argVal(argc, argv, "b", "v04");
    pc.games = atoi(argVal(argc, argv, "games", "200").c_str());
    pc.rotations = atoi(argVal(argc, argv, "rotations", "2").c_str());
    pc.seed = strtoull(argVal(argc, argv, "seed", "31").c_str(), nullptr, 10);
    pc.rules = rulesFrom(argc, argv);
    pc.threads = threads;
    HumanChanStats st = runHumanChan(pc);
    std::cout << pc.specA << " vs " << pc.specB << "\n";
    printHumanChan(st, std::cout);
    return 0;
  }


  // --- appended (P4 adversarial policy-correctness probe) -------------------
  if (cmd == "p4probe") {
    std::string spec = argVal(argc, argv, "a", "p4:instr=1");
    int games = atoi(argVal(argc, argv, "games", "200").c_str());
    uint64_t sd = strtoull(argVal(argc, argv, "seed", "31").c_str(), nullptr, 10);
    Rules r = rulesFrom(argc, argv);
    MatchStats ms;
    p4::P4Stats st = p4::runProbe(spec, games, sd, r, &ms);
    printf("spec=%s games=%d seed=%llu\n", spec.c_str(), games, (unsigned long long)sd);
    printf("  events/game            %.1f   limitHits %d\n", double(ms.events) / games, ms.limitHits);
    printf("  declarations           %lld  wrong %.2f%%\n", ms.decl[0] + ms.decl[1],
           100.0 * double((ms.decl[0] + ms.decl[1]) - (ms.declCorrect[0] + ms.declCorrect[1])) /
           std::max(1.0, double(ms.decl[0] + ms.decl[1])));
    printf("  forced declarations    %lld  wrong %.2f%%\n", ms.fdecl[0] + ms.fdecl[1],
           100.0 * double((ms.fdecl[0] + ms.fdecl[1]) - (ms.fdeclCorrect[0] + ms.fdeclCorrect[1])) /
           std::max(1.0, double(ms.fdecl[0] + ms.fdecl[1])));
    printf("[expectedRun]\n");
    printf("  calls                  %lld\n", st.runCalls);
    printf("  own entry dropped      %lld (%.2f%%)\n", st.runSkipFired, 100.0 * st.runSkipFired / std::max(1LL, st.runCalls));
    printf("  own entry DOUBLE-COUNTED %lld (%.2f%%)\n", st.runSelfCounted, 100.0 * st.runSelfCounted / std::max(1LL, st.runCalls));
    printf("  mean |f18 shipped-fixed| %.5f   max %.5f\n", st.runAbsErr / std::max(1LL, st.runCalls), st.runMaxErr);
    printf("[threatOf/exposureOf]\n");
    printf("  threat set-evals       %lld  activity=0 %.2f%%  activity=1 %.2f%%  mean %.4f\n",
           st.threatCalls, 100.0 * st.threatActivity0 / std::max(1LL, st.threatCalls),
           100.0 * st.threatActivityFull / std::max(1LL, st.threatCalls),
           st.threatActivitySum / std::max(1LL, st.threatCalls));
    printf("  exposure calls         %lld  saturated(=1) %.2f%%  mean %.4f\n",
           st.expCalls, 100.0 * st.expSat / std::max(1LL, st.expCalls), st.expSum / std::max(1LL, st.expCalls));
    printf("[evaluateSet Fast]\n");
    printf("  evaluations            %lld\n", st.evalFast);
    printf("  pAlloc > cheap         %lld (%.2f%%)  mean excess %.4f  max %.4f\n",
           st.evalAllocGtCheap, 100.0 * st.evalAllocGtCheap / std::max(1LL, st.evalFast),
           st.evalAllocCheapExcess / std::max(1LL, st.evalAllocGtCheap), st.evalMaxExcess);
    printf("  teamFloor passed ONLY because of the max() %lld\n", st.evalFloorSavedByMax);
    printf("[declareByValue]\n");
    printf("  calls %lld  fired %lld (%.2f%%)\n", st.dbvCalls, st.dbvTrue,
           100.0 * st.dbvTrue / std::max(1LL, st.dbvCalls));
    printf("  breakeven pAlloc: mean %.4f  min %.4f  max %.4f  <0.70 %.2f%%  <0.60 %.2f%%\n",
           st.dbvThreshSum / std::max(1LL, st.dbvCalls), st.dbvThreshMin, st.dbvThreshMax,
           100.0 * st.dbvBelow70 / std::max(1LL, st.dbvCalls), 100.0 * st.dbvBelow60 / std::max(1LL, st.dbvCalls));
    printf("[stale aggregates in proposeDeclaration]\n");
    printf("  declaration opportunities %lld  (press0 %lld press1 %lld press2 %lld)\n",
           st.declOpps, st.press0, st.press1, st.press2);
    printf("  opportunities with a STALE belief %lld (%.2f%%)\n", st.staleOpps,
           100.0 * st.staleOpps / std::max(1LL, st.declOpps));
    printf("  mean max|eH stale-fresh| %.4f   max %.4f\n",
           st.staleEHSum / std::max(1LL, st.staleOpps), st.staleEHMax);
    printf("  declareNow decisions flipped by the staleness %lld\n", st.staleDeclFlip);
    printf("[ask-score decision influence: mean spread of each term across the candidate list]\n");
    { static const char* fn[20] = {"hit p","hit p^2","certain hit","own progress","team control",
        "lock completion","continuation","completion bonus","reply threat","info leak","target hand",
        "empties target","repeats set","known team cards","location entropy","team owns set",
        "exposure on miss","trailing pressure","runway","leak magnitude"};
      p4::P4Config dflt;
      double tot = 0; for (int j = 0; j < 20; j++) tot += st.featRange[j];
      for (int j = 0; j < 20; j++)
        printf("  f%-2d %-20s w=%8.4f  mean spread %.4f  (%.1f%% of linear)\n", j, fn[j],
               dflt.w[j], st.featRange[j] / std::max(1LL, st.featDecisions),
               100.0 * st.featRange[j] / std::max(1e-9, tot));
      printf("  f8+f16 combined spread %.4f (vs %.4f + %.4f separately)\n",
             st.f8f16Range / std::max(1LL, st.featDecisions),
             st.featRange[8] / std::max(1LL, st.featDecisions), st.featRange[16] / std::max(1LL, st.featDecisions));
      printf("  linear total spread %.4f   one-ply EV spread %.4f   two-ply add-on spread %.4f\n",
             st.linRange / std::max(1LL, st.featDecisions), st.evRange / std::max(1LL, st.featDecisions),
             st.twoPlyRange / std::max(1LL, st.featDecisions)); }
    printf("[declareNow branch that authorised a candidate declaration]\n");
    printf("  press>=2 (unconditional) %lld   press>=1 && pAlloc>=0.5 %lld\n", st.brPress2, st.brPress1);
    printf("  urgent && pAlloc>=declThreshold %lld   urgent && LOCKED && pAlloc>=0.5 %lld\n", st.brUrgThr, st.brUrgLocked);
    printf("  declareByValue %lld   non-value paths %lld\n", st.brValue, st.brOther);
    printf("[declaration pre-gate / bypass]\n");
    printf("  (opportunity,set) pairs %lld   cheapTeamProb below the gate %lld (%.2f%%)\n",
           st.gateSeen, st.gateWouldReject, 100.0 * st.gateWouldReject / std::max(1LL, st.gateSeen));
    printf("  opportunities with bypass on %lld, of which bypass actually re-admitted a set %lld (%.2f%%)\n",
           st.bypassOpps, st.bypassLoadBearing, 100.0 * st.bypassLoadBearing / std::max(1LL, st.bypassOpps));
    printf("[searchTopK]\n");
    printf("  ask decisions %lld   two-ply rerank changed the pick %lld (%.2f%%)\n",
           st.askDecisions, st.runChangedPick, 100.0 * st.runChangedPick / std::max(1LL, st.askDecisions));
    return 0;
  }

  if (cmd == "p4match") {
    MatchConfig mc;
    mc.specA = argVal(argc, argv, "a", "p4");
    mc.specB = argVal(argc, argv, "b", "v04");
    mc.games = atoi(argVal(argc, argv, "games", "600").c_str());
    mc.seed = strtoull(argVal(argc, argv, "seed", "20260821").c_str(), nullptr, 10);
    mc.rules = rulesFrom(argc, argv);
    mc.threads = threads;
    mc.rotations = atoi(argVal(argc, argv, "rotations", "2").c_str());
    MatchStats st = p4::runMatchP4(mc);
    double m, lo, hi;
    clusterBootstrap(st.paired, mc.rotations, m, lo, hi);
    long long dA = st.decl[0], cA = st.declCorrect[0], dB = st.decl[1], cB = st.declCorrect[1];
    long long fA = st.fdecl[0], fcA = st.fdeclCorrect[0];
    printf("A=%s  B=%s  deals=%d rot=%d seed=%llu\n", mc.specA.c_str(), mc.specB.c_str(),
           st.games, mc.rotations, (unsigned long long)mc.seed);
    printf("  A win %.2f%% [%.2f, %.2f]   sets %lld-%lld   events/game %.1f   %.1fs (%.1f g/s)\n",
           100 * m, 100 * lo, 100 * hi, st.sets[0], st.sets[1],
           double(st.events) / (st.games * mc.rotations), st.seconds,
           st.games * mc.rotations / std::max(1e-9, st.seconds));
    printf("  A decl %lld wrong %.2f%%  (forced %lld wrong %.2f%%)   B decl %lld wrong %.2f%%\n",
           dA, 100.0 * double(dA - cA) / std::max(1.0, double(dA)),
           fA, 100.0 * double(fA - fcA) / std::max(1.0, double(fA)),
           dB, 100.0 * double(dB - cB) / std::max(1.0, double(dB)));
    printf("  limit-hit games %d/%d\n", st.limitHits, st.games * mc.rotations);
    return 0;
  }

  if (cmd == "p4horizon") {
    std::string spec = argVal(argc, argv, "a", "p4");
    int games = atoi(argVal(argc, argv, "games", "300").c_str());
    uint64_t sd = strtoull(argVal(argc, argv, "seed", "31").c_str(), nullptr, 10);
    int hz = atoi(argVal(argc, argv, "horizon", "220").c_str());
    int cut = atoi(argVal(argc, argv, "deadcut", "6").c_str());
    Rules r = rulesFrom(argc, argv);
    p4::HorizonStats hs = p4::runHorizon(spec, games, sd, r, hz, cut);
    printf("spec=%s games=%lld horizon=%d deadRunCut=%d\n", spec.c_str(), hs.games, hz, cut);
    printf("  games reaching the horizon        %lld (%.2f%%)\n", hs.reached, 100.0 * hs.reached / std::max(1LL, hs.games));
    printf("    of which deadlocked (run>=%d)   %lld (%.2f%% of reached)\n", cut, hs.reachedDeadlocked,
           100.0 * hs.reachedDeadlocked / std::max(1LL, hs.reached));
    printf("    of which NOT deadlocked         %lld (%.2f%% of reached)\n", hs.reachedHealthy,
           100.0 * hs.reachedHealthy / std::max(1LL, hs.reached));
    printf("  voluntary declarations before the horizon %lld, wrong %.2f%%\n", hs.declPre,
           100.0 * hs.declPreWrong / std::max(1LL, hs.declPre));
    printf("  voluntary declarations at/after  the horizon %lld, wrong %.2f%%\n", hs.declPost,
           100.0 * hs.declPostWrong / std::max(1LL, hs.declPost));
    printf("    of those, in NON-deadlocked games %lld, wrong %.2f%%\n", hs.declPostHealthy,
           100.0 * hs.declPostHealthyWrong / std::max(1LL, hs.declPostHealthy));
    printf("  forced-endgame declarations %lld, wrong %.2f%%\n", hs.forced,
           100.0 * hs.forcedWrong / std::max(1LL, hs.forced));
    printf("  wrong voluntary declarations: team DID hold all six (allocation misnamed) %lld;"
           " an opponent held one or more %lld\n", hs.wrongTeamOwnedAll, hs.wrongOppHeldSome);
    printf("  mean cards of the six actually held by the declaring team, on a WRONG declaration: %.2f\n",
           double(hs.ourHeldOnWrong) / std::max(1LL, hs.wrongTeamOwnedAll + hs.wrongOppHeldSome));
    printf("  stated confidence (pAlloc) bucket -> wrong rate:\n");
    for (int b = 0; b < 10; b++) if (hs.confN[b])
      printf("    [%.1f,%.1f)  n=%lld  wrong %.2f%%\n", b / 10.0, (b + 1) / 10.0, hs.confN[b],
             100.0 * hs.confW[b] / hs.confN[b]);
    return 0;
  }

  if (cmd == "declcard") {
    int games = atoi(argVal(argc, argv, "games", "150").c_str());
    uint64_t sd = strtoull(argVal(argc, argv, "seed", "31").c_str(), nullptr, 10);
    int mode = atoi(argVal(argc, argv, "mode", "1").c_str());
    Rules r = rulesFrom(argc, argv);
    auto st = fish::declcard::runDeclCard(games, sd, r, mode);
    double n = std::max(1LL, st.dbvCalls);
    printf("declcard games=%d seed=%llu mode=%d (1=posterior split, 2=fixed 4.21/1.79)\n",
           games, (unsigned long long)sd, mode);
    printf("  declareByValue calls (both variants took the value branch) %lld\n", st.dbvCalls);
    printf("  shipped verdict=declare %lld (%.2f%%)   corrected verdict=declare %lld (%.2f%%)\n",
           st.shipDeclare, 100.0*st.shipDeclare/n, st.fixDeclare, 100.0*st.fixDeclare/n);
    printf("  flips wait->declare %lld (%.3f%%)   declare->wait %lld (%.3f%%)\n",
           st.flipW2D, 100.0*st.flipW2D/n, st.flipD2W, 100.0*st.flipD2W/n);
    printf("  |dvWrong| per call: mean %.5f  max %.5f      (|declareMargin| = %.5f)\n",
           st.sumBranchErr/n, st.maxBranchErr, 0.03420);
    printf("  |dvDeclare| = (1-pAlloc)*|dvWrong|: mean %.5f  max %.5f\n",
           st.sumEvErr/n, st.maxEvErr);
    printf("  mean shipped slack (vDeclare - vWait - margin) %.5f\n", st.sumSlack/n);
    printf("  mean corrected E[our cards of the six] used in the wrong branch %.3f (n=%lld)\n",
           st.sumOurW/std::max(1LL, st.nOurW), st.nOurW);
    printf("  [faithfulness] voluntary decls %lld correct %lld  events %lld\n", st.setsA, st.setsB, st.evts);
    printf("  pAlloc bucket -> calls / flips:\n");
    for (int b = 0; b < 10; b++) if (st.pallocBucketN[b])
      printf("    [%.1f,%.1f)  n=%lld  flips=%lld (%.2f%%)\n", b/10.0, (b+1)/10.0,
             st.pallocBucketN[b], st.pallocBucketFlip[b],
             100.0*st.pallocBucketFlip[b]/st.pallocBucketN[b]);
    return 0;
  }

  if (cmd == "p4blockcmp") {
    int games = atoi(argVal(argc, argv, "games", "40").c_str());
    uint64_t sd = strtoull(argVal(argc, argv, "seed", "31").c_str(), nullptr, 10);
    double floorv = atof(argVal(argc, argv, "floor", "0.7925").c_str());
    Rules r = rulesFrom(argc, argv);
    std::string bspec = argVal(argc, argv, "a", "p4");
    p4::BlockCmp bc = p4::runBlockCmp(games, sd, r, floorv, bspec);
    printf("games=%d seed=%llu teamFloor=%.4f   samples=%lld\n", games, (unsigned long long)sd, floorv, bc.n);
    printf("  Fast pAlloc > Fast cheap          %lld (%.2f%%)\n", bc.allocOverTeamFast,
           100.0 * bc.allocOverTeamFast / std::max(1LL, bc.n));
    printf("  Fast pAlloc > EXACT P(team owns)  %lld (%.2f%%)   <- impossible for a correct pair\n",
           bc.allocOverTeamExact, 100.0 * bc.allocOverTeamExact / std::max(1LL, bc.n));
    printf("  cheap - exact pTeam:   mean signed %+.4f  mean abs %.4f  max abs %.4f  cheap>exact %.2f%%\n",
           bc.sumCheapSigned / std::max(1LL, bc.n), bc.sumCheapErr / std::max(1LL, bc.n), bc.maxCheapErr,
           100.0 * bc.cheapOverExact / std::max(1LL, bc.n));
    printf("  Fast pAlloc - exact alloc: mean signed %+.4f\n", bc.sumAllocSigned / std::max(1LL, bc.n));
    printf("  |max(cheap,pAlloc) - exact pTeam| mean %.4f  max %.4f\n", bc.sumMaxErr / std::max(1LL, bc.n), bc.maxMaxErr);
    printf("  |Fast pAlloc - exact best alloc|  mean %.4f  max %.4f   >0.10 in %lld (%.2f%%)\n",
           bc.sumAllocErr / std::max(1LL, bc.n), bc.maxAllocErr, bc.declFlip,
           100.0 * bc.declFlip / std::max(1LL, bc.n));
    printf("  teamFloor passed by the shipped value but NOT by the exact %lld (%.2f%%)\n",
           bc.gateFlipUp, 100.0 * bc.gateFlipUp / std::max(1LL, bc.n));
    printf("  teamFloor failed by the shipped value but passed by the exact %lld (%.2f%%)\n",
           bc.gateFlipDown, 100.0 * bc.gateFlipDown / std::max(1LL, bc.n));
    printf("  half-suits called LOCKED (pTeam>.9995) %lld, of which not locked exactly %lld (%.2f%%)\n",
           bc.lockClaim, bc.lockClaimWrong, 100.0 * bc.lockClaimWrong / std::max(1LL, bc.lockClaim));
    return 0;
  }

  if (cmd == "shadow") {
    std::string base = argVal(argc, argv, "base", "v04");
    std::string vs = argVal(argc, argv, "variants", "");
    std::vector<std::string> variants;
    { std::stringstream ss(vs); std::string it; while (std::getline(ss, it, ';')) if (!it.empty()) variants.push_back(it); }
    int g = atoi(argVal(argc, argv, "games", "60").c_str());
    uint64_t sd = strtoull(argVal(argc, argv, "seed", "99001").c_str(), nullptr, 10);
    Rules r = rulesFrom(argc, argv);
    ShadowStats st;
    runShadow(base, variants, g, sd, r, threads, st);
    printf("base=%s games=%d seed=%llu\n", base.c_str(), g, (unsigned long long)sd);
    for (size_t i = 0; i < variants.size(); i++)
      printf("  %-46s ask-decisions %lld/%lld differ (%.4f%%)   decl-decisions %lld/%lld differ (%.4f%%)\n",
             variants[i].c_str(), st.askDiff[i], st.askTotal[i],
             st.askTotal[i] ? 100.0 * double(st.askDiff[i]) / double(st.askTotal[i]) : 0.0,
             st.declDiff[i], st.declTotal[i],
             st.declTotal[i] ? 100.0 * double(st.declDiff[i]) / double(st.declTotal[i]) : 0.0);
    return 0;
  }

  if (cmd == "litpath") {
    LitPathCfg c;
    c.games = atoi(argVal(argc, argv, "games", "200").c_str());
    c.rotations = atoi(argVal(argc, argv, "rotations", "2").c_str());
    c.seed = strtoull(argVal(argc, argv, "seed", "777001").c_str(), nullptr, 10);
    c.rules = rulesFrom(argc, argv);
    c.threads = threads;
    c.liveOnly = atoi(argVal(argc, argv, "liveonly", "0").c_str());
    c.timeCost = atof(argVal(argc, argv, "timecost", "0").c_str());
    { std::string vm = argVal(argc, argv, "vmargin", ""); if (!vm.empty()) { c.haveMargin = true; c.vmargin = atof(vm.c_str()); } }
    { std::string af = argVal(argc, argv, "askfloor", ""); if (!af.empty()) c.askFloor = atof(af.c_str()); }
    PathologyStats st = runLitPathology(c);
    printf("litpath seed=%llu liveonly=%d timecost=%g vmargin=%s askfloor=%s\n",
           (unsigned long long)c.seed, c.liveOnly, c.timeCost,
           c.haveMargin ? std::to_string(c.vmargin).c_str() : "default",
           c.askFloor >= 0 ? std::to_string(c.askFloor).c_str() : "default");
    printPathology(st, std::cout);
    return 0;
  }

  if (cmd == "lith2h") {
    LitPathCfg c;
    c.games = atoi(argVal(argc, argv, "games", "400").c_str());
    c.seed = strtoull(argVal(argc, argv, "seed", "777001").c_str(), nullptr, 10);
    c.rules = rulesFrom(argc, argv);
    c.threads = threads;
    c.liveOnly = atoi(argVal(argc, argv, "liveonly", "0").c_str());
    c.timeCost = atof(argVal(argc, argv, "timecost", "0").c_str());
    LitH2H h = runLitH2H(c);
    printf("lith2h games=%lld liveonly=%d timecost=%g  A sets %lld  B sets %lld  A winrate %.3f%% (draws %lld)\n",
           h.games, c.liveOnly, c.timeCost, h.setsA, h.setsB,
           h.games ? 100.0 * (double(h.winA) + 0.5 * double(h.draws)) / double(h.games) : 0.0, h.draws);
    return 0;
  }

  if (cmd == "litdecl") {
    int g = atoi(argVal(argc, argv, "games", "200").c_str());
    int rot = atoi(argVal(argc, argv, "rotations", "2").c_str());
    uint64_t sd = strtoull(argVal(argc, argv, "seed", "777001").c_str(), nullptr, 10);
    Rules r = rulesFrom(argc, argv);
    bool vdOff = argVal(argc, argv, "vdecl", "1") == "0";
    std::string vm = argVal(argc, argv, "vmargin", "");
    litp::LitStats st;
    runLitProbe(g, rot, sd, r, threads, st, vdOff, vm.empty() ? 0.0 : atof(vm.c_str()), !vm.empty());
    printf("litdecl games=%d rot=%d seed=%llu vdecl=%d vmargin=%s\n", g, rot,
           (unsigned long long)sd, vdOff ? 0 : 1, vm.empty() ? "default" : vm.c_str());
    printLitProbe(st, std::cout);
    return 0;
  }

  if (cmd == "passverify") {
    fish::passverify::PVConfig pc;
    pc.games = atoi(argVal(argc, argv, "games", "1500").c_str());
    pc.rotations = atoi(argVal(argc, argv, "rotations", "2").c_str());
    pc.seed = strtoull(argVal(argc, argv, "seed", "31").c_str(), nullptr, 10);
    pc.rules = rulesFrom(argc, argv);
    pc.threads = threads;
    fish::passverify::PVStats st = fish::passverify::runPassVerify(pc);
    printf("passverify games=%d rot=%d seed=%llu\n", pc.games, pc.rotations,
           (unsigned long long)pc.seed);
    fish::passverify::printPassVerify(st, std::cout);
    return 0;
  }

  if (cmd == "vforced") {
    vfe::VFCfg c;
    c.specA = argVal(argc, argv, "a", "v04");
    c.specB = argVal(argc, argv, "b", "v04");
    c.games = atoi(argVal(argc, argv, "games", "300").c_str());
    c.rotations = atoi(argVal(argc, argv, "rotations", "2").c_str());
    c.seed = strtoull(argVal(argc, argv, "seed", "31").c_str(), nullptr, 10);
    c.rules = rulesFrom(argc, argv);
    { std::string fl = argVal(argc, argv, "forcedlow", "");
      if (!fl.empty()) c.rules.forcedTh[6] = atof(fl.c_str()); }
    vfe::VFOut o = vfe::runVerifyForced(c);
    std::cout << c.specA << " vs " << c.specB << "  seed=" << c.seed
              << "  games=" << c.games << "x" << c.rotations << "\n";
    vfe::printVerifyForced(c, o, std::cout);
    return 0;
  }

  // --- appended (adversarial verification of P1 deadlock finding) ----------
  if (cmd == "vdeadlock") {
    DeadlockCfg dc;
    dc.spec = argVal(argc, argv, "spec", "v04");
    dc.games = atoi(argVal(argc, argv, "games", "60").c_str());
    dc.rotations = atoi(argVal(argc, argv, "rotations", "2").c_str());
    dc.seed = strtoull(argVal(argc, argv, "seed", "31").c_str(), nullptr, 10);
    dc.minEvents = atoi(argVal(argc, argv, "minev", "300").c_str());
    dc.dump = atoi(argVal(argc, argv, "dump", "2").c_str());
    dc.stride = atoi(argVal(argc, argv, "stride", "40").c_str());
    dc.maxStates = atoi(argVal(argc, argv, "states", "3").c_str());
    dc.rules = rulesFrom(argc, argv);
    runVDeadlock(dc, std::cout);
    return 0;
  }

  // --- appended (adversarial verification of P8 turn-transfer finding) -----
  if (cmd == "vturnxfer") {
    using namespace fish::probeturnxfer;
    CoordConfig cc;
    cc.specA = argVal(argc, argv, "a", "v04");
    cc.specB = argVal(argc, argv, "b", "v04");
    cc.games = atoi(argVal(argc, argv, "games", "300").c_str());
    cc.rotations = atoi(argVal(argc, argv, "rotations", "2").c_str());
    cc.seed = strtoull(argVal(argc, argv, "seed", "31").c_str(), nullptr, 10);
    cc.rules = rulesFrom(argc, argv);
    cc.threads = threads;
    std::string pol = argVal(argc, argv, "pass", "unilateral");
    cc.cc.policy = pol == "oracle" ? PassPolicy::Oracle
                 : pol == "ladder" ? PassPolicy::Ladder
                 : pol == "low"    ? PassPolicy::LowSeat
                 : pol == "cards"  ? PassPolicy::MostCards
                 : PassPolicy::Unilateral;
    cc.cc.measure = !argFlag(argc, argv, "no-measure");
    std::cout << cc.specA << " (pass=" << pol << ") vs " << cc.specB << " (pass=unilateral)\n";
    std::vector<int> perGame;
    std::string dump = argVal(argc, argv, "dump", "");
    if (!dump.empty()) { perGame.assign(size_t(cc.games) * cc.rotations, 0); cc.perGameA = &perGame; }
    CoordStats st = runCoord(cc);
    if (!dump.empty()) { std::ofstream f(dump); for (size_t i = 0; i < perGame.size(); i++) f << perGame[i] << "\n"; }
    printCoord(st, cc, std::cout);
    return 0;
  }

  if (cmd == "polreview") {
    using namespace fish::polreview;
    RevCfg c;
    c.games = atoi(argVal(argc, argv, "games", "100").c_str());
    c.rotations = atoi(argVal(argc, argv, "rotations", "2").c_str());
    c.seed = strtoull(argVal(argc, argv, "seed", "31").c_str(), nullptr, 10);
    c.rules = rulesFrom(argc, argv);
    c.fixA = atoi(argVal(argc, argv, "fixa", "0").c_str());
    c.fixB = atoi(argVal(argc, argv, "fixb", "0").c_str());
    c.measure = atoi(argVal(argc, argv, "measure", "1").c_str()) != 0;
    c.dump = atoi(argVal(argc, argv, "dump", "0").c_str()) != 0;
    c.dumpThresh = atof(argVal(argc, argv, "dumpthresh", "0.5").c_str());
    std::string bm = argVal(argc, argv, "belief", "fast");
    if (bm == "block") c.belief = BeliefMode::Block;
    RevOut o = runReview(c);
    const RevStats& s = o.st;
    int n2 = o.ms.games * c.rotations;
    double lo, hi; wilson(o.ms.winsA, n2, lo, hi);
    std::cout << "polreview games=" << c.games << " rot=" << c.rotations << " seed=" << c.seed
              << " fixA=" << c.fixA << " fixB=" << c.fixB << " belief=" << bm << "\n";
    std::cout << "  A win " << 100.0 * o.ms.winsA / n2 << "% [" << 100 * lo << ", " << 100 * hi
              << "]  sets " << o.ms.sets[0] << "-" << o.ms.sets[1]
              << "  events/game " << double(o.ms.events) / n2
              << "  A decl " << o.ms.decl[0] << " wrong "
              << (o.ms.decl[0] ? 100.0 * (o.ms.decl[0] - o.ms.declCorrect[0]) / o.ms.decl[0] : 0.0) << "%\n";
    if (c.measure) {
      std::cout << "  declaration opportunities        " << s.opps
                << "  (press0 " << s.oppsPress[0] << " press1 " << s.oppsPress[1]
                << " press2 " << s.oppsPress[2] << ")\n";
      std::cout << "  ... with dirty belief on entry   " << s.oppsStale
                << " (" << (s.opps ? 100.0 * s.oppsStale / s.opps : 0.0) << "%)\n";
      std::cout << "  ... reaching refresh()           " << s.refreshReached << "\n";
      std::cout << "  mean events since last refresh   " << (s.opps ? double(s.ageSum) / s.opps : 0.0)
                << "   max " << s.ageMax << "\n";
      std::cout << "  eH compared (stale vs fresh)     " << s.cmp
                << "  differing " << s.cmpDiff
                << " (" << (s.cmp ? 100.0 * s.cmpDiff / s.cmp : 0.0) << "%)\n";
      std::cout << "  mean max|eH stale-fresh|         " << (s.cmp ? s.sumMaxDiff / s.cmp : 0.0)
                << "   max " << s.maxMaxDiff << "\n";
      std::cout << "  declareNow calls                 " << s.declNowCalls
                << "  (via declareByValue " << s.valueRuleCalls << ")\n";
      std::cout << "  declareNow verdicts flipped      " << s.flips
                << "  (of which value-rule " << s.valueFlips << ")\n";
      std::cout << "  final actions changed            " << s.actionChanged << "\n";
    }
    return 0;
  }

  if (cmd == "vhorizon") {
    uint64_t sd = strtoull(argVal(argc, argv, "seed", "4242").c_str(), nullptr, 10);
    int deals = atoi(argVal(argc, argv, "deals", "200").c_str());
    fish::vpol::runHorizonUnit(sd, deals);
    fish::vpol::runGateAttribution(sd, deals);
    return 0;
  }

  std::cout << "usage: fish <match|verify|matrix|bench|serve> [--a=SPEC] [--b=SPEC] [--games=N] [--seed=S] [--legacy] [--json] [--audit]\n";
  std::cout << "       fish serve [--port=8173] [--web=DIR]   interactive table in a browser\n";
  std::cout << "         --lan             also listen on the local network, for players in the room\n";
  std::cout << "         --public          publish an https address via a tunnel, for players anywhere\n";
  std::cout << "         --tunnel=NAME     which tunnel to use: cloudflared, ssh, or auto (default)\n";
  std::cout << "         --auth            require credentials even on loopback\n";
  std::cout << "         --invite=CODE     fix the invite code instead of generating one\n";
  std::cout << "         --bots=DIR        where uploaded bot packages live (default engine/bots)\n";
  std::cout << "         --lock-bots       only the host may upload a bot package\n";
  std::cout << "       fish botserve --policy=SPEC   speak the bot protocol on stdin/stdout,\n";
  std::cout << "                                     so one of our policies can be packaged\n";
  std::cout << "       fish bots <list|add|check|prepare|remove> [arg]   uploaded bot packages\n";
  std::cout << "         a package installed here is seatable and measurable as bot:<id>\n";
  std::cout << "         format: docs/BOT_PACKAGE.md\n";
  std::cout << "         (--lan/--public mint a host token and per-seat tokens: see docs/PLAY.md)\n";
  return 0;
}
