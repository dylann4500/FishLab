// One-shot decision shim for dylann4500's FishBot v0.7 ("Dylan's FishBot").
//
// Our engine (Python) is the arbiter; this binary answers exactly ONE decision
// for one seat and exits.  Stateless by design so the same binary serves both
// the local match harness and the Vercel function, which replays the public
// log on every request anyway.
//
// stdin, line-oriented:
//   SPEC <their frozen spec string, one line>
//   RULES <deckSets> <outOfTurnDeclare 0/1> <cardlessMayDeclare 0/1>
//   SEAT <0..5>
//   HAND <uint64 bitmask, THEIR card ids>
//   SEED <uint64>
//   EV ASK <actor> <target> <card> <success 0/1> <hc0..hc5>
//   EV DECL <actor> <set> <success 0/1> <o0..o5> <hc0..hc5> <s0> <s1>
//   EV PASS <actor> <target> <hc0..hc5>
//   TURN <seat on move>
//   DECIDE TURN            -> "DECL <set> <o0..o5>" or "ASK <target> <card>"
//   DECIDE PASSTO <k> <c0> ... <ck-1>   -> "PASS <teammate>"
//   DECIDE FORCED <set>    -> "DECL <set> <o0..o5>"   (bestGuess path)
//
// The declaration poll runs before the ask poll on DECIDE TURN, mirroring
// their own game loop's order for the turn holder.  Their out-of-turn
// declaration channel is never polled here because our rules do not have it;
// RULES therefore arrives with outOfTurnDeclare=0 so the policy's own view of
// the rules matches the game it is actually playing.
#include "fish.hpp"
#include "game.hpp"
#include "factory.hpp"
#include <iostream>
#include <sstream>
#include <string>

using namespace fish;

int main() {
    std::ios::sync_with_stdio(false);
    std::string line, spec;
    Rules rules;
    int seat = 0;
    uint64_t hand = 0, seed = 1;
    PublicState pub;
    std::unique_ptr<Agent> agent;
    // reset() is deferred until the first EV/TURN/DECIDE so SPEC/RULES/HAND
    // order does not matter within the header block.
    bool booted = false;
    auto boot = [&]() {
        if (booted) return;
        agent = makeAgent(spec);
        if (!agent) { std::cerr << "bad spec\n"; std::exit(2); }
        agent->reset(seat, hand, rules, seed);
        pub.rules = rules;
        for (int s = 0; s < NSET; s++) pub.setActive[s] = (s < rules.deckSets);
        pub.score[0] = pub.score[1] = 0;
        // Hand counts start at the deal (54/6 = 9 each); with no events yet the
        // policy would otherwise see a cardless table and emit a junk move.
        for (int p = 0; p < NPLAY; p++) pub.handCount[p] = uint8_t(rules.deckSets);
        booted = true;
    };
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string tag; in >> tag;
        if (tag == "SPEC") { std::getline(in, spec); while (!spec.empty() && spec[0]==' ') spec.erase(0,1); }
        else if (tag == "RULES") { int oot, cl; in >> rules.deckSets >> oot >> cl;
            rules.outOfTurnDeclare = oot; rules.cardlessMayDeclare = cl; }
        else if (tag == "SEAT") in >> seat;
        else if (tag == "HAND") in >> hand;
        else if (tag == "SEED") in >> seed;
        else if (tag == "EV") {
            boot();
            std::string kind; in >> kind;
            Event e{};
            if (kind == "ASK") {
                int a,t,c,s; in >> a >> t >> c >> s;
                e.kind = Kind::Ask; e.actor=uint8_t(a); e.target=uint8_t(t);
                e.card=uint8_t(c); e.set=uint8_t(setOf(c)); e.success=s;
                for (int p = 0; p < NPLAY; p++) { int h; in >> h; e.handCount[p]=uint8_t(h); }
                pub.nAsks++;
            } else if (kind == "DECL") {
                int a,st,s; in >> a >> st >> s;
                e.kind = Kind::Declare; e.actor=uint8_t(a); e.set=uint8_t(st); e.success=s;
                e.decl.set = uint8_t(st);
                for (int i = 0; i < SETSZ; i++) { int o; in >> o; e.decl.owner[i]=uint8_t(o); }
                for (int p = 0; p < NPLAY; p++) { int h; in >> h; e.handCount[p]=uint8_t(h); }
                int s0, s1; in >> s0 >> s1;
                pub.setActive[st] = false; pub.score[0]=uint8_t(s0); pub.score[1]=uint8_t(s1);
            } else if (kind == "PASS") {
                int a,t; in >> a >> t;
                e.kind = Kind::Pass; e.actor=uint8_t(a); e.target=uint8_t(t);
                for (int p = 0; p < NPLAY; p++) { int h; in >> h; e.handCount[p]=uint8_t(h); }
            } else { std::cerr << "bad ev\n"; return 2; }
            for (int p = 0; p < NPLAY; p++) pub.handCount[p] = e.handCount[p];
            pub.nEvents++;
            pub.history.push_back(e);
            agent->observe(e);
        }
        else if (tag == "TURN") { boot(); in >> pub.turn; }
        else if (tag == "DECIDE") {
            boot();
            std::string what; in >> what;
            if (what == "TURN") {
                Declaration d; double conf = 0;
                if (agent->proposeDeclaration(pub, d, conf) && pub.setActive[d.set]) {
                    std::cout << "DECL " << int(d.set);
                    for (int i = 0; i < SETSZ; i++) std::cout << ' ' << int(d.owner[i]);
                    std::cout << "\n"; return 0;
                }
                AskMove m = agent->chooseAsk(pub);
                if (isRepoll(m)) { std::cerr << "repoll from engine policy\n"; return 3; }
                std::cout << "ASK " << int(m.target) << ' ' << int(m.card) << "\n";
                return 0;
            }
            if (what == "PASSTO") {
                int n; in >> n; int cand[NPLAY];
                for (int i = 0; i < n && i < NPLAY; i++) in >> cand[i];
                int t = agent->choosePassTarget(pub, cand, n);
                std::cout << "PASS " << t << "\n"; return 0;
            }
            if (what == "FORCEDLADDER") {
                // Variant bridge (E3): the mover runs FishLab's forced-endgame ladder over every
                // live half-suit it may declare, as game.hpp::forcedEndgame does for one seat:
                // voluntary proposal first, then willingness at descending thresholds, then a
                // best guess on the most confident set.
                int n; in >> n; std::vector<int> sets;
                for (int i = 0; i < n; i++) { int s; in >> s; sets.push_back(s); }
                auto emit = [&](const Declaration& d) {
                    std::cout << "DECL " << int(d.set);
                    for (int i = 0; i < SETSZ; i++) std::cout << ' ' << int(d.owner[i]);
                    std::cout << "\n";
                };
                auto onTeam = [&](const Declaration& d) {
                    for (int i = 0; i < SETSZ; i++) if (teamOf(d.owner[i]) != teamOf(seat)) return false;
                    return true;
                };
                { Declaration p2; double c2 = 0;
                  if (agent->proposeDeclaration(pub, p2, c2) && pub.setActive[p2.set] && onTeam(p2)) {
                      bool live = false; for (int s : sets) if (s == int(p2.set)) live = true;
                      if (live) { emit(p2); return 0; }
                  } }
                for (int ti = 0; ti < pub.rules.nForcedTh; ti++) {
                    double th = pub.rules.forcedTh[ti];
                    if (th < 0) break;
                    for (int s : sets) {
                        Declaration d{}; double conf = 0;
                        if (!agent->willingForced(pub, s, d, conf, th)) continue;
                        d.set = uint8_t(s);
                        if (!onTeam(d)) continue;
                        emit(d); return 0;
                    }
                }
                Declaration best{}; double bestConf = -1; bool have = false;
                for (int s : sets) {
                    Declaration d{}; double conf = 0;
                    agent->bestGuess(pub, s, d, conf); d.set = uint8_t(s);
                    if (!onTeam(d)) continue;
                    if (!have || conf > bestConf) { best = d; bestConf = conf; have = true; }
                }
                if (!have) { best.set = uint8_t(sets[0]); for (int i = 0; i < SETSZ; i++) best.owner[i] = uint8_t(seat); }
                emit(best); return 0;
            }
            if (what == "FORCED") {
                int st; in >> st;
                Declaration d; double conf = 0;
                d.set = uint8_t(st);
                // proposeDeclaration may target a different set; honour a
                // matching proposal, else take the policy's best guess for the
                // set our engine requires.
                Declaration p2; double c2 = 0;
                if (agent->proposeDeclaration(pub, p2, c2) && p2.set == st) d = p2;
                else agent->bestGuess(pub, st, d, conf);
                std::cout << "DECL " << int(d.set);
                for (int i = 0; i < SETSZ; i++) std::cout << ' ' << int(d.owner[i]);
                std::cout << "\n"; return 0;
            }
            std::cerr << "bad decide\n"; return 2;
        }
    }
    std::cerr << "no DECIDE\n"; return 2;
}
