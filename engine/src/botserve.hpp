// `fish botserve` -- this engine's own policies, published as a bot package.
//
// extbot.hpp drives a foreign bot over the FishLab protocol.  This is the mirror
// image: it SPEAKS that protocol, so any policy this engine can build --
// SESTINA v1.0, v0.6, a baseline, an ablation -- can be zipped up and handed to
// somebody else's table, or seated on this one through the ordinary upload path
// like anybody else's entry.
//
//   fish botserve --policy=v07:...        # one decision per line, on stdin
//
// The protocol is stateless and an Agent is not, which is the whole of the work
// here.  Every request carries the entire public record; the agent wants
// reset() once and observe() per event, in order.  So this keeps the agent
// alive across requests and applies only the events it has not seen yet, and
// rebuilds from the deal whenever the record it is handed disagrees with the one
// it has been applying -- which is the only way a replay can silently diverge.
//
// TWO THINGS THIS IS NOT:
//
//  * It is not a reimplementation.  makeAgent() builds the same object the table
//    seats, from the same spec string, so what plays here is the policy and not
//    a description of it.
//  * It is not bit-identical to a native seat, and cannot be.  A seat is seeded
//    from the deal (game.hpp: mixSeed(seed, p + 77)); the protocol deliberately
//    does not carry the deal seed, because a bot that could read it could invert
//    it and recover the hands.  So a searching policy explores different
//    determinizations here than it would in-process.  Same policy, same
//    strength, different rollouts -- and the seed used is fixed, so a run of
//    this package is reproducible on its own terms.
#pragma once
#include "factory.hpp"
#include "minijson.hpp"
#include "extbot.hpp"     // cardCode / cardFromCode: one spelling of the deck
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace fish {
namespace botserve {

// A stable, deal-independent seed.  Fixed rather than random so that two runs of
// the same package over the same game make the same moves.
inline uint64_t seatSeed(int seat) { return mixSeed(0x5E571A0000000001ull, uint64_t(seat)); }

struct Server {
  std::string spec;
  std::string label = "SESTINA v1.0";
  std::string version = "1.0";
  std::unique_ptr<Agent> agent;
  int seat = -1;
  uint64_t initialHand = 0;
  Rules rules;
  bool haveGame = false;
  // The events already handed to the agent, kept verbatim so that a record that
  // changes underneath us is noticed rather than replayed on top of.
  std::vector<std::string> applied;

  static std::string jesc(const std::string& s) { return mj::escape(s); }

  // ------------------------------------------------------------- rebuilding
  static bool eventFrom(const mj::Value& e, Event& out, std::string& err) {
    std::string t = e.s("t");
    out = Event{};
    const mj::Value* counts = e.find("counts");
    if (counts && counts->isArr() && counts->arr.size() == NPLAY)
      for (int p = 0; p < NPLAY; p++) out.handCount[p] = uint8_t(counts->arr[size_t(p)].num);
    if (t == "ask") {
      out.kind = Kind::Ask;
      out.actor = uint8_t(e.n("actor", 0));
      out.target = uint8_t(e.n("target", 0));
      int c = extbot::cardFromCode(e.s("card"));
      if (c < 0) { err = "an ask event names a card that is not in the deck"; return false; }
      out.card = uint8_t(c);
      out.set = uint8_t(setOf(c));
      out.success = e.bo("success", false);
      return true;
    }
    if (t == "declare") {
      out.kind = e.bo("forced", false) ? Kind::ForcedDeclare : Kind::Declare;
      out.actor = uint8_t(e.n("actor", 0));
      out.set = uint8_t(e.n("set", 0));
      out.success = e.bo("success", false);
      out.decl.set = out.set;
      const mj::Value* own = e.find("owner");
      if (!own || !own->isArr() || own->arr.size() != SETSZ) {
        err = "a declaration event has no six-seat owner array";
        return false;
      }
      for (int i = 0; i < SETSZ; i++) out.decl.owner[i] = uint8_t(own->arr[size_t(i)].num);
      return true;
    }
    if (t == "pass") {
      out.kind = Kind::Pass;
      out.actor = uint8_t(e.n("actor", 0));
      out.target = uint8_t(e.n("target", 0));
      return true;
    }
    err = "unknown event type '" + t + "'";
    return false;
  }

  // Canonical text for one event, used only to tell "the same record, extended"
  // from "a different record".  Cheap and exact; the alternative is trusting a
  // length, which is precisely how a replay goes quietly wrong.
  static std::string sig(const mj::Value& e) {
    std::string s = e.s("t");
    s += ":" + std::to_string(int(e.n("actor", -1)));
    s += ":" + std::to_string(int(e.n("target", -1)));
    s += ":" + e.s("card");
    s += ":" + std::to_string(int(e.n("set", -1)));
    s += ":" + std::string(e.bo("success", false) ? "1" : "0");
    const mj::Value* own = e.find("owner");
    if (own && own->isArr())
      for (const auto& v : own->arr) s += "," + std::to_string(int(v.num));
    return s;
  }

  bool rebuild(std::string& err) {
    if (seat < 0) { err = "no new_game was sent, so there is no deal to rebuild from"; return false; }
    agent = makeAgent(spec);
    agent->reset(seat, initialHand, rules, seatSeed(seat));
    applied.clear();
    (void)err;
    return true;
  }

  // Brings the agent up to date with the record in this request.
  bool sync(const mj::Value& state, std::string& err) {
    const mj::Value* hist = state.find("history");
    if (!hist || !hist->isArr()) { err = "the state has no history array"; return false; }
    // A record shorter than what we have applied, or one whose prefix differs,
    // is a different game (or a host that replays); start again from the deal.
    bool diverged = hist->arr.size() < applied.size();
    if (!diverged)
      for (size_t i = 0; i < applied.size(); i++)
        if (sig(hist->arr[i]) != applied[i]) { diverged = true; break; }
    if (diverged && !rebuild(err)) return false;
    if (!agent) { err = "no agent: new_game has not been sent"; return false; }
    for (size_t i = applied.size(); i < hist->arr.size(); i++) {
      Event e{};
      if (!eventFrom(hist->arr[i], e, err)) return false;
      agent->observe(e);
      applied.push_back(sig(hist->arr[i]));
    }
    return true;
  }

  // The engine's PublicState, as the policies expect to read it.
  bool publicState(const mj::Value& state, PublicState& pub, std::string& err) {
    pub = PublicState{};
    pub.rules = rules;
    const mj::Value* hc = state.find("hand_counts");
    if (!hc || !hc->isArr() || hc->arr.size() != NPLAY) { err = "hand_counts must be six numbers"; return false; }
    for (int p = 0; p < NPLAY; p++) pub.handCount[p] = uint8_t(hc->arr[size_t(p)].num);
    const mj::Value* sa = state.find("set_active");
    if (!sa || !sa->isArr() || sa->arr.size() != NSET) { err = "set_active must be nine booleans"; return false; }
    for (int s = 0; s < NSET; s++) pub.setActive[s] = sa->arr[size_t(s)].b;
    const mj::Value* sc = state.find("score");
    if (sc && sc->isArr() && sc->arr.size() == 2)
      for (int t = 0; t < 2; t++) pub.score[t] = uint8_t(sc->arr[size_t(t)].num);
    pub.turn = int(state.n("turn", 0));
    const mj::Value* hist = state.find("history");
    if (hist && hist->isArr()) {
      for (const auto& ev : hist->arr) {
        Event e{};
        std::string ignored;
        if (!eventFrom(ev, e, ignored)) continue;
        pub.history.push_back(e);
        if (e.kind == Kind::Ask) pub.nAsks++;
      }
    }
    // nEvents is what the v0.7 stall and forcing horizons are measured against,
    // so it has to be the count the driver would have, not zero.
    pub.nEvents = int(pub.history.size());
    return true;
  }

  void readRules(const mj::Value& r) {
    rules = Rules{};
    if (!r.isObj()) return;
    rules.outOfTurnDeclare = r.bo("out_of_turn_declare", rules.outOfTurnDeclare);
    rules.cardlessMayDeclare = r.bo("cardless_may_declare", rules.cardlessMayDeclare);
    rules.maxAsks = int(r.n("max_asks", rules.maxAsks));
    rules.deckSets = int(r.n("deck_sets", rules.deckSets));
  }

  // ---------------------------------------------------------------- replies
  static std::string declJson(const Declaration& d, double conf) {
    std::string o = "{\"action\":\"declare\",\"set\":" + std::to_string(int(d.set)) + ",\"owner\":[";
    for (int i = 0; i < SETSZ; i++) o += (i ? "," : "") + std::to_string(int(d.owner[i]));
    char c[32];
    snprintf(c, sizeof(c), "%.4f", conf < 0 ? 0.0 : (conf > 1 ? 1.0 : conf));
    return o + "],\"confidence\":" + c + "}";
  }

  std::string handle(const std::string& line) {
    mj::Value req;
    std::string err;
    if (!mj::parse(line, req, err)) return "{\"error\":" + jesc("bad JSON: " + err) + "}";
    if (!req.isObj()) return "{\"error\":\"a request must be an object\"}";
    const std::string op = req.s("op");

    if (op == "hello") {
      std::string got = req.s("protocol", extbot::PROTOCOL);
      if (got != extbot::PROTOCOL)
        return "{\"error\":" + jesc("this package speaks " + std::string(extbot::PROTOCOL) +
                                    ", the host asked for " + got) + "}";
      return "{\"ok\":true,\"name\":" + jesc(label) + ",\"version\":" + jesc(version) +
             ",\"protocol\":\"" + extbot::PROTOCOL + "\",\"spec\":" + jesc(spec) + "}";
    }

    if (op == "new_game") {
      seat = int(req.n("seat", -1));
      if (seat < 0 || seat >= NPLAY) return "{\"error\":\"new_game without a seat\"}";
      const mj::Value* r = req.find("rules");
      if (r) readRules(*r);
      const mj::Value* h = req.find("hand");
      initialHand = 0;
      if (h && h->isArr())
        for (const auto& c : h->arr) {
          int i = extbot::cardFromCode(c.str);
          if (i < 0) return "{\"error\":" + jesc("new_game names a card not in the deck: " + c.str) + "}";
          initialHand |= bit(i);
        }
      if (!rebuild(err)) return "{\"error\":" + jesc(err) + "}";
      haveGame = true;
      return "{\"ok\":true}";
    }

    const mj::Value* st = req.find("state");
    if (!st || !st->isObj()) return "{\"error\":" + jesc("op '" + op + "' needs a state object") + "}";
    if (!haveGame) return "{\"error\":\"no new_game has been sent for this deal\"}";
    if (!sync(*st, err)) return "{\"error\":" + jesc(err) + "}";
    PublicState pub;
    if (!publicState(*st, pub, err)) return "{\"error\":" + jesc(err) + "}";

    if (op == "declare_poll") {
      Declaration d{};
      double conf = 0;
      if (!agent->proposeDeclaration(pub, d, conf)) return "{\"action\":\"none\"}";
      return declJson(d, conf);
    }
    if (op == "ask") {
      AskMove m = agent->chooseAsk(pub);
      if (m.card >= NCARD || m.target >= NPLAY)
        return "{\"error\":\"the policy produced no legal ask from this position\"}";
      return "{\"action\":\"ask\",\"card\":\"" + extbot::cardCode(m.card) + "\",\"target\":" +
             std::to_string(int(m.target)) + "}";
    }
    if (op == "pass") {
      const mj::Value* cand = req.find("candidates");
      if (!cand || !cand->isArr() || cand->arr.empty())
        return "{\"error\":\"a pass request with no candidates\"}";
      int c[NPLAY];
      int n = 0;
      for (const auto& v : cand->arr) if (n < NPLAY) c[n++] = int(v.num);
      int to = agent->choosePassTarget(pub, c, n);
      bool ok = false;
      for (int i = 0; i < n; i++) if (c[i] == to) ok = true;
      if (!ok) to = c[n - 1];
      return "{\"action\":\"pass\",\"to\":" + std::to_string(to) + "}";
    }
    if (op == "forced") {
      int set = int(req.n("set", -1));
      if (set < 0 || set >= NSET) return "{\"error\":\"a forced request without a half-suit\"}";
      double th = req.n("threshold", 0.0);
      bool last = req.bo("last_resort", false);
      Declaration d{};
      double conf = 0;
      if (last) agent->bestGuess(pub, set, d, conf);
      else if (!agent->willingForced(pub, set, d, conf, th)) return "{\"action\":\"none\"}";
      // The host asked about THIS half-suit and will not reshape an answer about
      // another; the engine's own driver overwrites d.set for the same reason.
      d.set = uint8_t(set);
      return declJson(d, conf);
    }
    return "{\"error\":" + jesc("unknown op '" + op + "'") + "}";
  }
};

inline int run(const std::string& spec, const std::string& name, const std::string& version) {
  Server srv;
  srv.spec = spec;
  srv.label = name;
  srv.version = version;
  std::string line;
  // Unbuffered replies: a line sitting in a buffer is indistinguishable from a
  // bot that has hung, which is the first thing docs/BOT_PACKAGE.md warns about.
  setvbuf(stdout, nullptr, _IONBF, 0);
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    std::string reply = srv.handle(line);
    fputs(reply.c_str(), stdout);
    fputc('\n', stdout);
    fflush(stdout);
  }
  return 0;
}

}  // namespace botserve
}  // namespace fish
