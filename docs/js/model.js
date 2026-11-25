// Browser version of the Python engine (src/matchups/optimize.py + model.py).
// Every number comes from docs/data/*.json, written by src/export_results.py.
// Checked against Python-computed cases in fixtures.json (scripts/check_js.mjs).
//
//   matchup:  pts/100 matchup poss = 100 * exp(intercept + scorer + defender
//                                         + size_strength * (w . size/style terms))
//   lineup:   scored(A vs B) = base + a*sum(O-RAPM of A)
//                                   + b*(best matchup defense of B on A - dm_mean)
//                                   + c*sum(D-RAPM of B)
//             net = (scored - base) - slider * (allowed - base)
//
// With 5 defenders and 5 scorers there are only 120 possible assignments, so we
// check all of them: the same optimum the Hungarian algorithm finds in Python.

export const PERMS = (() => {
  const out = [];
  const go = (rest, acc) => {
    if (!rest.length) { out.push(acc); return; }
    rest.forEach((x, i) => go(rest.slice(0, i).concat(rest.slice(i + 1)), acc.concat(x)));
  };
  go([0, 1, 2, 3, 4], []);
  return out;
})();

export function combinations(n, k) {
  const out = [];
  const go = (start, acc) => {
    if (acc.length === k) { out.push(acc); return; }
    for (let i = start; i <= n - (k - acc.length); i++) go(i + 1, acc.concat(i));
  };
  go(0, []);
  return out;
}

export class Engine {
  constructor(meta, players) {
    this.meta = meta;
    this.m = meta.matchup;
    this.cal = meta.calibration;
    this.sizeStrength = 1.0;
    this.p = new Map(players.map((x) => [x.id, x]));
  }

  _player(id) { return this.p.get(id) || {}; }

  // Predicted pts per 100 matchup possessions when `defender` guards `scorer`.
  predictPair(scorer, defender) {
    const m = this.m, s = this._player(scorer), d = this._player(defender);
    const hs = s.h ?? m.height_mean, hd = d.h ?? m.height_mean, ts = s.t ?? m.three_rate_mean;
    const feats = [
      (hd - hs) ** 2 / 10 - m.centers[0],
      (ts - m.three_rate_mean) * (hd - m.height_mean) - m.centers[1],
    ];
    let eta = m.intercept + (s.sc ?? 0) + (d.df ?? 0);
    for (let k = 0; k < feats.length; k++) eta += this.sizeStrength * m.weights[k] * feats[k];
    return m.kind === 'poisson' ? 100 * Math.exp(eta) : eta;
  }

  // M[i][j] = predicted pts/100 if defenders[i] guards scorers[j]
  matrix(defenders, scorers) {
    return defenders.map((d) => scorers.map((s) => this.predictPair(s, d)));
  }

  // All 120 assignments of 5 defenders (rows) to 5 scorers (cols), best first.
  assignments(M) {
    return PERMS.map((perm) => ({ perm, total: perm.reduce((t, j, i) => t + M[i][j], 0) }))
      .sort((x, y) => x.total - y.total);
  }

  bestAssignment(M) {
    let best = null;
    for (const perm of PERMS) {
      const total = perm.reduce((t, j, i) => t + M[i][j], 0);
      if (!best || total < best.total) best = { perm, total };
    }
    return best;
  }

  sum(ids, key) { return ids.reduce((t, id) => t + (this._player(id)[key] ?? 0), 0); }

  // Calibrated pts/100 for lineup `offense` against lineup `defense`.
  scored(offense, defense) {
    const c = this.cal;
    const dm = this.bestAssignment(this.matrix(defense, offense)).total;
    return c.base + c.a * this.sum(offense, 'o') + c.b * (dm - c.dm_mean) + c.c * this.sum(defense, 'd');
  }

  // Top-k assignments of my five onto their five, as [{pairs:[{def,scorer,pts}], total}]
  topAssignments(mine, theirs, k = 2) {
    const M = this.matrix(mine, theirs);
    return this.assignments(M).slice(0, k).map(({ perm, total }) => ({
      total, pairs: perm.map((j, i) => ({ def: mine[i], scorer: theirs[j], pts: M[i][j] })),
    }));
  }

  lineupValue(mine, theirs, slider = 1) {
    const c = this.cal;
    const scored = this.scored(mine, theirs), allowed = this.scored(theirs, mine);
    return { lineup: mine, scored, allowed, net: (scored - c.base) - slider * (allowed - c.base) };
  }

  // Every 5-man combination of `roster` against `theirs`, best `top` by net.
  rankLineups(roster, theirs, slider = 1, top = 10) {
    const c = this.cal;
    const mDef = this.matrix(roster, theirs);           // our i guards their j
    const mOff = this.matrix(theirs, roster);           // their i guards our j
    const oUs = roster.map((id) => this._player(id).o ?? 0);
    const dUs = roster.map((id) => this._player(id).d ?? 0);
    const oThem = this.sum(theirs, 'o'), dThem = this.sum(theirs, 'd');
    const out = [];
    for (const idx of combinations(roster.length, 5)) {
      const subDef = idx.map((i) => mDef[i]);
      const subOff = mOff.map((row) => idx.map((i) => row[i]));
      const dmAllowed = this.bestAssignment(subDef).total;
      const dmScored = this.bestAssignment(subOff).total;
      const sumO = idx.reduce((t, i) => t + oUs[i], 0), sumD = idx.reduce((t, i) => t + dUs[i], 0);
      const scored = c.base + c.a * sumO + c.b * (dmScored - c.dm_mean) + c.c * dThem;
      const allowed = c.base + c.a * oThem + c.b * (dmAllowed - c.dm_mean) + c.c * sumD;
      out.push({ lineup: idx.map((i) => roster[i]), scored, allowed,
                 net: (scored - c.base) - slider * (allowed - c.base) });
    }
    return out.sort((x, y) => y.net - x.net).slice(0, top);
  }

  // A user-chosen setup. assignment: Map/obj myId -> theirId. Points allowed use
  // that assignment; points scored assume they defend my five optimally.
  evaluate(mine, theirs, assignment, slider = 1) {
    const c = this.cal;
    const get = (id) => (assignment instanceof Map ? assignment.get(id) : assignment[id]);
    const pairs = mine.map((d) => ({ def: d, scorer: get(d), pts: this.predictPair(get(d), d) }));
    const dmAllowed = pairs.reduce((t, x) => t + x.pts, 0);
    const allowed = c.base + c.a * this.sum(theirs, 'o') + c.b * (dmAllowed - c.dm_mean) + c.c * this.sum(mine, 'd');
    const scored = this.scored(mine, theirs);
    const net = (scored - c.base) - slider * (allowed - c.base);
    const best = this.lineupValue(mine, theirs, slider);
    return { pairs, scored, allowed, net, bestNet: best.net, cost: best.net - net,
             best: this.topAssignments(mine, theirs, 1)[0] };
  }
}
