// Check the browser model (docs/js/model.js) against Python-computed cases.
//   node scripts/check_js.mjs [path/to/data/dir]
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dataDir = process.argv[2] || path.join(root, 'docs', 'data');
const { Engine } = await import(pathToFileURL(path.join(root, 'docs', 'js', 'model.js')));
const load = (f) => JSON.parse(readFileSync(path.join(dataDir, f), 'utf8'));
const meta = load('meta.json'), players = load('players.json'), fx = load('fixtures.json');

const engine = new Engine(meta, players);
let fails = 0, checks = 0;
const close = (a, b, tol, label) => {
  checks++;
  if (Math.abs(a - b) > tol) { fails++; console.log(`FAIL ${label}: js=${a} py=${b}`); }
};

for (const set of fx.pairs) {
  engine.sizeStrength = set.size;
  set.pairs.forEach(([s, d], i) => close(engine.predictPair(s, d), set.pred[i], 1e-3, `pair ${s}/${d} size ${set.size}`));
}
for (const r of fx.rank) {
  engine.sizeStrength = r.size;
  const top = engine.rankLineups(fx.mine, fx.theirs, r.slider, r.top.length);
  r.top.forEach((t, i) => {
    checks++;
    const same = JSON.stringify([...top[i].lineup].sort((a, b) => a - b)) === JSON.stringify(t.lineup);
    if (!same) { fails++; console.log(`FAIL rank #${i + 1} lineup (slider ${r.slider}, size ${r.size})`); }
    close(top[i].net, t.net, 1e-3, `rank #${i + 1} net`);
    close(top[i].scored, t.scored, 1e-3, `rank #${i + 1} scored`);
    close(top[i].allowed, t.allowed, 1e-3, `rank #${i + 1} allowed`);
  });
}
engine.sizeStrength = 1;
const e = fx.evaluate;
const assign = Object.fromEntries(Object.entries(e.assignment).map(([k, v]) => [Number(k), v]));
const ev = engine.evaluate(e.mine, fx.theirs, assign, 1);
close(ev.net, e.net, 1e-3, 'evaluate net');
close(ev.bestNet, e.best_net, 1e-3, 'evaluate best net');
close(ev.cost, e.cost, 1e-3, 'evaluate cost');

console.log(`${checks - fails}/${checks} checks passed`);
process.exit(fails ? 1 : 0);
