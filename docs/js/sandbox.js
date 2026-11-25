import { loadData, renderChrome, applyTeamColors, teamSelect, getParams, setParams, playerSearch,
  mostUsedLineup, teamPool, fmt, esc, debounce, showError, label } from './ui.js';

const $ = (id) => document.getElementById(id);
let D, S;

// Offensive spots on the half court (basket at the top), in the order of their five.
const SPOTS = [[250, 330], [95, 250], [405, 250], [60, 110], [440, 110]];
const BASKET = [250, 52];

function init(data) {
  D = data;
  const q = getParams();
  const abbrs = Object.keys(D.teams).sort();
  const team = D.teams[q.team] ? q.team : (D.teams.ATL ? 'ATL' : abbrs[0]);
  const opp = D.teams[q.opp] && q.opp !== team ? q.opp : (team !== 'NYK' && D.teams.NYK ? 'NYK' : abbrs.find((a) => a !== team));
  S = {
    team, opp, slider: Number(q.slider) || 1, size: q.size != null ? Number(q.size) : 1,
    mine: q.mine ? q.mine.split(',').map(Number) : mostUsedLineup(D.teams, team),
    theirs: q.their ? q.their.split(',').map(Number) : mostUsedLineup(D.teams, opp),
    assign: null,
  };
  if (q.assign) {
    const ids = q.assign.split(',').map(Number);
    if (ids.length === 5) S.assign = new Map(S.mine.map((m, i) => [m, ids[i]]));
  }
  renderChrome('sandbox.html', D.meta);
  $('season').textContent = D.meta.season;
  teamSelect($('team'), D.teams, S.team, (v) => { S.team = v; S.mine = mostUsedLineup(D.teams, v); S.assign = null; refresh(); });
  teamSelect($('opp'), D.teams, S.opp, (v) => { S.opp = v; S.theirs = mostUsedLineup(D.teams, v); S.assign = null; refresh(); });
  $('slider').value = S.slider; $('size').value = S.size;
  $('slider').oninput = () => { S.slider = Number($('slider').value); compute(); };
  $('size').oninput = () => { S.size = Number($('size').value); compute(); };
  $('reset-mine').onclick = () => { S.mine = mostUsedLineup(D.teams, S.team); S.assign = null; refresh(); };
  $('reset-their').onclick = () => { S.theirs = mostUsedLineup(D.teams, S.opp); S.assign = null; refresh(); };
  $('use-best').onclick = () => { S.assign = null; compute(); };
  refresh();
}

const name = (id) => D.byId.get(id)?.name || String(id);
const last = (id) => fmt.last(name(id));

function refresh() {
  applyTeamColors(S.team, S.opp);
  $('team').value = S.team; $('opp').value = S.opp;
  slots('my-slots', S.mine, teamPool(D.teams, S.team, S.mine), (i, id) => {
    if (S.theirs.includes(id)) return false;
    const old = S.mine[i];
    S.mine[i] = id;
    if (S.assign) { S.assign.set(id, S.assign.get(old)); S.assign.delete(old); }
    return true;
  });
  slots('their-slots', S.theirs, teamPool(D.teams, S.opp, S.theirs), (i, id) => {
    if (S.mine.includes(id)) return false;
    const old = S.theirs[i];
    S.theirs[i] = id;
    if (S.assign) for (const [k, v] of S.assign) if (v === old) S.assign.set(k, id);
    return true;
  });
  compute();
}

function slots(elId, arr, pool, onSet) {
  $(elId).innerHTML = arr.map((id, i) =>
    `<div class="slot"><span class="pos">${i + 1}</span><input type="search" data-i="${i}" value="${esc(D.byId.get(id) ? label(D.byId.get(id)) : '')}"></div>`).join('');
  $(elId).querySelectorAll('input').forEach((inp) => playerSearch(inp, D, (id) => {
    const i = Number(inp.dataset.i);
    if (arr.includes(id) && arr[i] !== id) { inp.classList.add('bad'); return; }
    if (!onSet(i, id)) { inp.classList.add('bad'); return; }
    compute();
  }, pool));
}

const compute = debounce(() => {
  $('slider-out').textContent = `${S.slider.toFixed(1)}×`;
  $('size-out').textContent = `${Math.round(S.size * 100)}%`;
  D.engine.sizeStrength = S.size;
  const best = D.engine.topAssignments(S.mine, S.theirs, 1)[0];
  const valid = S.assign && S.mine.every((m) => S.theirs.includes(S.assign.get(m)))
    && new Set(S.mine.map((m) => S.assign.get(m))).size === 5;
  if (!valid) S.assign = new Map(best.pairs.map((p) => [p.def, p.scorer]));
  const ev = D.engine.evaluate(S.mine, S.theirs, S.assign, S.slider);
  setParams({ team: S.team, opp: S.opp, mine: S.mine.join(','), their: S.theirs.join(','),
              assign: S.mine.map((m) => S.assign.get(m)).join(','),
              slider: S.slider === 1 ? '' : S.slider, size: S.size === 1 ? '' : S.size });

  $('net').textContent = fmt.signed(ev.net);
  $('bug-net').textContent = fmt.signed(ev.net);
  $('scored').textContent = fmt.num(ev.scored);
  $('allowed').textContent = fmt.num(ev.allowed);
  const cost = Math.max(0, ev.cost);
  $('verdict').innerHTML = cost < 0.05
    ? 'This is the best assignment for these five.'
    : `You're giving up <span class="num neg-num">${cost.toFixed(2)}</span> pts per 100 vs. the best assignment <span class="num muted">(${fmt.signed(ev.bestNet)})</span>.`;

  const bestFor = new Map(ev.best.pairs.map((p) => [p.def, p]));
  $('assign').innerHTML = ev.pairs.map((p) => {
    const b = bestFor.get(p.def);
    const differs = b.scorer !== p.scorer;
    return `<div class="assign-row">
      <span class="me">${esc(last(p.def))}</span><span class="arrow">→</span>
      <select class="plain" data-def="${p.def}" aria-label="Who ${esc(name(p.def))} guards">
        ${S.theirs.map((t) => `<option value="${t}" ${t === p.scorer ? 'selected' : ''}>${esc(name(t))}</option>`).join('')}
      </select>
      <span class="num ${differs ? 'neg-num' : 'muted'}" title="${differs ? `Best: on ${esc(name(b.scorer))}` : 'Matches the best assignment'}">${p.pts.toFixed(1)}</span>
    </div>`;
  }).join('') + '<p class="note">Numbers are the scorer\'s predicted points per 100 matchup possessions. Red = differs from the best assignment. Points allowed use your assignment; points scored assume they guard your five as well as possible.</p>';
  $('assign').querySelectorAll('select').forEach((sel) => sel.onchange = () => {
    const def = Number(sel.dataset.def), to = Number(sel.value);
    const from = S.assign.get(def);
    for (const [k, v] of S.assign) if (v === to && k !== def) S.assign.set(k, from);   // swap
    S.assign.set(def, to);
    compute();
  });
  drawCourt(ev, bestFor);
}, 40);

function drawCourt(ev, bestFor) {
  const scorerSpot = new Map(S.theirs.map((id, i) => [id, SPOTS[i]]));
  const toward = ([x, y], k) => [x + (BASKET[0] - x) * k, y + (BASKET[1] - y) * k];
  const court = `
    <rect class="floor" x="0" y="0" width="500" height="390"/>
    <rect class="line" x="10" y="10" width="480" height="370"/>
    <rect class="line" x="180" y="10" width="140" height="180"/>
    <circle class="line" cx="250" cy="190" r="60"/>
    <path class="line" d="M 40 10 L 40 120 A 222 222 0 0 0 460 120 L 460 10"/>
    <circle class="line" cx="${BASKET[0]}" cy="${BASKET[1]}" r="9"/>
    <line class="line" x1="220" y1="38" x2="280" y2="38"/>`;
  const links = ev.pairs.map((p) => {
    const s = scorerSpot.get(p.scorer), d = toward(s, 0.22);
    const off = bestFor.get(p.def).scorer !== p.scorer;
    return `<line class="link ${off ? 'off' : ''}" x1="${d[0]}" y1="${d[1]}" x2="${s[0]}" y2="${s[1]}"/>`;
  }).join('');
  const them = S.theirs.map((id, i) => {
    const [x, y] = SPOTS[i];
    return `<g class="them"><circle cx="${x}" cy="${y}" r="15"/>
      <text x="${x}" y="${y + 36}">${esc(last(id))}</text></g>`;
  }).join('');
  const us = ev.pairs.map((p) => {
    const [x, y] = toward(scorerSpot.get(p.scorer), 0.22);
    return `<g class="us"><circle cx="${x}" cy="${y}" r="13"/>
      <text x="${x}" y="${y - 20}">${esc(last(p.def))}</text>
      <text class="small" x="${x}" y="${y + 4.5}" style="fill:#fff">${p.pts.toFixed(0)}</text></g>`;
  }).join('');
  $('court').innerHTML = court + links + them + us;
}

loadData().then(init).catch(showError);
