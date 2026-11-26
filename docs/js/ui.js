// Shared helpers: data loading, team colors, navigation, player search, formatting.
import { Engine } from './model.js';

export const TEAM_COLORS = {
  ATL: ['#C8102E', '#FDB927'], BOS: ['#007A33', '#BA9653'], BKN: ['#000000', '#FFFFFF'],
  CHA: ['#1D1160', '#00788C'], CHI: ['#CE1141', '#000000'], CLE: ['#860038', '#FDBB30'],
  DAL: ['#00538C', '#B8C4CA'], DEN: ['#0E2240', '#FEC524'], DET: ['#C8102E', '#1D42BA'],
  GSW: ['#1D428A', '#FFC72C'], HOU: ['#CE1141', '#C4CED4'], IND: ['#002D62', '#FDBB30'],
  LAC: ['#C8102E', '#1D428A'], LAL: ['#552583', '#FDB927'], MEM: ['#5D76A9', '#12173F'],
  MIA: ['#98002E', '#F9A01B'], MIL: ['#00471B', '#EEE1C6'], MIN: ['#0C2340', '#236192'],
  NOP: ['#0C2340', '#C8102E'], NYK: ['#006BB6', '#F58426'], OKC: ['#007AC1', '#EF3B24'],
  ORL: ['#0077C0', '#C4CED4'], PHI: ['#006BB6', '#ED174C'], PHX: ['#1D1160', '#E56020'],
  POR: ['#E03A3E', '#000000'], SAC: ['#5A2D81', '#63727A'], SAS: ['#000000', '#C4CED4'],
  TOR: ['#CE1141', '#000000'], UTA: ['#002B5C', '#F9A01B'], WAS: ['#002B5C', '#E31837'],
};

export const PAGES = [
  ['index.html', 'Game plan'],
  ['sandbox.html', 'Sandbox'],
  ['players.html', 'Players'],
  ['teams.html', 'Teams'],
  ['guide.html', 'How to use'],
  ['about.html', 'About'],
];

export async function loadData() {
  const get = (f) => fetch(`data/${f}`).then((r) => {
    if (!r.ok) throw new Error(`Couldn't load data/${f}`);
    return r.json();
  });
  const [meta, players, teams] = await Promise.all([get('meta.json'), get('players.json'), get('teams.json')]);
  const byId = new Map(players.map((p) => [p.id, p]));
  return { meta, players, teams, byId, engine: new Engine(meta, players) };
}

// ---------- URL state (team / opp carry across pages) ----------
export function getParams() { return Object.fromEntries(new URLSearchParams(location.search)); }
export function setParams(obj) {
  const q = new URLSearchParams({ ...getParams(), ...obj });
  for (const [k, v] of [...q]) if (v === '' || v == null) q.delete(k);
  history.replaceState(null, '', `${location.pathname}?${q}`);
  document.querySelectorAll('[data-nav]').forEach((a) => {
    const keep = new URLSearchParams();
    ['team', 'opp'].forEach((k) => q.get(k) && keep.set(k, q.get(k)));
    a.href = `${a.dataset.nav}?${keep}`;
  });
}

// ---------- header / footer ----------
export function renderChrome(active, meta) {
  const nav = document.getElementById('nav');
  nav.innerHTML = `
    <a class="brand" href="index.html">NBA Lineup <span>&amp;</span> Matchup Lab</a>
    <div class="links">${PAGES.map(([href, label]) =>
      `<a data-nav="${href}" href="${href}" class="${href === active ? 'on' : ''}">${label}</a>`).join('')}</div>`;
  const foot = document.getElementById('foot');
  if (foot && meta) {
    foot.innerHTML = `
      <span>Data: NBA.com play-by-play and matchup tracking, ${meta.seasons[0]} to ${meta.seasons.at(-1)}</span>
      <span>Updated ${meta.updated}</span>
      <span>© 2026 Anish Bhatia. All rights reserved.</span>`;
  }
  setParams({});
}

export function applyTeamColors(team, opp) {
  const root = document.documentElement.style;
  const [t1, t2] = TEAM_COLORS[team] || ['#444', '#888'];
  const [o1, o2] = TEAM_COLORS[opp] || ['#444', '#888'];
  root.setProperty('--team', t1); root.setProperty('--team2', t2);
  root.setProperty('--opp', o1); root.setProperty('--opp2', o2);
  root.setProperty('--team-hi', readable(t1, t2));
  root.setProperty('--opp-hi', readable(o1, o2));
}

// A team color bright enough to read on the black background.
function readable(c1, c2) {
  const lum = (hex) => {
    const n = parseInt(hex.slice(1), 16);
    const [r, g, b] = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => v / 255);
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  return lum(c1) > 0.18 ? c1 : (lum(c2) > 0.18 ? c2 : '#DDDDDD');
}

// ---------- team picker (lives inside the score bar) ----------
export function teamSelect(el, teams, value, onChange) {
  el.innerHTML = Object.keys(teams).sort().map((a) =>
    `<option value="${a}" ${a === value ? 'selected' : ''}>${a}</option>`).join('');
  el.onchange = () => onChange(el.value);
}

// ---------- player search: input + datalist ----------
// pool: list of player ids to offer (e.g. one team's roster); omit for the whole league.
// Lists are sorted A-Z by name.
const lists = new Map();
function datalistFor(data, pool) {
  const key = pool ? `pl-${[...pool].sort((x, y) => x - y).join('-')}` : 'pl-all';
  if (!lists.has(key)) {
    const players = (pool ? pool.map((id) => data.byId.get(id)).filter(Boolean) : data.players)
      .slice().sort((x, y) => x.name.localeCompare(y.name));
    const dl = document.createElement('datalist');
    dl.id = key;
    dl.innerHTML = players.map((p) => `<option value="${esc(label(p))}"></option>`).join('');
    document.body.appendChild(dl);
    lists.set(key, players.map((p) => p.id));
  }
  return { id: key, ids: lists.get(key) };
}

export function playerSearch(input, data, onPick, pool = null) {
  const dl = datalistFor(data, pool);
  input.setAttribute('list', dl.id);
  input.addEventListener('focus', () => input.select());
  input.addEventListener('change', () => {
    const p = findPlayer(data, input.value, dl.ids);
    if (p) onPick(p.id);
    else input.classList.add('bad');
  });
  input.addEventListener('input', () => input.classList.remove('bad'));
}

// Players on a team's roster this season, plus any extra ids (e.g. the current lineup).
export function teamPool(teams, abbr, extra = []) {
  const ids = (teams[abbr]?.roster || []).map((r) => r.id);
  return [...new Set(ids.concat(extra))];
}

export function label(p) { return `${p.name} (${p.team || 'FA'})`; }

export function findPlayer(data, text, ids = null) {
  const t = text.trim().toLowerCase();
  if (!t) return null;
  const pool = ids ? ids.map((id) => data.byId.get(id)).filter(Boolean) : data.players;
  const exact = pool.find((p) => label(p).toLowerCase() === t || p.name.toLowerCase() === t);
  if (exact) return exact;
  const hits = pool.filter((p) => p.name.toLowerCase().includes(t));
  return hits.length === 1 ? hits[0] : null;
}

// ---------- lineup defaults ----------
export function mostUsedLineup(teams, abbr) {
  const t = teams[abbr];
  if (t?.lineups?.length) return [...t.lineups[0].ids];
  return (t?.roster || []).slice(0, 5).map((r) => r.id);
}

// Default: the 12 players with the most minutes this season.
export function defaultRoster(teams, abbr, n = 12) {
  return (teams[abbr]?.roster || []).filter((x) => x.min > 0).slice(0, n).map((x) => x.id);
}

// ---------- formatting ----------
export const fmt = {
  signed: (x, d = 1) => (x == null || Number.isNaN(x) ? '—' : `${x > 0 ? '+' : x < 0 ? '−' : ''}${Math.abs(x).toFixed(d)}`),
  num: (x, d = 1) => (x == null || Number.isNaN(x) ? '—' : x.toFixed(d)),
  pct: (x) => (x == null ? '—' : `${x > 0 ? '+' : x < 0 ? '−' : ''}${Math.abs(x).toFixed(1)}%`),
  int: (x) => (x == null ? '—' : Math.round(x).toLocaleString()),
  last: (name) => name.split(' ').slice(1).join(' ') || name,
  height: (h) => (h ? `${Math.floor(h / 12)}′${Math.round(h % 12)}″` : '—'),
};

export function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

export function debounce(fn, ms = 120) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

export function showError(err) {
  const main = document.querySelector('main');
  main.innerHTML = `<div class="error"><h2>Couldn't load the data</h2><p>${esc(err.message)}</p>
    <p>If you're opening the file directly, serve the folder instead: <code>python -m http.server -d docs</code></p></div>`;
}
