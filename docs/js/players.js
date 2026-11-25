import { loadData, renderChrome, applyTeamColors, getParams, setParams, fmt, esc, debounce, showError, TEAM_COLORS } from './ui.js';

const $ = (id) => document.getElementById(id);
let D;
const S = { sort: 'tot_disp', dir: -1, team: 'ALL', limit: 100 };

const COLS = [
  { key: 'name', label: 'Player', fmt: (p) => `<span class="names">${esc(p.name)}</span>` },
  { key: 'team', label: 'Team', fmt: (p) => p.team ? `<span class="teamchip" style="background:${(TEAM_COLORS[p.team] || ['#444'])[0]}">${p.team}</span>` : '—' },
  { key: 'tot_disp', label: 'RAPM', n: 1, fmt: (p) => colored(p.tot_disp, 2), title: 'Total RAPM (offense + defense), pts per 100' },
  { key: 'o_disp', label: 'Off', n: 1, fmt: (p) => colored(p.o_disp, 2), title: 'Offensive RAPM' },
  { key: 'd_disp', label: 'Def', n: 1, fmt: (p) => colored(p.d_disp, 2), title: 'Defensive RAPM (higher = better)' },
  { key: 'saved', label: 'Pts saved', n: 1, fmt: (p) => colored(p.saved, 2), title: 'Calibrated defensive impact: c × D-RAPM + b × one-on-one effect' },
  { key: 'df_pct', label: '1-on-1 D', n: 1, inv: true, fmt: (p) => colored(p.df_pct == null ? null : -p.df_pct, 1, '%', p.df_pct), title: "Change in his man's scoring vs. an average defender (negative = better)" },
  { key: 'sc_pct', label: 'Scoring', n: 1, fmt: (p) => colored(p.sc_pct, 1, '%'), title: 'Scoring vs. an average scorer, one-on-one' },
  { key: 'flag', label: 'Flag', fmt: (p) => p.flag ? `<span class="tag ${p.flag}">${p.flag === 'switched_off' ? 'Switched off' : 'Help / rim'}</span>` : '' },
  { key: 'poss', label: 'Poss', n: 1, fmt: (p) => `<span class="muted">${fmt.int(p.poss)}</span>` },
];

// Color by "goodness" but print the true value (for 1-on-1 D, negative is good).
function colored(goodness, d, unit = '', shown = goodness) {
  if (goodness == null) return '<span class="muted">—</span>';
  const cls = goodness > 0.05 ? 'pos-num' : goodness < -0.05 ? 'neg-num' : '';
  const text = unit === '%' ? fmt.pct(shown) : fmt.signed(shown, d);
  return `<span class="${cls}">${text}</span>`;
}

function init(data) {
  D = data;
  const q = getParams();
  renderChrome('players.html', D.meta);
  $('season').textContent = D.meta.season;
  const sel = $('team');
  sel.innerHTML = `<option value="ALL">NBA</option>` + Object.keys(D.teams).sort().map((a) => `<option>${a}</option>`).join('');
  S.team = q.team && D.teams[q.team] && q.view === 'team' ? q.team : 'ALL';
  sel.value = S.team;
  sel.onchange = () => { S.team = sel.value; S.limit = 100; render(); };
  ['q', 'minposs', 'active'].forEach((id) => $(id).addEventListener('input', debounce(render, 80)));
  $('head').innerHTML = COLS.map((c) => `<th class="sort ${c.n ? 'n' : ''}" data-k="${c.key}" title="${esc(c.title || '')}">${c.label}</th>`).join('');
  $('head').querySelectorAll('th').forEach((th) => th.onclick = () => {
    const k = th.dataset.k;
    S.dir = S.sort === k ? -S.dir : (k === 'name' || k === 'team' || k === 'df_pct' ? 1 : -1);
    S.sort = k;
    render();
  });
  render();
}

function render() {
  applyTeamColors(S.team === 'ALL' ? 'ATL' : S.team, 'NYK');
  setParams({ view: S.team === 'ALL' ? '' : 'team', team: S.team === 'ALL' ? getParams().team : S.team });
  const text = $('q').value.trim().toLowerCase();
  const minPoss = Number($('minposs').value) || 0;
  const active = $('active').checked;
  let rows = D.players.filter((p) => p.poss >= minPoss
    && (!active || p.active)
    && (S.team === 'ALL' || p.team === S.team)
    && (!text || p.name.toLowerCase().includes(text)));
  const k = S.sort;
  rows.sort((a, b) => {
    const x = a[k], y = b[k];
    if (x == null && y == null) return 0;
    if (x == null) return 1;
    if (y == null) return -1;
    return (typeof x === 'string' ? x.localeCompare(y) : x - y) * S.dir;
  });
  $('head').querySelectorAll('th').forEach((th) => th.classList.toggle('on', th.dataset.k === k));
  $('rows').innerHTML = rows.slice(0, S.limit).map((p) =>
    `<tr>${COLS.map((c) => `<td class="${c.n ? 'n' : ''}">${c.fmt(p)}</td>`).join('')}</tr>`).join('')
    || '<tr><td colspan="10" class="muted">No players match these filters.</td></tr>';
  const more = rows.length > S.limit;
  $('count').innerHTML = `${rows.length.toLocaleString()} players${more ? ` · showing ${S.limit}` : ''}. Click a column to sort.`
    + (more ? ' <button id="more">Show all</button>' : '');
  if (more) $('more').onclick = () => { S.limit = Infinity; render(); };
}

loadData().then(init).catch(showError);
