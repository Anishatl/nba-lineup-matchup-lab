import { loadData, renderChrome, applyTeamColors, teamSelect, getParams, setParams, fmt, esc, showError } from './ui.js';

const $ = (id) => document.getElementById(id);
const SD_PER_POSS = 1.15;   // typical spread of points on one possession
const RANGE = 30;           // bar chart spans -30..+30 pts per 100
const Z = 1.645;            // 90% band: pure chance flags ~10% of lineups
let D;

function colored(x, d = 2, unit = '', shown = x) {
  if (x == null) return '<span class="muted">—</span>';
  const cls = x > 0.05 ? 'pos-num' : x < -0.05 ? 'neg-num' : '';
  return `<span class="${cls}">${unit === '%' ? fmt.pct(shown) : fmt.signed(shown, d)}</span>`;
}

// Expected net vs an average lineup, using the calibrated weights: raw RAPM sums
// overshoot real results (held-out calibration gives a ~0.55 on offense and
// c ~0.45 on defense), so a + c scale the five players' offense and defense.
function expectedNet(ids) {
  const { a, c } = D.meta.calibration;
  return ids.reduce((t, id) => {
    const p = D.byId.get(id) || {};
    return t + a * (p.o_disp ?? 0) + c * (p.d_disp ?? 0);
  }, 0);
}

// Standard error of a net rating from op offensive / dp defensive possessions.
function noise(l) {
  const op = l.op ?? l.poss / 2, dp = l.dp ?? l.poss / 2;
  if (!op || !dp) return Infinity;
  return 100 * SD_PER_POSS * Math.sqrt(1 / op + 1 / dp);
}

function render(abbr) {
  const t = D.teams[abbr];
  applyTeamColors(abbr, abbr === 'NYK' ? 'BOS' : 'NYK');
  setParams({ team: abbr });
  $('teamname').textContent = `${t.city} ${t.name}`;

  const r = t.record;
  const net = r && r.ortg != null && r.drtg != null ? r.ortg - r.drtg : null;
  const ctx = t.def_context_pct;
  $('tiles').innerHTML = [
    r ? [`${r.w}–${r.l}`, 'Record', `${D.meta.season}`] : ['—', 'Record', 're-export data'],
    [net == null ? '—' : fmt.signed(net), 'Net rating', r?.ortg != null ? `${r.ortg} off · ${r.drtg} def` : ''],
    [ctx == null ? '—' : fmt.pct(ctx), 'Defensive scheme', 'effect on every one-on-one matchup'],
    [t.roster.length, 'Players used', `${t.lineups.length} lineups shown`],
  ].map(([b, s, small]) => `<div><b>${b}</b><span>${s}</span><small>${esc(small)}</small></div>`).join('');

  const pos = (x) => `${50 + 50 * Math.max(-RANGE, Math.min(RANGE, x)) / RANGE}%`;
  $('lineups').innerHTML = t.lineups.map((l) => {
    const exp = expectedNet(l.ids), act = l.net ?? 0, gap = act - exp, se = noise(l);
    const z = gap / se;
    const read = z > Z ? '<span class="verdict-tag hot">Running hot</span>'
      : z < -Z ? '<span class="verdict-tag cold">Running cold</span>'
        : '<span class="verdict-tag noise">Within noise</span>';
    const q = new URLSearchParams({ team: abbr, mine: l.ids.join(',') });
    const names = l.ids.map((id) => esc(fmt.last(D.byId.get(id)?.name || String(id)))).join('<span class="sep">·</span>');
    const mins = l.min != null ? `${Math.round(l.min)} min` : `${fmt.int(l.poss)} poss`;
    return `<tr>
      <td class="names"><a href="sandbox.html?${q}" title="Open in the sandbox">${names}</a>
        <div class="muted" style="font:12px var(--mono)">${mins}</div></td>
      <td class="n">${colored(act, 1)}</td>
      <td class="n">${colored(exp, 1)}</td>
      <td class="n">${colored(gap, 1)}</td>
      <td class="n muted">±${Number.isFinite(se) ? (Z * se).toFixed(0) : '—'}</td>
      <td><div class="bar" title="Actual ${fmt.signed(act)} · expected ${fmt.signed(exp)} · 90% noise band ±${(Z * se).toFixed(1)}">
        <div class="zero"></div>
        <div class="band" style="left:${pos(exp - Z * se)};width:calc(${pos(exp + Z * se)} - ${pos(exp - Z * se)})"></div>
        <div class="exp" style="left:${pos(exp)}"></div><div class="act" style="left:${pos(act)}"></div></div></td>
      <td>${read}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="7" class="muted">No lineups yet this season.</td></tr>';

  $('roster').innerHTML = t.roster.map((r) => {
    const p = D.byId.get(r.id) || { name: String(r.id) };
    return `<tr>
      <td class="names">${esc(p.name)}</td>
      <td class="n muted">${fmt.int(r.min)}</td>
      <td class="n">${colored(p.tot_disp)}</td>
      <td class="n">${colored(p.o_disp)}</td>
      <td class="n">${colored(p.d_disp)}</td>
      <td class="n">${colored(p.saved)}</td>
      <td class="n">${colored(p.df_pct == null ? null : -p.df_pct, 1, '%', p.df_pct)}</td>
      <td class="n">${colored(p.sc_pct, 1, '%')}</td>
      <td>${p.flag ? `<span class="tag ${p.flag}">${p.flag === 'switched_off' ? 'Switched off' : 'Help / rim'}</span>` : ''}</td>
    </tr>`;
  }).join('');
}

function init(data) {
  D = data;
  const q = getParams();
  const abbrs = Object.keys(D.teams).sort();
  const team = D.teams[q.team] ? q.team : (D.teams.ATL ? 'ATL' : abbrs[0]);
  renderChrome('teams.html', D.meta);
  $('season').textContent = D.meta.season;
  teamSelect($('team'), D.teams, team, render);
  render(team);
}

loadData().then(init).catch(showError);
