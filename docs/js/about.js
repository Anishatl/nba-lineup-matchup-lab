import { loadData, renderChrome, applyTeamColors, showError } from './ui.js';

const $ = (id) => document.getElementById(id);
const n = (x, d = 2) => (x == null ? '—' : Number(x).toFixed(d));
const int = (x) => (x == null ? '—' : Number(x).toLocaleString());
const missing = (what) => `<p class="missing">${what} will appear after the next model run and export.</p>`;

function table(head, rows, highlight = -1) {
  return `<div class="tablewrap"><table><thead><tr>${head.map((h, i) =>
    `<th class="${i ? 'n' : ''}">${h}</th>`).join('')}</tr></thead><tbody>${rows.map((r, k) =>
    `<tr>${r.map((c, i) => `<td class="${i ? 'n' : ''} ${k === highlight ? 'hi' : ''}">${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function coverage(meta) {
  const c = meta.coverage;
  if (!c) return missing('Data coverage');
  const tot = c.seasons.reduce((a, s) => ({ games: a.games + s.games, stints: a.stints + s.stints,
    exact: a.exact + s.exact, matchups: a.matchups + s.matchups }), { games: 0, stints: 0, exact: 0, matchups: 0 });
  const pct = (x) => `${(100 * x / tot.games).toFixed(1)}%`;
  const rows = c.seasons.map((s) => [s.season, int(s.games), int(s.stints), int(s.exact), int(s.matchups)])
    .concat([[`<b>Total</b>`, `<b>${int(tot.games)}</b>`, `<b>${int(tot.stints)} (${pct(tot.stints)})</b>`,
      `<b>${int(tot.exact)} (${pct(tot.exact)})</b>`, `<b>${int(tot.matchups)}</b>`]]);
  return table(['Season', 'Games', 'Stints loaded', 'Exact box-score match', 'Matchup data'], rows)
    + `<p class="note">${int(c.stint_rows)} stints and ${int(c.matchup_rows)} matchup rows. The most recent games are added once NBA.com posts their tracking data.</p>`;
}

function rapmVal(v) {
  if (!v) return missing('RAPM validation');
  const rows = v.rows.map((r) => [r.predictor, n(r.stint_rmse), n(r.game_rmse), n(r.game_corr, 3)]);
  const get = (name) => v.rows.find((r) => r.predictor === name);
  const rapm = get('RAPM'), raw = get('Raw +/-');
  let lead = '';
  if (rapm && raw) {
    const more = (rapm.game_corr ** 2 / raw.game_corr ** 2 - 1) * 100;
    lead = `<p>On ${v.test}, a season it never saw, RAPM explains about <b>${more.toFixed(0)}% more</b> of the game-to-game variation in margin than raw plus-minus (r = ${n(rapm.game_corr, 2)} vs. ${n(raw.game_corr, 2)}). Single games are noisy, so modest numbers are normal; winning on unseen data is what counts.</p>`;
  }
  return table(['Predictor', 'Stint error', 'Game margin error', 'Correlation with margin'], rows, rows.length - 1)
    + `<p class="note">Trained on ${v.train.join(', ')} · tested on ${v.test} · errors in points per 100 possessions (lower is better).</p>` + lead;
}

function calibration(meta) {
  const c = meta.validation?.matchup?.calibration;
  if (!c) return missing('Results');
  const rows = c.rows.map((r) => [r.model, n(r.game_rmse), n(r.game_corr, 3)]);
  return table(['Held-out ' + c.test, 'Game margin error', 'Correlation with margin'], rows, rows.length - 1);
}

function init(D) {
  renderChrome('about.html', D.meta);
  applyTeamColors('ATL', 'NYK');
  $('season').textContent = D.meta.season;
  $('coverage').innerHTML = coverage(D.meta);
  $('rapm-val').innerHTML = rapmVal(D.meta.validation?.rapm);
  $('cal').innerHTML = calibration(D.meta);
}

loadData().then(init).catch(showError);
