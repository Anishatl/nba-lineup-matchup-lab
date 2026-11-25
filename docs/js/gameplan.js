import { loadData, renderChrome, applyTeamColors, teamSelect, getParams, setParams, playerSearch,
  mostUsedLineup, defaultRoster, fmt, esc, debounce, showError, label } from './ui.js';

const MAX_ROSTER = 15;
const $ = (id) => document.getElementById(id);

let D, S;

function init(data) {
  D = data;
  const q = getParams();
  const abbrs = Object.keys(D.teams).sort();
  const team = D.teams[q.team] ? q.team : (D.teams.ATL ? 'ATL' : abbrs[0]);
  const opp = D.teams[q.opp] && q.opp !== team ? q.opp : (team !== 'NYK' && D.teams.NYK ? 'NYK' : abbrs.find((a) => a !== team));
  S = {
    team, opp,
    slider: Number(q.slider) || 1, size: q.size != null ? Number(q.size) : 1,
    theirs: q.their ? q.their.split(',').map(Number) : mostUsedLineup(D.teams, opp),
    roster: new Set(defaultRoster(D.teams, team)),
    extra: [],            // players added from outside the roster
    selected: 0,
  };
  renderChrome('index.html', D.meta);
  $('season').textContent = D.meta.season;
  teamSelect($('team'), D.teams, S.team, (v) => { S.team = v; S.roster = new Set(defaultRoster(D.teams, v)); S.extra = []; refresh(); });
  teamSelect($('opp'), D.teams, S.opp, (v) => { S.opp = v; S.theirs = mostUsedLineup(D.teams, v); refresh(); });
  $('slider').value = S.slider; $('size').value = S.size;
  $('slider').oninput = () => { S.slider = Number($('slider').value); compute(); };
  $('size').oninput = () => { S.size = Number($('size').value); compute(); };
  $('reset-their').onclick = () => { S.theirs = mostUsedLineup(D.teams, S.opp); refresh(); };
  playerSearch($('add-player'), D, (id) => {
    if (!S.extra.includes(id) && !(D.teams[S.team].roster || []).some((r) => r.id === id)) S.extra.push(id);
    S.roster.add(id);
    $('add-player').value = '';
    refresh();
  });
  refresh();
}

function refresh() {
  applyTeamColors(S.team, S.opp);
  $('team').value = S.team; $('opp').value = S.opp;
  renderTheirs();
  renderRoster();
  compute();
}

function name(id) { return D.byId.get(id)?.name || String(id); }

function renderTheirs() {
  $('their-slots').innerHTML = S.theirs.map((id, i) =>
    `<div class="slot"><span class="pos">${i + 1}</span><input type="search" data-i="${i}" value="${esc(D.byId.get(id) ? label(D.byId.get(id)) : '')}"></div>`).join('');
  $('their-slots').querySelectorAll('input').forEach((inp) => playerSearch(inp, D, (id) => {
    const i = Number(inp.dataset.i);
    if (S.theirs.includes(id) && S.theirs[i] !== id) { inp.classList.add('bad'); return; }
    S.theirs[i] = id;
    compute();
  }));
}

function renderRoster() {
  const base = (D.teams[S.team].roster || []).map((r) => ({ id: r.id, min: r.min }));
  const rows = base.concat(S.extra.map((id) => ({ id, min: null })));
  $('roster').innerHTML = rows.map((r) => `
    <label><input type="checkbox" value="${r.id}" ${S.roster.has(r.id) ? 'checked' : ''}>
      <span>${esc(name(r.id))}${r.min == null ? ' <span class="muted">(added)</span>' : ''}</span>
      <span class="num">${r.min == null ? '' : `${r.min} min`}</span></label>`).join('');
  $('roster').querySelectorAll('input').forEach((cb) => cb.onchange = () => {
    const id = Number(cb.value);
    if (cb.checked) {
      if (S.roster.size >= MAX_ROSTER) { cb.checked = false; return; }
      S.roster.add(id);
    } else S.roster.delete(id);
    compute();
  });
}

const compute = debounce(() => {
  $('slider-out').textContent = `${S.slider.toFixed(1)}×`;
  $('size-out').textContent = `${Math.round(S.size * 100)}%`;
  setParams({ team: S.team, opp: S.opp, slider: S.slider === 1 ? '' : S.slider, size: S.size === 1 ? '' : S.size,
              their: S.theirs.join(',') });
  const roster = [...S.roster].filter((id) => !S.theirs.includes(id));
  $('roster-count').textContent = `${roster.length} selected`;
  if (roster.length < 5) {
    $('lineups').innerHTML = '<tr><td colspan="5" class="muted">Select at least 5 of your players.</td></tr>';
    $('assignments').innerHTML = '';
    return;
  }
  D.engine.sizeStrength = S.size;
  S.ranked = D.engine.rankLineups(roster, S.theirs, S.slider, 10);
  S.selected = Math.min(S.selected, S.ranked.length - 1);
  renderLineups();
  renderAssignments();
}, 60);

function renderLineups() {
  $('lineups').innerHTML = S.ranked.map((r, i) => `
    <tr class="pick ${i === S.selected ? 'sel' : ''}" data-i="${i}">
      <td class="rank">${i + 1}</td>
      <td class="names">${r.lineup.map((id) => esc(fmt.last(name(id)))).join('<span class="sep">·</span>')}</td>
      <td class="n"><span class="box ${i === 0 ? 'best' : ''}">${fmt.signed(r.net)}</span></td>
      <td class="n">${fmt.num(r.scored)}</td>
      <td class="n">${fmt.num(r.allowed)}</td>
    </tr>`).join('');
  $('lineups').querySelectorAll('tr').forEach((tr) => tr.onclick = () => {
    S.selected = Number(tr.dataset.i);
    renderLineups();
    renderAssignments();
  });
}

function renderAssignments() {
  const pick = S.ranked[S.selected];
  const opts = D.engine.topAssignments(pick.lineup, S.theirs, 2);
  const b = D.meta.calibration.b;
  $('assign-for').textContent = `lineup #${S.selected + 1}`;
  $('assignments').innerHTML = opts.map((o, k) => {
    const gap = o.total - opts[0].total;
    const head = k === 0 ? 'Option 1 · best'
      : `Option 2 <span class="num">+${gap.toFixed(1)} matchup pts/100 · about ${fmt.signed(b * gap, 2)} allowed per 100 after calibration</span>`;
    const cards = [...o.pairs].sort((x, y) => y.pts - x.pts).map((p) => `
      <div class="card"><div class="who">${esc(fmt.last(name(p.def)))}</div>
        <div class="on">on <b>${esc(fmt.last(name(p.scorer)))}</b></div>
        <div class="pts">${p.pts.toFixed(1)} pts/100</div></div>`).join('');
    return `<div class="opt-head">${head}</div><div class="cards">${cards}</div>`;
  }).join('');
  const q = new URLSearchParams({ team: S.team, opp: S.opp, mine: pick.lineup.join(','), their: S.theirs.join(',') });
  $('to-sandbox').href = `sandbox.html?${q}`;
}

loadData().then(init).catch(showError);
