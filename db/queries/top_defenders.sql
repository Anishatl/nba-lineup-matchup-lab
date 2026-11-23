-- League-wide (or one team's) best one-on-one defenders by points allowed
-- per 100 partial possessions in tracked matchups.
--
-- Usage (psql):
--   psql -d nba_lab -v season=2025-26 -v min_poss=1500 -v team=ALL -f db/queries/top_defenders.sql
--   psql -d nba_lab -v season=2025-26 -v min_poss=500  -v team=ATL -f db/queries/top_defenders.sql
--
-- Note: raw numbers, not adjusted for who each defender had to guard.
-- The matchup model (week 3) handles that adjustment.

WITH def AS (
    SELECT
        m.def_player_id,
        SUM(m.partial_poss)       AS poss,
        SUM(m.player_pts)         AS pts_allowed,
        SUM(m.fgm)                AS fgm,
        SUM(m.fga)                AS fga,
        SUM(m.switches_on)        AS switches
    FROM matchups m
    JOIN games g USING (game_id)
    WHERE g.season = :'season'
    GROUP BY m.def_player_id
    HAVING SUM(m.partial_poss) >= :min_poss
)
SELECT
    p.full_name,
    tm.abbreviation                                   AS team,
    ROUND(d.poss)                                     AS partial_poss,
    ROUND(100.0 * d.pts_allowed / d.poss, 1)          AS pts_per_100,
    ROUND(100.0 * d.fgm / NULLIF(d.fga, 0), 1)        AS fg_pct_allowed,
    d.switches
FROM def d
JOIN players p ON p.player_id = d.def_player_id
-- main team that season = team he played the most minutes for
LEFT JOIN LATERAL (
    SELECT t.abbreviation
    FROM player_team_season pts JOIN teams t USING (team_id)
    WHERE pts.player_id = d.def_player_id AND pts.season = :'season'
    ORDER BY pts.minutes DESC LIMIT 1
) tm ON TRUE
WHERE :'team' = 'ALL' OR tm.abbreviation = :'team'
ORDER BY pts_per_100 ASC
LIMIT 25;
