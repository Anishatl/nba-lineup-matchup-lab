-- On/off splits: how a team performs with each player on vs. off the floor.
--
-- Usage (psql):
--   psql -d nba_lab -v team=ATL -v season=2025-26 -f db/queries/player_on_off.sql

WITH t AS (
    SELECT team_id FROM teams WHERE abbreviation = :'team'
),
team_side AS (
    SELECT
        CASE WHEN g.home_team_id = t.team_id THEN s.home_players ELSE s.away_players END AS lineup,
        CASE WHEN g.home_team_id = t.team_id THEN s.home_poss    ELSE s.away_poss    END AS off_poss,
        CASE WHEN g.home_team_id = t.team_id THEN s.away_poss    ELSE s.home_poss    END AS def_poss,
        CASE WHEN g.home_team_id = t.team_id THEN s.home_pts     ELSE s.away_pts     END AS pts_for,
        CASE WHEN g.home_team_id = t.team_id THEN s.away_pts     ELSE s.home_pts     END AS pts_against
    FROM stints s
    JOIN games g USING (game_id)
    CROSS JOIN t
    WHERE g.season = :'season'
      AND t.team_id IN (g.home_team_id, g.away_team_id)
),
roster AS (
    SELECT pts.player_id
    FROM player_team_season pts CROSS JOIN t
    WHERE pts.team_id = t.team_id AND pts.season = :'season' AND pts.minutes >= 200
),
splits AS (
    SELECT
        r.player_id,
        (r.player_id = ANY (ts.lineup)) AS on_court,
        SUM(ts.off_poss) AS off_poss, SUM(ts.def_poss) AS def_poss,
        SUM(ts.pts_for) AS pts_for,  SUM(ts.pts_against) AS pts_against
    FROM roster r CROSS JOIN team_side ts
    GROUP BY r.player_id, on_court
),
net AS (
    SELECT player_id, on_court, off_poss + def_poss AS poss,
           100.0 * pts_for / NULLIF(off_poss, 0) - 100.0 * pts_against / NULLIF(def_poss, 0) AS net_rtg
    FROM splits
)
SELECT
    p.full_name,
    on_s.poss                                  AS on_poss,
    ROUND(on_s.net_rtg, 1)                     AS on_net,
    ROUND(off_s.net_rtg, 1)                    AS off_net,
    ROUND(on_s.net_rtg - off_s.net_rtg, 1)     AS on_off_diff
FROM net on_s
JOIN net off_s ON off_s.player_id = on_s.player_id AND NOT off_s.on_court
JOIN players p ON p.player_id = on_s.player_id
WHERE on_s.on_court
ORDER BY on_off_diff DESC;
