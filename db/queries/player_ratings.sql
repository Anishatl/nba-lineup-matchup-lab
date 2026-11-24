-- RAPM player ratings from the latest model run, for one team or the whole league.
--
-- Usage (psql):
--   psql -d nba_lab -v team=ATL -v min_poss=1000 -f db/queries/player_ratings.sql
--   psql -d nba_lab -v team=ALL -v min_poss=5000 -f db/queries/player_ratings.sql
--
-- d_rtg is flipped for reading (higher = better defense). The database stores
-- the raw coefficient, where negative = better defense.

WITH latest AS (
    SELECT run_id, seasons FROM model_runs WHERE model = 'rapm'
    ORDER BY run_id DESC LIMIT 1
),
current_team AS (
    -- team each player logged the most minutes for in the latest season of the run
    SELECT DISTINCT ON (pts.player_id) pts.player_id, t.abbreviation
    FROM player_team_season pts
    JOIN teams t USING (team_id)
    CROSS JOIN latest l
    WHERE pts.season = (SELECT MAX(s) FROM unnest(l.seasons) s)
    ORDER BY pts.player_id, pts.minutes DESC
)
SELECT
    p.full_name,
    ct.abbreviation                       AS team,
    ROUND(r.total_rapm, 2)                AS total,
    ROUND(r.o_rapm, 2)                    AS o_rtg,
    ROUND(-r.d_rapm, 2)                   AS d_rtg,
    r.off_poss + r.def_poss               AS poss
FROM player_ratings r
JOIN latest l USING (run_id)
JOIN players p USING (player_id)
LEFT JOIN current_team ct USING (player_id)
WHERE r.off_poss + r.def_poss >= :min_poss
  AND (:'team' = 'ALL' OR ct.abbreviation = :'team')
ORDER BY r.total_rapm DESC
LIMIT 40;
