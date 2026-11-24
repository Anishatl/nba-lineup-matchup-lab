-- One-on-one matchup skills from the latest matchup model run.
--
-- Usage (psql):
--   psql -d nba_lab -v team=ALL -v min_poss=1500 -f db/queries/matchup_ratings.sql
--   psql -d nba_lab -v team=ATL -v min_poss=500  -f db/queries/matchup_ratings.sql
--
-- defender: points per 100 matchup possessions allowed vs. average (negative = better)
-- scorer:   points per 100 matchup possessions scored vs. average (positive = better)
-- Both are adjusted for who each player faced (ridge regression), unlike
-- top_defenders.sql which uses raw totals.

WITH latest AS (
    SELECT run_id, seasons FROM model_runs WHERE model = 'matchup'
    ORDER BY run_id DESC LIMIT 1
),
current_team AS (
    SELECT DISTINCT ON (s.player_id) s.player_id, t.abbreviation
    FROM player_team_season s JOIN teams t USING (team_id) CROSS JOIN latest l
    WHERE s.season = (SELECT MAX(x) FROM unnest(l.seasons) x)
    ORDER BY s.player_id, s.minutes DESC
)
SELECT
    p.full_name,
    ct.abbreviation                   AS team,
    ROUND(r.defender_skill, 2)        AS defender,
    ROUND(r.def_poss)                 AS def_poss,
    ROUND(r.scorer_skill, 2)          AS scorer,
    ROUND(r.off_poss)                 AS off_poss,
    r.height_inches                   AS height
FROM matchup_ratings r
JOIN latest l USING (run_id)
JOIN players p USING (player_id)
LEFT JOIN current_team ct USING (player_id)
WHERE r.def_poss >= :min_poss
  AND (:'team' = 'ALL' OR ct.abbreviation = :'team')
ORDER BY r.defender_skill ASC
LIMIT 40;
