-- Best 5-man lineups for one team in one season, by net rating.
--
-- Usage (psql):
--   psql -d nba_lab -v team=ATL -v season=2025-26 -v min_poss=150 -f db/queries/best_lineups.sql
--
-- Ratings are points per 100 possessions. min_poss filters out tiny samples.

WITH t AS (
    SELECT team_id FROM teams WHERE abbreviation = :'team'
),
team_side AS (
    SELECT
        CASE WHEN g.home_team_id = t.team_id THEN s.home_players ELSE s.away_players END AS lineup,
        CASE WHEN g.home_team_id = t.team_id THEN s.home_poss    ELSE s.away_poss    END AS off_poss,
        CASE WHEN g.home_team_id = t.team_id THEN s.away_poss    ELSE s.home_poss    END AS def_poss,
        CASE WHEN g.home_team_id = t.team_id THEN s.home_pts     ELSE s.away_pts     END AS pts_for,
        CASE WHEN g.home_team_id = t.team_id THEN s.away_pts     ELSE s.home_pts     END AS pts_against,
        s.seconds
    FROM stints s
    JOIN games g USING (game_id)
    CROSS JOIN t
    WHERE g.season = :'season'
      AND t.team_id IN (g.home_team_id, g.away_team_id)
),
by_lineup AS (
    SELECT
        lineup,
        ROUND(SUM(seconds) / 60.0, 1)                            AS minutes,
        SUM(off_poss) + SUM(def_poss)                            AS total_poss,
        ROUND(100.0 * SUM(pts_for)     / NULLIF(SUM(off_poss), 0), 1) AS off_rtg,
        ROUND(100.0 * SUM(pts_against) / NULLIF(SUM(def_poss), 0), 1) AS def_rtg
    FROM team_side
    GROUP BY lineup
    HAVING SUM(off_poss) + SUM(def_poss) >= :min_poss
)
SELECT
    (SELECT string_agg(p.full_name, ', ' ORDER BY p.full_name)
       FROM players p WHERE p.player_id = ANY (b.lineup)) AS players,
    b.minutes,
    b.total_poss,
    b.off_rtg,
    b.def_rtg,
    b.off_rtg - b.def_rtg AS net_rtg
FROM by_lineup b
ORDER BY net_rtg DESC
LIMIT 20;
