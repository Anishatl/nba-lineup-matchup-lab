# NBA Lineup & Matchup Lab

Pick any NBA team and any opponent, and find out:

1. **Which five players to put on the floor.** Every five-man combination from your roster is ranked against the opponent's lineup.
2. **Who should guard whom.** For each lineup, the lab finds the defensive assignment that gives up the fewest points.

**Live site:** https://anishatl.github.io/nba-lineup-matchup-lab/

---

## What's on the site

| Page | What it does |
|---|---|
| **Game plan** | The best lineups from your roster against any opposing five, and the top two defensive assignments for each |
| **Sandbox** | Set your own five and your own matchups, and see what they cost compared with the best assignment |
| **Players** | Offensive and defensive ratings, points saved, one-on-one numbers and flags for every player in the league |
| **Teams** | A one-page scouting report, including a lineup luck check (which lineups are over- or under-performing) |
| **How to use** | A step-by-step guide to every page |
| **About** | The data, the models and how they were tested |

## What's behind it

- **Data:** NBA.com play-by-play and player-tracking matchup data for every regular-season game since 2023-24, checked game by game against the official box scores.
- **RAPM** (regularized adjusted plus-minus): offensive (O-RAPM) and defensive (D-RAPM) ratings for every player.
- **A matchup model** that predicts how many points a given scorer produces against a given defender, including size effects.
- **An assignment optimizer** (Hungarian algorithm) that finds the best defensive matchups for every lineup.
- Every model was tested on a season it never saw before being used on the site.

## Built with

Python (pandas, NumPy, SciPy, scikit-learn), PostgreSQL, nba_api, pbpstats, and a static JavaScript front end hosted on GitHub Pages.

---

© 2026 Anish Bhatia. All rights reserved. No part of this project's code, models or data pipeline may be copied, modified or redistributed without written permission.
