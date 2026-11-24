"""Small compatibility patches for pbpstats 1.3.x with newer NBA live feeds.

Starting in 2026-27, NBA's liveData play-by-play omits `personId` on events
that have no player (e.g. start of period) instead of sending `personId: 0`.
pbpstats only sets `player1_id` when the field is present, then reads
`event.player1_id` on every event while finding period starters, so every
2026-27 game crashed with:
    AttributeError: 'LiveStartOfPeriod' object has no attribute 'player1_id'

This restores the old behaviour (missing personId -> 0). Import this module
before parsing any games.
"""
from pbpstats.resources.enhanced_pbp.live.enhanced_pbp_item import LiveEnhancedPbpItem

_original_init = LiveEnhancedPbpItem.__init__


def _patched_init(self, item, game_id):
    _original_init(self, item, game_id)
    if not hasattr(self, "player1_id"):
        self.player1_id = 0


if not "_nba_lab_patched" in vars(LiveEnhancedPbpItem):
    LiveEnhancedPbpItem.__init__ = _patched_init
    LiveEnhancedPbpItem._nba_lab_patched = True


# --- Rebound -> missed shot lookup --------------------------------------------
# LiveRebound.missed_shot finds the missed shot by `shotActionNumber`. In some
# 2026-27 games that number doesn't match any earlier event, and pbpstats then
# crashes on a typo (`Rebound.EventOrderError`). Fall back to the most recent
# missed field goal / free throw in the same period, as older feeds did.
from pbpstats.resources.enhanced_pbp import FieldGoal, FreeThrow, StartOfPeriod  # noqa: E402
from pbpstats.resources.enhanced_pbp.live.rebound import LiveRebound  # noqa: E402
from pbpstats.resources.enhanced_pbp.rebound import EventOrderError  # noqa: E402


def _missed_shot(self):
    target = getattr(self, "shot_action_number", None)
    prev = self.previous_event
    while target is not None and prev is not None:
        if getattr(prev, "event_num", None) == target:
            return prev
        prev = prev.previous_event
    prev = self.previous_event
    while prev is not None and not isinstance(prev, StartOfPeriod):
        if isinstance(prev, (FieldGoal, FreeThrow)) and not prev.is_made:
            return prev
        prev = prev.previous_event
    raise EventOrderError(f"no missed shot found before rebound {self}")


if not "_nba_lab_patched" in vars(LiveRebound):
    LiveRebound.missed_shot = property(_missed_shot)
    LiveRebound._nba_lab_patched = True
