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


if not getattr(LiveEnhancedPbpItem, "_nba_lab_patched", False):
    LiveEnhancedPbpItem.__init__ = _patched_init
    LiveEnhancedPbpItem._nba_lab_patched = True
