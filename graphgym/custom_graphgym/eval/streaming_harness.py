"""
streaming_harness.py

Mixed-cadence rolling-window mechanics for the real-time (1-minute) cadence
robustness study -- see the plan at
~/.claude/plans/squishy-wobbling-bunny.md ("Robustness Test Plan for Input
Cadence Mismatch at Inference Time") for the full design rationale. This
module implements only the WINDOW MECHANICS (pure, model-agnostic, CPU-only,
no torch/GPU dependency) -- wiring a real model's forward() into
`run_stream()` is Phase 2 work, done once trained checkpoints exist and can
be evaluated without contending with the in-progress training run's GPU
usage.

Window design (confirmed with the user, not assumed):
  - A fixed-length 288-slot window: positions 1-283 hold 5-minute-resolution
    consumption values (older history, ~23.6h of context); positions
    284-288 hold the 5 most recent 1-minute-resolution readings (native
    cadence, matching live deployment ticks).
  - Only the CONSUMPTION channel is refreshed at 1-minute cadence here --
    PV/generation is natively 5-min (the inverter stream's own ceiling) and
    is never part of this window's refresh; it only ever arrives at
    5-min-aligned ticks and feeds Phase 2's sparse accuracy check, not this
    module.
  - `coarse` (283 slots) only advances once every 5 ticks -- not continuously
    -- because a genuine 5-minute-resolution measurement can only refresh
    every 5 real minutes; there is no such thing as a continuously-updating
    5-min value between real 5-min ticks. The just-completed group of 5
    one-minute readings collapses into the new 5-min-equivalent coarse
    entry via **median** (matching
    /home/sagemaker-user/exploratory-data-analysis/src/ingestion.py's
    resample_to_cadence(), which median-aggregates point/snapshot (KW/W)
    columns -- consumption_w is one -- after a +half-cadence shift to align
    to the interval midpoint; the shift itself isn't needed here since we're
    aging a fixed-size rolling buffer, not resampling a static dataframe,
    but the aggregation function must match).
  - `fine` (5 slots) is a plain sliding window of the most recent ticks
    (`collections.deque(maxlen=5)`); it always shows "the 5 most recent
    ticks" once >=5 real ticks have arrived. Before that (the first 4
    ticks after streaming starts), it's a mix of leftover seed values (the
    clean 5-min window's own tail, so `as_array()` is always a valid
    288-length window, even on the very first call) and however many real
    ticks have arrived so far -- a gradual, non-jarring cold-start
    transition rather than a discontinuity.
  - No double-counting: each real tick contributes to exactly one median
    computation (the one that fires when it's the 5th tick in `fine` since
    the last aging event), then naturally ages out of `fine` on the next
    append (deque maxlen eviction).

Usage (window mechanics only, no model call):
    from custom_graphgym.eval.streaming_harness import WindowState

    window = WindowState(initial_5min_window=clean_288_values)
    for reading in one_minute_ticks:
        window.push_tick(reading)
        current_288_values = window.as_array()
"""
from collections import deque
from statistics import median
from typing import Sequence


class WindowState:
    """Mixed-cadence 288-slot rolling window for one household's consumption
    series. See module docstring for the full design rationale."""

    COARSE_LEN = 283
    FINE_LEN = 5
    TOTAL_LEN = COARSE_LEN + FINE_LEN

    def __init__(self, initial_5min_window: Sequence[float]):
        if len(initial_5min_window) != self.TOTAL_LEN:
            raise ValueError(
                f"initial_5min_window must have exactly {self.TOTAL_LEN} "
                f"values (a clean 5-min-native window), got "
                f"{len(initial_5min_window)}"
            )
        self.coarse: deque = deque(
            initial_5min_window[: self.COARSE_LEN], maxlen=self.COARSE_LEN
        )
        # Seeded with the clean window's own tail -- stale until real ticks
        # displace them, but valid, so as_array() is correct from the very
        # first call, before any real 1-min tick has arrived.
        self.fine: deque = deque(
            initial_5min_window[self.COARSE_LEN :], maxlen=self.FINE_LEN
        )
        self._ticks_since_last_age = 0
        self.n_real_ticks_seen = 0

    def push_tick(self, one_min_value: float) -> None:
        """Advance the window by one real 1-minute tick."""
        self.fine.append(one_min_value)  # auto-evicts fine's oldest
        self.n_real_ticks_seen += 1
        self._ticks_since_last_age += 1
        if self._ticks_since_last_age == self.FINE_LEN:
            # 5 genuine 1-min ticks now sit in `fine` (the ones that
            # triggered this: only true once n_real_ticks_seen >= FINE_LEN,
            # since _ticks_since_last_age can't reach FINE_LEN before then)
            # -- collapse them into one 5-min-equivalent coarse entry.
            self.coarse.append(median(self.fine))  # auto-evicts coarse's oldest
            self._ticks_since_last_age = 0

    def as_array(self) -> list:
        """Current 288-length mixed-cadence window, oldest to newest."""
        return list(self.coarse) + list(self.fine)

    def fine_is_all_real(self) -> bool:
        """True once every slot in `fine` holds a genuine 1-min reading
        (rather than a leftover cold-start seed value) -- useful for tests
        and for Phase 2 deciding when a prediction is "genuinely" mixed-
        cadence vs. still transitioning."""
        return self.n_real_ticks_seen >= self.FINE_LEN
