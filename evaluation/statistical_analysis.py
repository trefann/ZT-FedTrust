"""Statistical summaries and significance testing across repeated (multi-seed) runs (Phase 5).

Research-integrity guardrail: `paired_significance_test` refuses to compute
a p-value from too few paired observations (`MIN_PAIRED_SAMPLES`) rather
than silently returning a number that looks precise but isn't meaningful —
per the explicit requirement not to report significance without enough
repeated runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from scipy import stats

MIN_PAIRED_SAMPLES = 3  # fewer than this, a paired t-test isn't meaningful


@dataclass
class SummaryStatistics:
    """Descriptive statistics over repeated-seed results for one metric."""

    n: int
    mean: float
    stdev: float
    median: float
    minimum: float
    maximum: float
    ci95_low: float
    ci95_high: float


def summarize(values: List[float]) -> SummaryStatistics:
    """Compute mean/stdev/median/min/max/95% CI over `values` (one per seed).

    Uses the t-distribution for the CI (appropriate for the small sample
    sizes repeated-seed experiments typically have) rather than assuming a
    normal distribution. With a single value, stdev and the CI half-width
    are both 0 — there's no spread to measure from one observation.
    """
    if not values:
        raise ValueError("Cannot summarize an empty list of values.")

    n = len(values)
    mean = sum(values) / n
    if n > 1:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        stdev = math.sqrt(variance)
    else:
        stdev = 0.0

    sorted_values = sorted(values)
    mid = n // 2
    median = (
        sorted_values[mid] if n % 2 == 1 else (sorted_values[mid - 1] + sorted_values[mid]) / 2
    )

    if n > 1:
        standard_error = stdev / math.sqrt(n)
        t_critical = float(stats.t.ppf(0.975, df=n - 1))
        margin = t_critical * standard_error
    else:
        margin = 0.0

    return SummaryStatistics(
        n=n,
        mean=mean,
        stdev=stdev,
        median=median,
        minimum=min(values),
        maximum=max(values),
        ci95_low=mean - margin,
        ci95_high=mean + margin,
    )


@dataclass
class SignificanceTestResult:
    """Result of a paired significance test between two matched-seed samples."""

    test_name: str
    n_pairs: int
    statistic: float
    p_value: float
    significant_at_0_05: bool
    mean_difference: float


def paired_significance_test(
    baseline_values: List[float], comparison_values: List[float], alpha: float = 0.05
) -> Optional[SignificanceTestResult]:
    """Paired t-test comparing `comparison_values` against `baseline_values`.

    Values must be paired by matching seed/index — `baseline_values[i]` and
    `comparison_values[i]` must come from the same seed, differing only in
    which algorithm/condition was run. A paired test is only valid under
    that correspondence; it is the caller's responsibility to align the
    lists that way (e.g. by seed) before calling this.

    Returns `None` — rather than a fabricated statistic — when there are
    fewer than `MIN_PAIRED_SAMPLES` pairs.
    """
    if len(baseline_values) != len(comparison_values):
        raise ValueError(
            f"Paired test requires equal-length samples, got "
            f"{len(baseline_values)} vs {len(comparison_values)}"
        )
    if len(baseline_values) < MIN_PAIRED_SAMPLES:
        return None

    raw_statistic, raw_p_value = stats.ttest_rel(comparison_values, baseline_values)
    statistic = float(raw_statistic)
    p_value = float(raw_p_value)
    mean_difference = (
        sum(comparison_values) / len(comparison_values) - sum(baseline_values) / len(baseline_values)
    )
    return SignificanceTestResult(
        test_name="paired_t_test",
        n_pairs=len(baseline_values),
        statistic=statistic,
        p_value=p_value,
        significant_at_0_05=bool(p_value < alpha),
        mean_difference=mean_difference,
    )
