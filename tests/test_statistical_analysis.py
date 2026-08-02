"""Tests for evaluation/statistical_analysis.py: summary stats and the paired significance test."""

import pytest

from evaluation.statistical_analysis import MIN_PAIRED_SAMPLES, paired_significance_test, summarize


def test_summarize_hand_computed():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = summarize(values)
    assert stats.n == 5
    assert stats.mean == 3.0
    assert stats.median == 3.0
    assert stats.minimum == 1.0
    assert stats.maximum == 5.0
    assert stats.stdev == pytest.approx(1.5811388, rel=1e-5)
    assert stats.ci95_low < stats.mean < stats.ci95_high


def test_summarize_single_value_has_zero_spread():
    stats = summarize([0.75])
    assert stats.n == 1
    assert stats.mean == 0.75
    assert stats.stdev == 0.0
    assert stats.ci95_low == stats.ci95_high == 0.75


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        summarize([])


def test_summarize_even_count_median_is_average_of_middle_two():
    stats = summarize([1.0, 2.0, 3.0, 4.0])
    assert stats.median == 2.5


def test_paired_significance_test_returns_none_below_minimum_samples():
    assert MIN_PAIRED_SAMPLES >= 2
    too_few = MIN_PAIRED_SAMPLES - 1
    baseline = [0.5] * too_few
    comparison = [0.6] * too_few
    assert paired_significance_test(baseline, comparison) is None


def test_paired_significance_test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_significance_test([0.5, 0.6], [0.5])


def test_paired_significance_test_detects_consistent_improvement():
    baseline = [0.70, 0.71, 0.69, 0.72, 0.70]
    comparison = [0.90, 0.91, 0.89, 0.92, 0.90]  # consistently higher, same seeds
    result = paired_significance_test(baseline, comparison)
    assert result is not None
    assert result.n_pairs == 5
    assert result.mean_difference > 0
    assert result.significant_at_0_05 is True
    assert result.p_value < 0.05


def test_paired_significance_test_no_difference_is_not_significant():
    # Small noise around a zero true difference (perfectly identical pairs
    # give the t-test a zero-variance denominator, an edge case scipy
    # reports as NaN, not a meaningful "not significant" result).
    baseline = [0.80, 0.81, 0.79, 0.80, 0.80]
    comparison = [0.81, 0.80, 0.80, 0.79, 0.81]
    result = paired_significance_test(baseline, comparison)
    assert result is not None
    assert result.mean_difference == pytest.approx(0.0, abs=0.01)
    assert result.significant_at_0_05 is False
