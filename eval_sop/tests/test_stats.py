import pytest

from eval_sop import stats


def test_mcnemar_exact_counts_and_p():
    a = {i: 0 for i in range(10)}
    b = {i: 1 if i < 8 else 0 for i in range(10)}
    r = stats.mcnemar_exact(a, b)
    assert (r["b_right_a_wrong"], r["a_right_b_wrong"]) == (8, 0)
    assert r["p_two_sided_exact"] == pytest.approx(2 / 2**8)  # 0.0078125


def test_mcnemar_no_discordant_pairs():
    assert stats.mcnemar_exact({1: 1}, {1: 1})["p_two_sided_exact"] == 1.0
