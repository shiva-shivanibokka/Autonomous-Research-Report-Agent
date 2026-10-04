from eval_sop import budget


def test_cjk_input_is_not_undercounted():
    # 10,000 CJK characters are roughly 10k+ tokens; chars/2.5 alone gave 4,050.
    assert budget.worst_input_tokens([{"content": "漢字" * 5000}]) >= 10_000


def test_english_estimate_unchanged():
    assert budget.worst_input_tokens([{"content": "a" * 2500}]) == 1000 + 50
