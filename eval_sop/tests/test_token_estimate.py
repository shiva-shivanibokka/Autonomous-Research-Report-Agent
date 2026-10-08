from eval_sop import budget


def test_cjk_input_is_not_undercounted():
    # 10,000 CJK characters are roughly 10k+ tokens; chars/2.5 alone gave 4,050.
    assert budget.worst_input_tokens([{"content": "漢字" * 5000}]) >= 10_000


def test_english_estimate_unchanged():
    assert budget.worst_input_tokens([{"content": "a" * 2500}]) == 1000 + 50


def test_list_content_blocks_are_counted():
    """attack3 #9: content given as a list of blocks used to crash the estimate."""
    w = budget.worst_input_tokens([{"role": "user", "content": [{"type": "text", "text": "a" * 1000}]}])
    assert w == 400 + 50
