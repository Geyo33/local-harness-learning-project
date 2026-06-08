from mcp_chatbot.frontend.disconnect import can_save_episode


def test_no_store_never_saves():
    assert can_save_episode(False, 50) is False


def test_store_but_too_few_messages():
    assert can_save_episode(True, 9) is False


def test_store_at_threshold_saves():
    assert can_save_episode(True, 10) is True


def test_store_well_over_threshold_saves():
    assert can_save_episode(True, 50) is True


def test_no_store_and_zero_messages():
    assert can_save_episode(False, 0) is False
