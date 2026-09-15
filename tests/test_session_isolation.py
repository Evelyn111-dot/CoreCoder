
from api.server import get_agent


def test_same_user_same_session():
    """同一个用户访问同一个 session，应得到同一个 Agent。"""

    agent1, lock1 = get_agent(
        "user_001",
        "session_001",
    )

    agent2, lock2 = get_agent(
        "user_001",
        "session_001",
    )

    assert agent1 is agent2
    assert lock1 is lock2


def test_same_user_different_session():
    """同一个用户的不同 session，应得到不同 Agent。"""

    agent1, lock1 = get_agent(
        "user_001",
        "session_001",
    )

    agent2, lock2 = get_agent(
        "user_001",
        "session_002",
    )

    assert agent1 is not agent2
    assert lock1 is not lock2


def test_different_user_same_session():
    """不同用户即使使用相同 session_id，也应得到不同 Agent。"""

    agent1, lock1 = get_agent(
        "user_001",
        "session_001",
    )

    agent2, lock2 = get_agent(
        "user_002",
        "session_001",
    )

    assert agent1 is not agent2
    assert lock1 is not lock2

