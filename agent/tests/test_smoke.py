from notebook_agent.app.workflow import build_graph


def test_graph_returns_recommendation():
    graph = build_graph()
    result = graph.invoke({"query_text": "7000 以内，写代码，偶尔游戏，轻一点，不要联想"})
    assert "extracted_constraints" in result
    assert "candidates" in result
