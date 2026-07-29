from learning_accelerator.graph.workflow import build_graph


def test_build_graph_compiles_without_interrupt_before(tmp_path):
    graph = build_graph(db_path=str(tmp_path / "test.sqlite"))
    assert graph is not None


def test_build_graph_compiles_with_interrupt_before(tmp_path):
    graph = build_graph(
        db_path=str(tmp_path / "test.sqlite"),
        interrupt_before=["quiz_generator"],
    )
    assert graph is not None
