import os
import subprocess
import sys

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


def test_importing_workflow_does_not_create_checkpoint_db(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "CHECKPOINT_DB_PATH"}
    result = subprocess.run(
        [sys.executable, "-c", "import learning_accelerator.graph.workflow"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / ".data").exists()


def test_calling_get_default_graph_creates_checkpoint_db(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "CHECKPOINT_DB_PATH"}
    script = (
        "import learning_accelerator.graph.workflow as w\n"
        "g1 = w.get_default_graph()\n"
        "g2 = w.get_default_graph()\n"
        "assert g1 is g2, 'get_default_graph() should be memoized'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / ".data" / "checkpoints.sqlite").exists()
