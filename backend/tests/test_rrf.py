import pytest

from app.store import rrf_fuse


def test_rrf_rewards_documents_ranked_by_both():
    fused = rrf_fuse([["a", "b", "c"], ["c", "a", "d"]], k=60)
    ids = [d for d, _ in fused]
    assert ids[0] == "a"  # ranks 1 and 2
    assert ids[1] == "c"  # ranks 3 and 1
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_score_formula():
    fused = dict(rrf_fuse([["x"], ["y", "x"]], k=60))
    assert fused["x"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["y"] == pytest.approx(1 / 61)


def test_rrf_empty():
    assert rrf_fuse([[], []]) == []
