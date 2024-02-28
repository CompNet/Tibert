from typing import List, Tuple
import pytest
from hypothesis import given, assume
import hypothesis.strategies as st
from tibert.bertcoref import Mention
from tibert import score_mention_detection, CoreferenceDocument, score_lea
from tests.strategies import coref_docs
from more_itertools import flatten


@given(docs=st.lists(coref_docs(min_size=1, max_size=32), min_size=1, max_size=3))
def test_mention_score_perfect_when_same_docs(docs: List[CoreferenceDocument]):
    assume(all([len(doc.coref_chains) > 0 for doc in docs]))
    assert score_mention_detection(docs, docs) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    "pred,ref,expected",
    [
        ([["A"]], [["A"]], (1.0, 1.0, 1.0)),
        (
            [["A", "B"], ["C", "D"], ["F", "G", "H", "I"]],
            [["A", "B", "C"], ["D", "E", "F", "G"]],
            (0.333, 0.24, 0.2779),
        ),
    ],
)
def test_lea_canonical_examples(
    pred: List[List[str]], ref: List[List[str]], expected: Tuple[float, float, float]
):
    pred_doc = CoreferenceDocument(
        list(flatten(pred)),
        [[Mention([mention], 0, 0) for mention in chain] for chain in pred],
    )
    ref_doc = CoreferenceDocument(
        list(flatten(ref)),
        [[Mention([mention], 0, 0) for mention in chain] for chain in ref],
    )

    precision, recall, f1 = score_lea([pred_doc], [ref_doc])
    assert precision == pytest.approx(expected[0], rel=1e-2)
    assert recall == pytest.approx(expected[1], rel=1e-2)
    assert f1 == pytest.approx(expected[2], rel=1e-2)
