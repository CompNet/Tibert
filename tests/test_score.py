from typing import List
from hypothesis import given, assume
import hypothesis.strategies as st
from tibert import score_mention_detection, CoreferenceDocument
from tests.strategies import coref_docs


@given(docs=st.lists(coref_docs(min_size=1, max_size=32), min_size=1, max_size=3))
def test_mention_score_perfect_when_same_docs(docs: List[CoreferenceDocument]):
    assume(all([len(doc.coref_chains) > 0 for doc in docs]))
    assert score_mention_detection(docs, docs) == (1.0, 1.0, 1.0)
