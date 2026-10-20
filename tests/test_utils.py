from hypothesis.core import given
import hypothesis.strategies as st
from tibert.utils import spans, spans_indexs


@given(lst=st.lists(st.text()), max_len=st.integers(max_value=10))
def test_spans_len_equals_spans_indexs_len(lst: list, max_len: int):
    assert len(spans(lst, max_len)) == len(spans_indexs(lst, max_len))


@given(lst=st.lists(st.text()), max_len=st.integers(max_value=10))
def test_longer_sequences_only_add_spans_at_end(lst: list, max_len: int):
    spans_base_seq = spans(lst, max_len)
    spans_long_seq = spans(lst + ["a"], max_len)
    assert spans_base_seq == spans_long_seq[: len(spans_base_seq)]
