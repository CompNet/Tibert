from hypothesis.core import given
import hypothesis.strategies as st
from tibert.utils import spans, spans_indexs


@given(lst=st.lists(st.text()), max_len=st.integers(max_value=10))
def test_spans_len_equals_spans_indexs_len(lst: list, max_len: int):
    assert len(spans(lst, max_len)) == len(spans_indexs(lst, max_len))
