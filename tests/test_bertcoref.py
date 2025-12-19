from hypothesis import given
from tibert.bertcoref import CoreferenceDocument
from tests.strategies import coref_docs


@given(doc=coref_docs())
def test_mention_labels_number_is_correct(doc: CoreferenceDocument):
    """
    The number of mentions labeled as such by
    :meth:`CoreferenceDocument.mention_labels` should be equal to the
    number of mentions of the document (that have a length lower then
    `max_span_size`)
    """
    max_span_size = min(4, len(doc))
    mentions = [
        mention
        for chain in doc.coref_chains
        for mention in chain
        if len(mention.tokens) <= max_span_size
    ]
    labels = doc.mention_labels(max_span_size)
    assert sum(labels) == len(mentions)
