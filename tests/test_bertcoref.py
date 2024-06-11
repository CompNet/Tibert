from hypothesis import given, settings, HealthCheck
from transformers import BertTokenizerFast
from pytest import fixture
from tibert.bertcoref import CoreferenceDocument, DataCollatorForSpanClassification
from tests.strategies import coref_docs


@fixture
def bert_tokenizer() -> BertTokenizerFast:
    return BertTokenizerFast.from_pretrained("bert-base-cased")


# we suppress the `function_scoped_fixture` health check since we want
# to execute the `bert_tokenizer` fixture only once.
@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(doc=coref_docs(min_size=5, max_size=10, max_span_size=4))
def test_doc_is_reconstructed(
    doc: CoreferenceDocument, bert_tokenizer: BertTokenizerFast
):
    max_span_size = min(4, len(doc))
    prep_doc, batch = doc.prepared_document(bert_tokenizer, max_span_size)
    print(prep_doc)
    collator = DataCollatorForSpanClassification(bert_tokenizer, max_span_size)
    batch = collator([batch])
    seq_size = batch["input_ids"].shape[1]
    wp_to_token = [batch.token_to_word(0, token_index=i) for i in range(seq_size)]
    reconstructed_doc = prep_doc.from_wpieced_to_tokenized(doc.tokens, wp_to_token)

    assert doc.tokens == reconstructed_doc.tokens
    assert doc.coref_chains == reconstructed_doc.coref_chains


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
