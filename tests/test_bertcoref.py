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
@given(doc=coref_docs(min_size=5, max_size=10))
def test_doc_is_reconstructed(
    doc: CoreferenceDocument, bert_tokenizer: BertTokenizerFast
):
    max_span_size = min(4, len(doc))
    prep_doc, batch = doc.prepared_document(bert_tokenizer, max_span_size)
    print(prep_doc)
    collator = DataCollatorForSpanClassification(bert_tokenizer, max_span_size)
    batch = collator([batch])
    reconstructed_doc = prep_doc.from_wpieced_to_tokenized(doc.tokens, batch, 0)

    assert doc.tokens == reconstructed_doc.tokens
    assert doc.coref_chains == reconstructed_doc.coref_chains
