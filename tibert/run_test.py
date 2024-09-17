from typing import Literal, Optional
import os
import functools as ft
from transformers import BertTokenizerFast, CamembertTokenizerFast  # type: ignore
from tqdm import tqdm
from sacred.experiment import Experiment
from sacred.run import Run
from sacred.commands import print_config
from tibert import predict
from tibert.bertcoref import (
    CoreferenceDataset,
    CoreferenceDocument,
    load_litbank_dataset,
    load_democrat_dataset,
    load_fr_litbank_dataset,
    BertForCoreferenceResolution,
    CamembertForCoreferenceResolution,
)
from tibert.score import score_coref_predictions, score_mention_detection
from tibert.predict import predict_coref
from tibert.utils import split_coreference_document_tokens

ex = Experiment()


@ex.config
def config():
    batch_size: int = 1
    # either "litbank", "fr-litbank" or "democrat"
    dataset_name: str = "litbank"
    dataset_path: str = os.path.expanduser("~/litbank")
    max_span_size: int = 10
    # in tokens
    limit_doc_size: Optional[int] = None
    hierarchical_merging: bool = False
    device_str: str = "auto"
    model_path: str


@ex.main
def main(
    _run: Run,
    batch_size: int,
    dataset_name: Literal["litbank", "fr-litbank", "democrat"],
    dataset_path: str,
    max_span_size: int,
    limit_doc_size: Optional[int],
    hierarchical_merging: bool,
    device_str: Literal["cuda", "cpu", "auto"],
    model_path: str,
):
    print_config(_run)

    dataset_configs = {
        "litbank": {
            "model_class": BertForCoreferenceResolution,
            "tokenizer_function": ft.partial(
                BertTokenizerFast.from_pretrained, "bert-base-cased"
            ),
            "loading_function": load_litbank_dataset,
        },
        "fr-litbank": {
            "model_class": CamembertForCoreferenceResolution,
            "tokenizer_function": ft.partial(
                CamembertTokenizerFast.from_pretrained, "camembert-base"
            ),
            "loading_function": load_fr_litbank_dataset,
        },
        "democrat": {
            "model_class": CamembertForCoreferenceResolution,
            "tokenizer_function": ft.partial(
                CamembertTokenizerFast.from_pretrained, "camembert-base"
            ),
            "loading_function": load_democrat_dataset,
        },
    }

    model = dataset_configs[dataset_name]["model_class"].from_pretrained(model_path)
    tokenizer = dataset_configs[dataset_name]["tokenizer_function"]()

    dataset: CoreferenceDataset = dataset_configs[dataset_name]["loading_function"](
        dataset_path, tokenizer, max_span_size
    )
    _, test_dataset = dataset.splitted(0.9)

    if limit_doc_size is None:
        all_annotated_docs = predict_coref(
            [doc.tokens for doc in test_dataset.documents],
            model,
            tokenizer,
            device_str=device_str,
            batch_size=batch_size,
        )
        assert isinstance(all_annotated_docs, list)
    else:
        all_annotated_docs = []
        for document in tqdm(test_dataset.documents):
            doc_dataset = CoreferenceDataset(
                split_coreference_document_tokens(document, limit_doc_size),
                tokenizer,
                max_span_size,
            )
            if hierarchical_merging:
                annotated_doc = predict_coref(
                    [doc.tokens for doc in doc_dataset.documents],
                    model,
                    tokenizer,
                    hierarchical_merging=True,
                    quiet=True,
                    device_str=device_str,
                    batch_size=batch_size,
                )
            else:
                annotated_docs = predict_coref(
                    [doc.tokens for doc in doc_dataset.documents],
                    model,
                    tokenizer,
                    quiet=True,
                    device_str=device_str,
                    batch_size=batch_size,
                )
                assert isinstance(annotated_docs, list)
                annotated_doc = CoreferenceDocument.concatenated(annotated_docs)
            all_annotated_docs.append(annotated_doc)

    mention_pre, mention_rec, mention_f1 = score_mention_detection(
        all_annotated_docs, test_dataset.documents
    )
    for metric_key, score in [
        ("precision", mention_pre),
        ("recall", mention_rec),
        ("f1", mention_f1),
    ]:
        print(f"mention.{metric_key}={score}")
        _run.log_scalar(f"mention.{metric_key}", score)

    scores = score_coref_predictions(all_annotated_docs, test_dataset.documents)
    for key, score_dict in scores.items():
        for metric_key, score in score_dict.items():
            print(f"{key}.{metric_key}={score}")
            _run.log_scalar(f"{key}.{metric_key}", score)
    conll_f1 = (scores["MUC"]["f1"] + scores["B3"]["f1"] + scores["CEAF"]["f1"]) / 3
    print(f"CONLL.f1={conll_f1}")


if __name__ == "__main__":
    ex.run_commandline()
