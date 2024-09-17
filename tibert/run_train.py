from typing import Literal, Optional, cast
import os
from torch.optim import optimizer
from transformers import BertTokenizerFast, CamembertTokenizerFast  # type: ignore
from sacred.experiment import Experiment
from sacred.run import Run
from sacred.commands import print_config
from tibert import (
    load_litbank_dataset,
    load_fr_litbank_dataset,
    BertForCoreferenceResolution,
    CamembertForCoreferenceResolution,
    train_coref_model,
    load_train_checkpoint,
    predict_coref,
    score_coref_predictions,
    score_mention_detection,
)
from tibert.bertcoref import CoreferenceDataset, load_democrat_dataset

ex = Experiment()


@ex.config
def config():
    batch_size: int = 1
    epochs_nb: int = 30
    # either "litbank", "fr-litbank" or "democrat"
    dataset_name: str = "litbank"
    dataset_path: str = os.path.expanduser("~/litbank")
    mentions_per_tokens: float = 0.4
    antecedents_nb: int = 350
    max_span_size: int = 10
    mention_scorer_hidden_size: int = 3000
    tokens_per_document: int = 512
    mention_loss_coeff: float = 0.1
    bert_lr: float = 1e-5
    task_lr: float = 2e-4
    dropout: float = 0.3
    segment_size: int = 128
    encoder: str = "bert-base-cased"
    out_model_dir: str = os.path.expanduser("~/tibert/model")
    checkpoint: Optional[str] = None
    example_tracking_path: Optional[str] = None


@ex.main
def main(
    _run: Run,
    batch_size: int,
    epochs_nb: int,
    dataset_name: Literal["litbank", "fr-litbank", "democrat"],
    dataset_path: str,
    mentions_per_tokens: float,
    antecedents_nb: int,
    max_span_size: int,
    mention_scorer_hidden_size: int,
    tokens_per_document: int,
    mention_loss_coeff: float,
    bert_lr: float,
    task_lr: float,
    dropout: float,
    segment_size: int,
    encoder: str,
    out_model_dir: str,
    checkpoint: Optional[str],
    example_tracking_path: Optional[str],
):
    print_config(_run)

    dataset_configs = {
        "litbank": {
            "model_class": BertForCoreferenceResolution,
            "tokenizer_class": BertTokenizerFast,
            "loading_function": load_litbank_dataset,
        },
        "fr-litbank": {
            "model_class": CamembertForCoreferenceResolution,
            "tokenizer_class": CamembertTokenizerFast,
            "loading_function": load_fr_litbank_dataset,
        },
        "democrat": {
            "model_class": CamembertForCoreferenceResolution,
            "tokenizer_class": CamembertTokenizerFast,
            "loading_function": load_democrat_dataset,
        },
    }

    if not dataset_name in dataset_configs:
        raise ValueError(f"unknown dataset: {dataset_name}")

    config = dataset_configs[dataset_name]

    if not checkpoint is None:
        model, optimizer = load_train_checkpoint(checkpoint, config["model_class"])
    else:
        model = config["model_class"].from_pretrained(
            encoder,
            mentions_per_tokens=mentions_per_tokens,
            antecedents_nb=antecedents_nb,
            max_span_size=max_span_size,
            segment_size=segment_size,
            mention_scorer_hidden_size=mention_scorer_hidden_size,
            mention_scorer_dropout=dropout,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            mention_loss_coeff=mention_loss_coeff,
        )
        optimizer = None

    tokenizer = config["tokenizer_class"].from_pretrained(encoder)

    dataset: CoreferenceDataset = config["loading_function"](
        dataset_path, tokenizer, max_span_size
    )
    train_dataset, valid_test_dataset = dataset.splitted(0.8)
    valid_dataset, test_dataset = valid_test_dataset.splitted(0.5)
    train_dataset.limit_doc_size_tokens_(tokens_per_document)
    valid_dataset.limit_doc_size_tokens_(tokens_per_document)
    test_dataset.limit_doc_size_tokens_(tokens_per_document)

    train_coref_model(
        model,
        train_dataset,
        valid_dataset,
        tokenizer,
        batch_size=batch_size,
        epochs_nb=epochs_nb,
        bert_lr=bert_lr,
        task_lr=task_lr,
        model_save_dir=out_model_dir,
        device_str="auto",
        _run=_run,
        optimizer=optimizer,
        example_tracking_path=example_tracking_path,
    )

    annotated_docs = predict_coref(
        [doc.tokens for doc in test_dataset.documents],
        model,
        tokenizer,
        hierarchical_merging=False,
        quiet=True,
        device_str="cuda",
        batch_size=batch_size,
    )
    assert isinstance(annotated_docs, list)

    mention_pre, mention_rec, mention_f1 = score_mention_detection(
        annotated_docs, test_dataset.documents
    )
    for metric_key, score in [
        ("precision", mention_pre),
        ("recall", mention_rec),
        ("f1", mention_f1),
    ]:
        print(f"mention.{metric_key}={score}")
        _run.log_scalar(f"mention.{metric_key}", score)

    metrics = score_coref_predictions(annotated_docs, test_dataset.documents)
    print(metrics)
    for key, score_dict in metrics.items():
        for metric_key, score in score_dict.items():
            print(f"{key}.{metric_key}={score}")
            _run.log_scalar(f"test.{key}_{metric_key}", score)


if __name__ == "__main__":
    ex.run_commandline()
