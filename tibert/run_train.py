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
)

ex = Experiment()


@ex.config
def config():
    batch_size: int = 1
    epochs_nb: int = 30
    # either "litbank" or "fr-litbank"
    dataset_name: str = "litbank"
    dataset_path: str = os.path.expanduser("~/litbank")
    mentions_per_tokens: float = 0.4
    antecedents_nb: int = 350
    max_span_size: int = 10
    mention_scorer_hidden_size: int = 3000
    sents_per_documents_train: int = 11
    mention_loss_coeff: float = 0.1
    bert_lr: float = 1e-5
    task_lr: float = 2e-4
    dropout: float = 0.3
    segment_size: int = 128
    encoder: str = "bert-base-cased"
    out_model_dir: str = os.path.expanduser("~/tibert/model")
    checkpoint: Optional[str] = None


@ex.main
def main(
    _run: Run,
    batch_size: int,
    epochs_nb: int,
    dataset_name: Literal["litbank", "fr-litbank"],
    dataset_path: str,
    mentions_per_tokens: float,
    antecedents_nb: int,
    max_span_size: int,
    mention_scorer_hidden_size: int,
    sents_per_documents_train: int,
    mention_loss_coeff: float,
    bert_lr: float,
    task_lr: float,
    dropout: float,
    segment_size: int,
    encoder: str,
    out_model_dir: str,
    checkpoint: Optional[str],
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

    dataset = config["loading_function"](dataset_path, tokenizer, max_span_size)

    train_coref_model(
        model,
        dataset,
        tokenizer,
        batch_size=batch_size,
        epochs_nb=epochs_nb,
        sents_per_documents_train=sents_per_documents_train,
        bert_lr=bert_lr,
        task_lr=task_lr,
        model_save_dir=out_model_dir,
        device_str="auto",
        _run=_run,
        optimizer=optimizer,
    )


if __name__ == "__main__":
    ex.run_commandline()
