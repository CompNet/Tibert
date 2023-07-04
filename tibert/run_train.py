from typing import cast
import os
from transformers import BertTokenizerFast  # type: ignore
from sacred.experiment import Experiment
from sacred.run import Run
from sacred.commands import print_config
from tibert import (
    load_litbank_dataset,
    BertForCoreferenceResolution,
    train_coref_model,
)

ex = Experiment()


@ex.config
def config():
    batch_size: int = 1
    epochs_nb: int = 30
    # only supports litbank for now
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
    out_model_path: str = os.path.expanduser("~/tibert/model")


@ex.main
def main(
    _run: Run,
    batch_size: int,
    epochs_nb: int,
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
    out_model_path: str,
):
    print_config(_run)

    model = BertForCoreferenceResolution.from_pretrained(
        encoder,
        mentions_per_tokens,
        antecedents_nb,
        max_span_size,
        segment_size=segment_size,
        mention_scorer_hidden_size=mention_scorer_hidden_size,
        mention_scorer_dropout=dropout,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        mention_loss_coeff=mention_loss_coeff,
    )

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")  # type: ignore
    tokenizer = cast(BertTokenizerFast, tokenizer)

    dataset = load_litbank_dataset(dataset_path, tokenizer, max_span_size)

    model = train_coref_model(
        model,
        dataset,
        tokenizer,
        batch_size,
        epochs_nb,
        sents_per_documents_train,
        bert_lr,
        task_lr,
        _run,
    )

    model.save_pretrained(out_model_path)


if __name__ == "__main__":

    ex.run_commandline()
