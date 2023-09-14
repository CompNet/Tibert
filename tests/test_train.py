import tempfile
import torch
from torch.optim import optimizer
from transformers import BertTokenizerFast
from tibert.bertcoref import BertForCoreferenceResolutionConfig
from tibert.train import _save_train_checkpoint, load_train_checkpoint
from tibert import BertForCoreferenceResolution, predict_coref_simple


def test_save_load_checkpoint():
    model = BertForCoreferenceResolution(BertForCoreferenceResolutionConfig())
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    bert_lr = 1e-5
    task_lr = 2e-4
    optimizer = torch.optim.AdamW(
        [
            {"params": model.bert_parameters(), "lr": bert_lr},
            {
                "params": model.task_parameters(),
                "lr": task_lr,
            },
        ],
        lr=task_lr,
    )

    text = "Sli did not want the earpods. He didn't like them."
    before_pred = predict_coref_simple(text, model, tokenizer)

    with tempfile.TemporaryDirectory() as d:
        checkpoint_f = f"{d}/checkpoint.pth"
        _save_train_checkpoint(checkpoint_f, model, 1, optimizer, bert_lr, task_lr)
        model, new_optimizer = load_train_checkpoint(
            checkpoint_f, BertForCoreferenceResolution
        )

    assert new_optimizer.state_dict() == optimizer.state_dict()

    after_pred = predict_coref_simple(text, model, tokenizer)
    assert before_pred == after_pred
