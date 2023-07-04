from typing import TYPE_CHECKING, Optional
import traceback
from statistics import mean
from more_itertools.recipes import flatten
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import BertTokenizerFast  # type: ignore
from tqdm import tqdm
from tibert import (
    BertForCoreferenceResolution,
    BertCoreferenceResolutionOutput,
    CoreferenceDataset,
    split_coreference_document,
    DataCollatorForSpanClassification,
    score_coref_predictions,
)
from tibert.utils import gpu_memory_usage


def train_coref_model(
    model: BertForCoreferenceResolution,
    dataset: CoreferenceDataset,
    tokenizer: BertTokenizerFast,
    batch_size: int = 1,
    epochs_nb: int = 30,
    sents_per_documents_train: int = 11,
    bert_lr: float = 1e-5,
    task_lr: float = 2e-4,
    _run: Optional["sacred.run.Run"] = None,
) -> BertForCoreferenceResolution:

    if _run:
        from sacred.commands import print_config

        print_config(_run)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CoreferenceDataset(
        dataset.documents[: int(0.9 * len(dataset))],
        dataset.tokenizer,
        dataset.max_span_size,
    )
    train_dataset.documents = list(
        flatten(
            [
                split_coreference_document(doc, sents_per_documents_train)
                for doc in train_dataset.documents
            ]
        )
    )

    test_dataset = CoreferenceDataset(
        dataset.documents[int(0.9 * len(dataset)) :],
        dataset.tokenizer,
        dataset.max_span_size,
    )
    test_dataset.documents = list(
        flatten(
            [
                # HACK: test on full documents
                split_coreference_document(doc, 200)
                for doc in test_dataset.documents
            ]
        )
    )

    data_collator = DataCollatorForSpanClassification(tokenizer, model.max_span_size)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator
    )

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

    best_f1 = 0
    best_model = model

    model = model.to(device)

    for epoch_i in range(epochs_nb):

        model = model.train()

        epoch_losses = []

        data_tqdm = tqdm(train_dataloader)
        for batch in data_tqdm:

            batch = batch.to(device)

            optimizer.zero_grad()

            try:
                out = model(**batch)
            except Exception as e:
                print(e)
                traceback.print_exc()
                continue

            assert not out.loss is None
            out.loss.backward()
            optimizer.step()

            _ = _run and _run.log_scalar("gpu_usage", gpu_memory_usage())

            data_tqdm.set_description(f"loss : {out.loss.item()}")
            epoch_losses.append(out.loss.item())
            if _run:
                _run.log_scalar("loss", out.loss.item())

        if _run:
            _run.log_scalar("epoch_mean_loss", mean(epoch_losses))

        # metrics computation
        model = model.eval()

        with torch.no_grad():

            try:

                preds = []
                losses = []

                for batch in tqdm(test_dataloader):

                    local_batch_size = batch["input_ids"].shape[0]
                    batch = batch.to(device)
                    out: BertCoreferenceResolutionOutput = model(**batch)
                    batch_preds = out.coreference_documents(
                        [
                            [tokenizer.decode(t) for t in batch["input_ids"][i]]
                            for i in range(local_batch_size)
                        ]
                    )
                    preds += batch_preds

                    assert not out.loss is None
                    losses.append(out.loss.item())

                _ = _run and _run.log_scalar("epoch_mean_test_loss", mean(losses))

                refs = [
                    doc.prepared_document(test_dataset.tokenizer, model.max_span_size)[
                        0
                    ]
                    for doc in test_dataset.documents
                ]
                metrics = score_coref_predictions(preds, refs)

                if _run:
                    _run.log_scalar("muc_precision", metrics["MUC"]["precision"])
                    _run.log_scalar("muc_recall", metrics["MUC"]["recall"])
                    _run.log_scalar("muc_f1", metrics["MUC"]["f1"])
                    _run.log_scalar("b3_precision", metrics["B3"]["precision"])
                    _run.log_scalar("b3_recall", metrics["B3"]["recall"])
                    _run.log_scalar("b3_f1", metrics["B3"]["f1"])
                    _run.log_scalar("ceaf_precision", metrics["CEAF"]["precision"])
                    _run.log_scalar("ceaf_recall", metrics["CEAF"]["recall"])
                    _run.log_scalar("ceaf_f1", metrics["CEAF"]["f1"])

                print(metrics)

                # keep the best model
                model_f1 = mean(
                    [metrics["MUC"]["f1"], metrics["B3"]["f1"], metrics["CEAF"]["f1"]]
                )

            except Exception as e:
                print(e)
                traceback.print_exc()
                model_f1 = 0

            if model_f1 > best_f1 or best_f1 == 0:
                best_model = model.to("cpu")
                best_f1 = model_f1

    return best_model
