from typing import Optional, Union, Literal
import traceback, copy
from statistics import mean
from more_itertools.recipes import flatten
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import BertTokenizerFast, CamembertTokenizerFast  # type: ignore
from tqdm import tqdm
from tibert import (
    BertForCoreferenceResolution,
    CamembertForCoreferenceResolution,
    CoreferenceDataset,
    split_coreference_document,
    DataCollatorForSpanClassification,
    score_coref_predictions,
    score_mention_detection,
)
from tibert.predict import predict_coref
from tibert.utils import gpu_memory_usage


def train_coref_model(
    model: Union[BertForCoreferenceResolution, CamembertForCoreferenceResolution],
    dataset: CoreferenceDataset,
    tokenizer: Union[BertTokenizerFast, CamembertTokenizerFast],
    batch_size: int = 1,
    epochs_nb: int = 30,
    sents_per_documents_train: int = 11,
    bert_lr: float = 1e-5,
    task_lr: float = 2e-4,
    model_save_path: Optional[str] = None,
    device_str: Literal["cpu", "cuda", "auto"] = "auto",
    _run: Optional["sacred.run.Run"] = None,
) -> BertForCoreferenceResolution:
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

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
                split_coreference_document(doc, 11)
                for doc in test_dataset.documents
            ]
        )
    )

    data_collator = DataCollatorForSpanClassification(
        tokenizer, model.config.max_span_size
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator
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

    for _ in range(epochs_nb):
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

        # Metrics Computation
        # -------------------
        preds = predict_coref(
            [doc.tokens for doc in test_dataset.documents],
            model,
            tokenizer,
            batch_size=batch_size,
            device_str=device_str,
        )
        metrics = score_coref_predictions(preds, test_dataset.documents)

        conll_f1 = mean(
            [metrics["MUC"]["f1"], metrics["B3"]["f1"], metrics["CEAF"]["f1"]]
        )
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
            _run.log_scalar("conll_f1", conll_f1)
        print(metrics)

        m_precision, m_recall, m_f1 = score_mention_detection(
            preds, test_dataset.documents
        )
        if _run:
            _run.log_scalar("mention_detection_precision", m_precision)
            _run.log_scalar("mention_detection_recall", m_recall)
            _run.log_scalar("mention_detection_f1", m_f1)
        print(
            f"mention detection metrics: (precision: {m_precision}, recall: {m_recall}, f1: {m_f1})"
        )

        if conll_f1 > best_f1 or best_f1 == 0:
            best_model = copy.deepcopy(model).to("cpu")
            if not model_save_path is None:
                best_model.save_pretrained(model_save_path)
            best_f1 = conll_f1

    return best_model
