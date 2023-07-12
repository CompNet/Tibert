from __future__ import annotations
from typing import TYPE_CHECKING, Literal, List, Union, cast
from transformers import PreTrainedTokenizerFast
import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from sacremoses import MosesTokenizer

if TYPE_CHECKING:
    from tibert.bertcoref import (
        CoreferenceDocument,
        BertCoreferenceResolutionOutput,
        BertForCoreferenceResolution,
    )


def predict_coref(
    documents: List[Union[str, List[str]]],
    model: BertForCoreferenceResolution,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int = 1,
    quiet: bool = False,
    device_str: Literal["cpu", "cuda", "auto"] = "auto",
    lang: str = "en",
) -> List[CoreferenceDocument]:
    """Predict coreference chains for a list of documents.

    :param documents: A list of documents, tokenized or not.  If
        documents are not tokenized, MosesTokenizer will tokenize them
        automatically.
    :param tokenizer:
    :param batch_size:
    :param quiet: If ``True``, will report progress using ``tqdm``.
    :param lang: lang for ``MosesTokenizer``

    :return: a list of ``CoreferenceDocument``, with annotated
             coreference chains.
    """
    from tibert import (
        CoreferenceDataset,
        CoreferenceDocument,
        DataCollatorForSpanClassification,
    )

    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if len(documents) == 0:
        return []

    # Tokenized input sentence if needed
    if isinstance(documents[0], str):
        m_tokenizer = MosesTokenizer(lang=lang)
        tokenized_documents = [
            m_tokenizer.tokenize(text, escape=False) for text in documents
        ]
    else:
        tokenized_documents = documents
    tokenized_documents = cast(List[List[str]], tokenized_documents)

    dataset = CoreferenceDataset(
        [CoreferenceDocument(doc, []) for doc in tokenized_documents],
        tokenizer,
        model.config.max_span_size,
    )
    data_collator = DataCollatorForSpanClassification(tokenizer, model.config.max_span_size)  # type: ignore
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False
    )

    model = model.eval()  # type: ignore
    model = model.to(device)

    preds = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, disable=quiet)):
            local_batch_size = batch["input_ids"].shape[0]

            start_idx = batch_size * i
            end_idx = batch_size * i + local_batch_size
            batch_docs = dataset.documents[start_idx:end_idx]

            batch = batch.to(device)
            out: BertCoreferenceResolutionOutput = model(**batch)
            out_docs = out.coreference_documents(
                [
                    [tokenizer.decode(t) for t in input_ids]  # type: ignore
                    for input_ids in batch["input_ids"]
                ]
            )
            out_docs = [
                out_doc.from_wpieced_to_tokenized(original_doc.tokens, batch, batch_i)
                for batch_i, (original_doc, out_doc) in enumerate(
                    zip(batch_docs, out_docs)
                )
            ]
            preds += out_docs

    return preds


def predict_coref_simple(
    text: Union[str, List[str]],
    model,
    tokenizer,
    device_str: Literal["cpu", "cuda", "auto"] = "auto",
    lang: str = "en",
) -> CoreferenceDocument:
    annotated_docs = predict_coref(
        [text],
        model,
        tokenizer,
        batch_size=1,
        device_str=device_str,
        quiet=True,
        lang=lang,
    )
    return annotated_docs[0]
