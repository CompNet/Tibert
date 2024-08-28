from __future__ import annotations
from typing import TYPE_CHECKING, Generator, Literal, List, Optional, Union, cast, Tuple
from statistics import mean
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import PreTrainedTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding
from tqdm import tqdm
from more_itertools import flatten
from sacremoses import MosesTokenizer
from tibert import (
    CoreferenceDataset,
    CoreferenceDocument,
    DataCollatorForSpanClassification,
)
from tibert.bertcoref import Mention
from tibert.utils import spans_indexs

if TYPE_CHECKING:
    from tibert.bertcoref import (
        CoreferenceDocument,
        BertCoreferenceResolutionOutput,
        BertForCoreferenceResolution,
    )


def merge_coref_outputs(
    outputs: List[CoreferenceDocument],
    hidden_states: List[torch.FloatTensor],
    model: BertForCoreferenceResolution,
    device_str: Literal["cpu", "cuda", "auto"] = "auto",
) -> Optional[CoreferenceDocument]:
    """Merge coreference clusters as in Gupta et al 2024

    :param outputs: output coreference documents
    :param hidden_states: the hidden state for tokens of each
        coreference document.  Each tensor should be of shape
        (len(doc.tokens), hidden_size)
    :param model: coreference model, used to compute scores between
        pairs of clusters

    :return: None if outputs is empty, a single merged document
             otherwise
    """
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    assert len(outputs) == len(hidden_states)

    if len(outputs) == 0:
        return None
    if len(outputs) == 1:
        return outputs[0]

    middle = int(len(outputs) / 2)
    merged_left = merge_coref_outputs(
        outputs[:middle], hidden_states[:middle], model, device_str=device_str
    )
    merged_right = merge_coref_outputs(
        outputs[middle:], hidden_states[middle:], model, device_str=device_str
    )
    assert merged_left and merged_right

    if len(merged_left.coref_chains) == 0 or len(merged_right.coref_chains) == 0:
        return CoreferenceDocument.concatenated([merged_left, merged_right])

    with torch.no_grad():
        b = 1
        a = len(merged_left.coref_chains)
        m = len(merged_right.coref_chains)
        h = model.config.hidden_size

        def tensorize_chains(
            chains: List[List[Mention]], hidden_states: torch.Tensor
        ) -> torch.Tensor:
            """
            :param chains: c chains
            :param hidden_states: ``(s, h)``
            :return: a tensor of shape ``(c, 2 * h)``
            """
            # (c, 2 * h)
            return torch.stack(
                [
                    # (2 * h)
                    torch.mean(
                        torch.stack(
                            [
                                torch.cat(
                                    [
                                        hidden_states[mention.start_idx],
                                        hidden_states[mention.end_idx - 1],
                                    ]
                                )
                                for mention in chain
                            ]
                        ),
                        dim=0,
                    )
                    for chain in chains
                ]
            )

        lhidden_states = torch.cat(tuple(hidden_states[:middle]))
        left_mentions_repr = tensorize_chains(
            merged_left.coref_chains, lhidden_states
        ).unsqueeze(0)
        left_mentions_repr = left_mentions_repr.reshape(b, a, 2, h)

        rhidden_states = torch.cat(tuple(hidden_states[middle:]))
        right_mentions_repr = tensorize_chains(
            merged_right.coref_chains, rhidden_states
        ).unsqueeze(0)
        right_mentions_repr = right_mentions_repr.reshape(b, m, 2, h)

        left_mentions_repr_repeated = left_mentions_repr.repeat(1, m, 1, 1, 1)
        assert left_mentions_repr_repeated.shape == (b, m, a, 2, h)

        span_bounds_combination = model.mention_pairs_repr(
            right_mentions_repr, left_mentions_repr_repeated
        )
        assert span_bounds_combination.shape == (b, m * a, 4 * h)

        # compute distance feature
        seq_size = len(merged_left.tokens) + len(merged_right.tokens)
        spans_idx = spans_indexs(list(range(seq_size)), model.config.max_span_size)

        left_mentions_idx = [
            max(m.end_idx for m in chain) for chain in merged_left.coref_chains
        ]
        left_mentions_idx = torch.tensor(left_mentions_idx).unsqueeze(0)
        assert left_mentions_idx.shape == (b, a)

        roffset = len(merged_left.tokens)
        right_mentions_idx = [
            min(m.start_idx + roffset for m in chain)
            for chain in merged_right.coref_chains
        ]
        right_mentions_idx = torch.tensor(right_mentions_idx).unsqueeze(0).to(device)
        assert right_mentions_idx.shape == (b, m)

        top_antecedents_index = (
            torch.stack([left_mentions_idx[0] for _ in range(m)])
            .unsqueeze(0)
            .to(device)
        )
        assert top_antecedents_index.shape == (b, m, a)

        spans_nb = len(spans_idx)
        dist_ft = model.distance_feature(
            top_antecedents_index, right_mentions_idx, spans_nb, seq_size
        )
        dist_ft = torch.flatten(dist_ft, start_dim=1, end_dim=2)

        mention_pairs_repr = torch.cat((span_bounds_combination, dist_ft), dim=2)
        compat_score = model.mention_compatibility_score(
            torch.flatten(mention_pairs_repr, start_dim=0, end_dim=1)
        )
        assert compat_score.shape == (b * m * a,)
        compat_score = compat_score.reshape(b, m, a)

        mention_score = (
            torch.tensor(
                [
                    [
                        mean([mention.mention_score for mention in rchain])  # type: ignore
                        + mean([mention.mention_score for mention in lchain])  # type: ignore
                        for lchain in merged_left.coref_chains
                    ]
                    for rchain in merged_right.coref_chains
                ]
            )
            .unsqueeze(0)
            .to(device)
        )
        assert mention_score.shape == (b, m, a)

        final_score = compat_score + mention_score

        # At MOST each cluster must correspond to another
        # cluster. this looks like bipartite matching. For now we
        # ignore that and do it greedily
        left_offset = len(merged_left.tokens)
        merged_right_chains = set()
        new_chains = []
        for left_chain_i, left_chain in enumerate(merged_left.coref_chains):
            scores = final_score[0, :, left_chain_i]
            scores[torch.tensor(list(merged_right_chains), dtype=torch.long)] = float(
                "-Inf"
            )
            best_score = torch.max(scores, dim=0)
            if best_score.values.item() < 0.0:
                new_chains.append(left_chain)
                continue
            r_chain_i = best_score.indices.item()
            right_chain = merged_right.coref_chains[r_chain_i]
            new_chains.append(
                left_chain + [m.shifted(left_offset) for m in right_chain]
            )
            merged_right_chains.add(r_chain_i)
        for r_chain_i, r_chain in enumerate(merged_right.coref_chains):
            if not r_chain_i in merged_right_chains:
                new_chains.append([m.shifted(left_offset) for m in r_chain])

    return CoreferenceDocument(merged_left.tokens + merged_right.tokens, new_chains)


def _stream_predict_coref_raw(
    documents: List[Union[str, List[str]]],
    model: BertForCoreferenceResolution,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int = 1,
    quiet: bool = False,
    device_str: Literal["cpu", "cuda", "auto"] = "auto",
    lang: str = "en",
    return_hidden_state: bool = False,
) -> Generator[
    Tuple[List[CoreferenceDocument], BertCoreferenceResolutionOutput], None, None
]:
    """Low level inference interface."""

    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if len(documents) == 0:
        return

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
    dataset.set_test_()
    data_collator = DataCollatorForSpanClassification(
        tokenizer, model.config.max_span_size, device_str
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False
    )

    model = model.eval()  # type: ignore
    model = model.to(device)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, disable=quiet)):
            local_batch_size = batch["input_ids"].shape[0]

            start_idx = batch_size * i
            end_idx = batch_size * i + local_batch_size
            batch_docs = dataset.documents[start_idx:end_idx]

            batch = batch.to(device)
            out: BertCoreferenceResolutionOutput = model(
                **batch, return_hidden_state=return_hidden_state
            )

            out_docs = out.coreference_documents([doc.tokens for doc in batch_docs])

            yield out_docs, out


def stream_predict_coref(
    documents: List[Union[str, List[str]]],
    model: BertForCoreferenceResolution,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int = 1,
    quiet: bool = False,
    device_str: Literal["cpu", "cuda", "auto"] = "auto",
    lang: str = "en",
) -> Generator[CoreferenceDocument, None, None]:
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
    for out_docs, _ in _stream_predict_coref_raw(
        documents, model, tokenizer, batch_size, quiet, device_str, lang
    ):
        for out_doc in out_docs:
            yield out_doc


def predict_coref(
    documents: List[Union[str, List[str]]],
    model: BertForCoreferenceResolution,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int = 1,
    quiet: bool = False,
    device_str: Literal["cpu", "cuda", "auto"] = "auto",
    lang: str = "en",
    hierarchical_merging: bool = False,
) -> Union[List[CoreferenceDocument], Optional[CoreferenceDocument]]:
    """Predict coreference chains for a list of documents.

    :param documents: A list of documents, tokenized or not.  If
        documents are not tokenized, MosesTokenizer will tokenize them
        automatically.
    :param tokenizer:
    :param batch_size:
    :param quiet: If ``True``, will report progress using ``tqdm``.
    :param lang: lang for ``MosesTokenizer``
    :param hierarchical_merging: if ``True``, will perform
        hierarchical cluster merging as in Gupta et al 2024.  This
        assumes that the input documents are contiguous.

    :return: a list of ``CoreferenceDocument``, with annotated
             coreference chains.
    """
    if hierarchical_merging:
        docs = []
        hidden_states = []

        if len(documents) == 0:
            return None

        for out_docs, out in _stream_predict_coref_raw(
            documents,
            model,
            tokenizer,
            batch_size,
            quiet,
            device_str,
            lang,
            return_hidden_state=True,
        ):
            docs += out_docs

            assert not out.hidden_states is None
            hidden_states += [h for h in out.hidden_states]

        merged_doc = merge_coref_outputs(docs, hidden_states, model, device_str)
        return merged_doc

    return list(
        stream_predict_coref(
            documents, model, tokenizer, batch_size, quiet, device_str, lang
        )
    )


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
    assert not annotated_docs is None
    assert isinstance(annotated_docs, list)
    return annotated_docs[0]
