from __future__ import annotations
from typing import TYPE_CHECKING, Generator, Literal, List, Optional, Union, cast, Tuple
import itertools as it
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
    assert len(outputs) == len(hidden_states)

    if len(outputs) == 0:
        return None
    if len(outputs) == 1:
        return outputs[0]

    middle = int(len(outputs) / 2)
    merged_left = merge_coref_outputs(outputs[:middle], hidden_states[:middle], model)
    merged_right = merge_coref_outputs(outputs[middle:], hidden_states[middle:], model)
    assert merged_left and merged_right

    with torch.no_grad():

        b = 1
        a = len(merged_left.coref_chains)
        m = len(merged_right.coref_chains)
        h = model.config.hidden_size

        lhidden_states = torch.cat(tuple(hidden_states[:middle]))
        left_mentions_repr = merged_left.mapmentions(
            lambda m: torch.cat(
                [lhidden_states[m.start_idx], lhidden_states[m.end_idx - 1]]
            )
        )
        left_chains_repr = torch.stack(
            [torch.mean(torch.stack(chain), dim=0) for chain in left_mentions_repr]
        ).unsqueeze(0)
        assert left_chains_repr.shape == (b, a, 2 * h)

        rhidden_states = torch.cat(tuple(hidden_states[middle:]))
        right_mentions_repr = merged_right.mapmentions(
            lambda m: torch.cat(
                [rhidden_states[m.start_idx], rhidden_states[m.end_idx - 1]]
            )
        )
        right_chains_repr = torch.stack(
            [torch.mean(torch.stack(chain), dim=0) for chain in right_mentions_repr]
        ).unsqueeze(0)
        assert right_chains_repr.shape == (b, m, 2 * h)

        # TODO: perf
        mention_pairs_repr = torch.stack(
            [
                torch.cat((l, r), dim=1)
                for l, r in it.product(left_chains_repr, right_chains_repr)
            ]
        )
        assert mention_pairs_repr.shape == (b, m * a, 4 * h)

        # compute distance feature
        seq_size = len(merged_left.tokens) + len(merged_right.tokens)
        spans_idx = spans_indexs(list(range(seq_size)), model.config.max_span_size)
        left_mentions_idx = [
            max(m.end_idx for m in chain) for chain in merged_left.coref_chains
        ]
        left_mentions_idx = torch.tensor(left_mentions_idx).unsqueeze(0)

        roffset = len(merged_left.tokens)
        right_mentions_idx = [
            min(m.start_idx for m in chain) for chain in merged_right.coref_chains
        ]
        right_mentions_idx = merged_right.mapmentions(
            lambda m: spans_idx.index((m.start_idx + roffset, m.end_idx + roffset))
        )
        right_mentions_idx = torch.tensor(right_mentions_idx).unsqueeze(0)

        top_antecedents_index = torch.stack(
            [left_mentions_idx for _ in range(right_mentions_idx.shape[1])]
        )

        spans_nb = len(spans_idx)
        dist_ft = model.distance_feature(
            top_antecedents_index, right_mentions_idx, spans_nb, seq_size
        )
        dist_ft = torch.flatten(dist_ft, start_dim=1, end_dim=2)

        mention_pairs_repr = torch.cat((mention_pairs_repr, dist_ft), dim=2)
        compat = model.mention_compatibility_score(
            torch.flatten(mention_pairs_repr, start_dim=0, end_dim=1)
        )
        assert compat.shape == (b * m * a,)
        compat = compat.reshape(b, m, a)

        # - Should we assume mention score is 1?
        # - What is the range of the compat score?
        # from what I remember mention_score + pair_score should be < 0 if no match...
        # => let's assume that for now and forget about mention_score. Seems (not) legit.
        # Also, at MOST each cluster must correspond to another
        # cluster. this looks like bipartite matching. Let's ignore that
        # and do it greedily?
        left_offset = len(merged_left.tokens)
        right_chains_remaining = set(range(len(merged_right.coref_chains)))
        new_chains = []
        for left_chain, scores in zip(merged_left.coref_chains, compat):
            scores[torch.tensor(list(right_chains_remaining))] = float("-Inf")
            best_score = torch.max(scores, dim=1)
            if best_score.values[0] < 0.0:
                new_chains.append(left_chain)
                continue
            r_chain_i = best_score.indices[0]
            right_chain = merged_right.coref_chains[r_chain_i]
            new_chains.append(
                left_chain + [m.shifted(left_offset) for m in right_chain]
            )
            right_chains_remaining.remove(r_chain_i)
        for r_chain_i in right_chains_remaining:
            new_chains.append(
                [m.shifted(left_offset) for m in merged_right.coref_chains[r_chain_i]]
            )

    return CoreferenceDocument(merged_left.tokens + merged_right.tokens, new_chains)


def _stream_predict_wpieced_coref_raw(
    documents: List[Union[str, List[str]]],
    model: BertForCoreferenceResolution,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int = 1,
    quiet: bool = False,
    device_str: Literal["cpu", "cuda", "auto"] = "auto",
    lang: str = "en",
    return_hidden_state: bool = False,
) -> Generator[
    Tuple[
        List[CoreferenceDocument],
        List[CoreferenceDocument],
        BatchEncoding,
        BertCoreferenceResolutionOutput,
    ],
    None,
    None,
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
    data_collator = DataCollatorForSpanClassification(tokenizer, model.config.max_span_size)  # type: ignore
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

            out_docs = out.coreference_documents(
                [
                    [tokenizer.decode(t) for t in input_ids]  # type: ignore
                    for input_ids in batch["input_ids"]
                ]
            )

            yield batch_docs, out_docs, batch, out


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
    for original_docs, out_docs, batch, out in _stream_predict_wpieced_coref_raw(
        documents, model, tokenizer, batch_size, quiet, device_str, lang
    ):
        for batch_i, (original_doc, out_doc) in enumerate(zip(original_docs, out_docs)):
            seq_size = batch["input_ids"].shape[1]
            wp_to_token = [
                batch.token_to_word(batch_i, token_index=i) for i in range(seq_size)
            ]
            doc = out_doc.from_wpieced_to_tokenized(original_doc.tokens, wp_to_token)
            yield doc


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
        all_tokens = []
        wp_to_token = []

        if len(documents) == 0:
            return None

        for original_docs, out_docs, batch, out in _stream_predict_wpieced_coref_raw(
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

            all_tokens += list(flatten(doc.tokens for doc in original_docs))

            batch_size = batch["input_ids"].shape[0]
            seq_size = batch["input_ids"].shape[1]
            for batch_i in range(batch_size):
                wp_to_token += [
                    batch.token_to_word(batch_i, token_index=i) for i in range(seq_size)
                ]

        merged_doc_wpieced = merge_coref_outputs(docs, hidden_states, model)
        assert not merged_doc_wpieced is None  # we know that len(docs) > 0

        return merged_doc_wpieced.from_wpieced_to_tokenized(all_tokens, wp_to_token)

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
