from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Collection, List, Tuple, cast
from more_itertools import windowed
from rich.console import Console
import torch

if TYPE_CHECKING:
    from tibert.bertcoref import CoreferenceDocument


T = TypeVar("T")


def spans(seq: Collection[T], max_len: int) -> List[Tuple[T]]:
    """Cut the input sequence into all possible spans up to a maximum length

    .. note::

        spans are ordered from the smallest to the biggest,
        from the beginning of seq to the end of seq.


    :param seq:
    :param max_len:
    :return:
    """
    out_spans = []
    for i in range(1, min(len(seq) + 1, max_len + 1)):
        for span in windowed(seq, i):
            out_spans.append(span)
    return out_spans


def spans_indexs(seq: List, max_len: int) -> List[Tuple[int, int]]:
    """"""
    indexs = []
    for i in range(1, min(len(seq) + 1, max_len + 1)):
        for span in windowed(range(len(seq)), i):
            span = cast(Tuple[int, ...], span)
            indexs.append((min(span), max(span) + 1))
    return indexs


def batch_index_select(
    input: torch.Tensor, dim: int, index: torch.Tensor
) -> torch.Tensor:
    """Batched version of :func:`torch.index_select`.
    Inspired by https://discuss.pytorch.org/t/batched-index-select/9115/8

    :param input: a torch tensor of shape ``(B, *)`` where ``*``
        is any number of additional dimensions.
    :param dim: the dimension in which to index
    :param index: index tensor of shape ``(B, I)``

    :return: a tensor which indexes ``input`` along dimension ``dim``
        using ``index``. This tensor has the same shape as ``input``,
        except in dimension ``dim``, where it has dimension ``I``.
    """
    batch_size = input.shape[0]

    view = [batch_size] + [1 if i != dim else -1 for i in range(1, len(input.shape))]

    expansion = list(input.shape)
    expansion[0] = batch_size
    expansion[dim] = -1

    return torch.gather(input, dim, index.view(view).expand(expansion))


def split_coreference_document(
    document: CoreferenceDocument, sents_nb: int
) -> List[CoreferenceDocument]:
    from tibert import CoreferenceDocument, Mention

    punctuation_indexs = []
    sents_counter = 0
    for i, token in enumerate(document.tokens):
        if token in (".", "?", "!"):
            sents_counter += 1
            if sents_counter == sents_nb:
                punctuation_indexs.append(i)
                sents_counter = 0

    documents = []
    for idx1, idx2 in windowed([0] + punctuation_indexs, 2):  # type: ignore
        if idx2 is None:
            idx2 = len(document.tokens) - 1
        assert idx1 is not None
        coref_chains = [
            [
                Mention(
                    mention.tokens, mention.start_idx - idx1, mention.end_idx - idx1
                )
                for mention in chain
                if mention.start_idx >= idx1 and mention.end_idx <= idx2
            ]
            for chain in document.coref_chains
        ]
        coref_chains = [c for c in coref_chains if not len(c) == 0]
        documents.append(
            CoreferenceDocument(
                document.tokens[idx1 : idx2 + 1],
                coref_chains,
            )
        )

    return documents


def pprint_coreference_document(doc: CoreferenceDocument):
    """Pretty-print a coreference document on the terminal."""

    console = Console(force_terminal=True, color_system="standard")

    mentions = []
    for chain_i, chain in enumerate(doc.coref_chains):
        for mention in chain:
            mentions.append((chain_i, mention.start_idx, mention.end_idx))

    out = []

    for token_i, token in enumerate(doc.tokens):

        related_chains = [
            (chain_i, start_i, end_i)
            for chain_i, start_i, end_i in mentions
            if start_i == token_i or end_i - 1 == token_i
        ]

        for chain_i, start_i, _ in related_chains:
            if token_i == start_i:
                out.append(f"[red]({chain_i}")

        out.append(token)

        for chain_i, _, end_i in related_chains:
            if token_i == end_i - 1:
                out.append(f")[/red]")

    try:
        console.print(" ".join(out))
    except Exception:
        print(out)


def gpu_memory_usage() -> float:
    mem_infos = torch.cuda.mem_get_info()
    return 1 - mem_infos[0] / mem_infos[1]  # type: ignore
