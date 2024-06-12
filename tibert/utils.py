from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Collection, List, Tuple, cast
from more_itertools import windowed
from rich.console import Console
import torch

if TYPE_CHECKING:
    from tibert.bertcoref import CoreferenceDocument


RICH_COLORS = [
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "bright_black",
    "bright_red",
    "bright_green",
    "bright_yellow",
    "bright_magenta",
    "bright_cyan",
]


def get_cat_color(index: int) -> str:
    """Get a categorical color from RICH_COLORS"""
    return RICH_COLORS[index % len(RICH_COLORS)]


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
    """Split a coreference document using sentences as delimitation"""
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


def split_coreference_document_tokens(
    document: CoreferenceDocument, tokens_nb: int
) -> List[CoreferenceDocument]:
    from tibert import CoreferenceDocument, Mention

    token_blocks = [
        document.tokens[block_start : block_start + tokens_nb]
        for block_start in range(0, len(document.tokens), tokens_nb)
    ]

    documents = []
    block_start = 0
    for block in token_blocks:
        block_end = block_start + tokens_nb
        coref_chains = [
            [
                Mention(
                    mention.tokens,
                    mention.start_idx - block_start,
                    mention.end_idx - block_start,
                )
                for mention in chain
                if mention.start_idx >= block_start and mention.end_idx <= block_end
            ]
            for chain in document.coref_chains
        ]
        coref_chains = [c for c in coref_chains if not len(c) == 0]
        documents.append(CoreferenceDocument(block, coref_chains))
        block_start += tokens_nb

    return documents


def pprint_coreference_document(doc: CoreferenceDocument):
    """Pretty-print a coreference document on the terminal."""

    console = Console(force_terminal=True, color_system="standard", highlight=False)

    mentions = []
    for chain_i, chain in enumerate(doc.coref_chains):
        for mention in chain:
            mentions.append((chain_i, mention.start_idx, mention.end_idx))

    out = []

    for token_i, token in enumerate(doc.tokens):
        related_mentions = [
            (chain_i, start_i, end_i)
            for chain_i, start_i, end_i in mentions
            if start_i == token_i or end_i - 1 == token_i
        ]
        # sort to have outermost mentions first
        related_mentions = sorted(related_mentions, key=lambda c: c[1] - c[2])

        for chain_i, start_i, _ in related_mentions:
            if token_i == start_i:
                color = get_cat_color(chain_i)
                out.append(f"[{color}]({chain_i}")

        out.append(token)

        for chain_i, _, end_i in related_mentions:
            if token_i == end_i - 1:
                color = get_cat_color(chain_i)
                out.append(f")[/{color}]")

    try:
        console.print(" ".join(out))
    except Exception:
        print(out)


def gpu_memory_usage() -> float:
    mem_infos = torch.cuda.mem_get_info()
    return 1 - mem_infos[0] / mem_infos[1]  # type: ignore
