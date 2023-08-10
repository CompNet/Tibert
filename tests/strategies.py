from hypothesis import strategies as st
from hypothesis.strategies import composite
from tibert import CoreferenceDocument
from tibert.bertcoref import Mention
from tibert.utils import spans_indexs


@composite
def coref_docs(
    draw, min_size: int = 0, max_size: int = 128, max_span_size: int = 5
) -> CoreferenceDocument:
    """A strategy to generate coreference documents.

    :param min_size: the minimum number of tokens of the document
    :param max_size: the maximum number of tokens of the document
    :param max_span_size: the maximum allowed size of spans
    """
    # generate tokens
    tokens = draw(
        st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "P")), min_size=1
            ),
            min_size=min_size,
            max_size=max_size,
        )
    )
    if len(tokens) == 0:
        return CoreferenceDocument(tokens, [])

    # generate mentions coordinates
    n = len(tokens)
    spans_idxs = spans_indexs(tokens, max_span_size)
    mentions_idxs = draw(
        st.lists(
            st.sampled_from(spans_idxs),
            min_size=0,
            max_size=int((n * (n + 1)) / 2),
            unique=True,
        )
    )

    # generate link between mentions (i.e. chains)
    chains_nb = (
        draw(st.integers(1, len(mentions_idxs))) if len(mentions_idxs) > 0 else 0
    )
    chains = [[] for _ in range(chains_nb)]
    for mention_idxs in mentions_idxs:
        chain_idx = draw(st.integers(0, chains_nb - 1))
        chains[chain_idx].append(mention_idxs)
    # remove empty chains
    chains = [c for c in chains if len(c) > 0]

    return CoreferenceDocument(
        tokens,
        [
            [Mention(tokens[m_idxs[0] : m_idxs[1]], *m_idxs) for m_idxs in chain]
            for chain in chains
        ],
    )
