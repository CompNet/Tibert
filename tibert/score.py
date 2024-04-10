from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Collection,
    Dict,
    List,
    Literal,
    Callable,
    Set,
    Tuple,
)
import itertools as it
from statistics import mean
import numpy as np
from neleval.coref_metrics import muc, b_cubed, ceaf, pairwise, pairwise_negative
from tibert.utils import spans_indexs

if TYPE_CHECKING:
    from tibert.bertcoref import CoreferenceDocument, Mention


def _coref_doc_to_neleval_format(doc: CoreferenceDocument, max_span_size: int):
    """Convert a coreference document to the format expected by ``neleval``"""

    spans_idxs = spans_indexs(doc.tokens, max_span_size)

    # { chain_id => {tokens_id} }
    clusters = {}

    for chain_i, chain in enumerate(doc.coref_chains):
        clusters[chain_i] = set(
            [
                spans_idxs.index((mention.start_idx, mention.end_idx))
                for mention in chain
            ]
        )

    return clusters


def _max_span_size(pred: CoreferenceDocument, ref: CoreferenceDocument) -> int:
    try:
        pred_max_span_size = max(
            [
                (mention.end_idx - mention.start_idx) + 1
                for chain in pred.coref_chains
                for mention in chain
            ]
        )
    except ValueError:
        pred_max_span_size = 0

    try:
        ref_max_span_size = max(
            [
                (mention.end_idx - mention.start_idx) + 1
                for chain in ref.coref_chains
                for mention in chain
            ]
        )
    except ValueError:
        ref_max_span_size = 0

    return max(pred_max_span_size, ref_max_span_size)


def _neleval_precision_recall_f1(
    pred: CoreferenceDocument,
    ref: CoreferenceDocument,
    neleval_fn: Callable[
        [Dict[int, Set[str]], Dict[int, Set[str]]],
        Tuple[float, float, float, float],
    ],
) -> Tuple[float, float, float]:
    """Get precision, recall and f1 for a predicted document from a neleval metrics."""
    max_span_size = _max_span_size(pred, ref)
    neleval_pred = _coref_doc_to_neleval_format(pred, max_span_size + 1)
    neleval_ref = _coref_doc_to_neleval_format(ref, max_span_size + 1)

    if neleval_pred == neleval_ref:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    else:
        # num = numerator
        # den = denominator
        p_num, p_den, r_num, r_den = neleval_fn(neleval_ref, neleval_pred)
        precision = p_num / p_den if p_den > 0 else 0.0
        recall = r_num / r_den if r_den > 0 else 0.0
        if precision + recall != 0.0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

    return precision, recall, f1


def score_muc(
    preds: List[CoreferenceDocument], refs: List[CoreferenceDocument]
) -> Tuple[float, float, float]:
    """Compute MUC using neleval.

    .. note::

        the returned metrics are the macro-wise mean across documents

    :return: (precision, recall, f1)
    """
    assert len(preds) > 0
    assert len(preds) == len(refs)

    # neleval use np.int and np.bool, which are deprecated
    # (https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations). The
    # two following lines fix the resulting crash.
    np.int = int  # type: ignore
    np.bool = bool  # type: ignore

    precisions, recalls, f1s = [], [], []
    for pred, ref in zip(preds, refs):
        p, r, f1 = _neleval_precision_recall_f1(pred, ref, muc)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return mean(precisions), mean(recalls), mean(f1s)


def score_b_cubed(
    preds: List[CoreferenceDocument], refs: List[CoreferenceDocument]
) -> Tuple[float, float, float]:
    """Compute B^3 using neleval.

    .. note::

        the returned metrics are the macro-wise mean across documents

    :return: (precision, recall, f1)
    """
    assert len(preds) > 0
    assert len(preds) == len(refs)

    # neleval use np.int and np.bool, which are deprecated
    # (https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations). The
    # two following lines fix the resulting crash.
    np.int = int  # type: ignore
    np.bool = bool  # type: ignore

    precisions, recalls, f1s = [], [], []
    for pred, ref in zip(preds, refs):
        p, r, f1 = _neleval_precision_recall_f1(pred, ref, b_cubed)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return mean(precisions), mean(recalls), mean(f1s)


def score_ceaf(
    preds: List[CoreferenceDocument], refs: List[CoreferenceDocument]
) -> Tuple[float, float, float]:
    """Compute CEAF using neleval.

    .. note::

        the returned metrics are the macro-wise mean across documents

    :return: (precision, recall, f1)
    """
    assert len(preds) > 0
    assert len(preds) == len(refs)

    # neleval use np.int and np.bool, which are deprecated
    # (https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations). The
    # two following lines fix the resulting crash.
    np.int = int  # type: ignore
    np.bool = bool  # type: ignore

    precisions, recalls, f1s = [], [], []
    for pred, ref in zip(preds, refs):
        p, r, f1 = _neleval_precision_recall_f1(pred, ref, ceaf)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return mean(precisions), mean(recalls), mean(f1s)


def score_blanc(
    preds: List[CoreferenceDocument], refs: List[CoreferenceDocument]
) -> Tuple[float, float, float]:
    assert len(preds) > 0
    assert len(preds) == len(refs)

    prf = []

    for pred, ref in zip(preds, refs):
        if pred.coref_chains == ref.coref_chains:
            prf.append((1, 1, 1))
            continue

        max_span_size = _max_span_size(pred, ref)
        neleval_pred = _coref_doc_to_neleval_format(pred, max_span_size + 1)
        neleval_ref = _coref_doc_to_neleval_format(ref, max_span_size + 1)

        p_num, p_den, r_num, r_den = pairwise(neleval_ref, neleval_pred)
        np_num, np_den, nr_num, nr_den = pairwise_negative(neleval_ref, neleval_pred)

        P_c = 0 if p_den == 0 else p_num / p_den
        P_n = 0 if np_den == 0 else np_num / np_den

        R_c = 0 if r_den == 0 else r_num / r_den
        R_n = 0 if nr_den == 0 else nr_num / nr_den

        F_c = 0 if P_c + R_c == 0 else (2 * P_c * R_c) / (P_c + R_c)
        F_n = 0 if P_n + R_n == 0 else (2 * P_n * R_n) / (P_n + R_n)

        prf.append(((P_c + P_n) / 2.0, (R_c + R_n) / 2.0, (F_c + F_n) / 2.0))

    return (
        mean([m[0] for m in prf]),
        mean([m[1] for m in prf]),
        mean([m[2] for m in prf]),
    )


def score_lea(
    preds: List[CoreferenceDocument], refs: List[CoreferenceDocument]
) -> Tuple[float, float, float]:
    """Score coreference prediction according to LEA

    .. note::

        the returned metrics are the macro-wise mean across documents

    :return: (precision, recall, f1)
    """
    assert len(preds) > 0
    assert len(preds) == len(refs)

    def lea_link(entity: List[Mention]) -> set:
        if len(entity) == 1:
            return set([(entity[0], entity[0])])
        return set(it.combinations(entity, 2))

    def lea_link_score(links: set) -> int:
        return len(links)

    def lea_res_score(entity_lea_link: set, entities_lea_link: List[set]) -> float:
        score = 0

        entity_link_score = lea_link_score(entity_lea_link)
        if entity_link_score == 0:
            return score

        for o_entity_lea_link in entities_lea_link:
            score += (
                lea_link_score(entity_lea_link.intersection(o_entity_lea_link))
                / entity_link_score
            )
        return score

    precisions, recalls, f1s = [], [], []

    for pred, ref in zip(preds, refs):
        ref_lea_links = [lea_link(chain) for chain in ref.coref_chains]
        pred_lea_links = [lea_link(chain) for chain in pred.coref_chains]

        precision_num = 0
        precision_den = 0
        for pred_lea_link, pred_chain in zip(pred_lea_links, pred.coref_chains):
            importance = len(pred_chain)
            precision_den += importance
            precision_num += importance * lea_res_score(pred_lea_link, ref_lea_links)

        precision = precision_num / precision_den if precision_den > 0 else 0
        precisions.append(precision)

        recall_num = 0
        recall_den = 0
        for ref_lea_link, ref_chain in zip(ref_lea_links, ref.coref_chains):
            importance = len(ref_chain)
            recall_den += importance
            recall_num += importance * lea_res_score(ref_lea_link, pred_lea_links)
        recall = recall_num / recall_den if recall_den > 0 else 0
        recalls.append(recall)

        if (precision + recall) == 0:
            f1 = 0
        else:
            f1 = (2 * precision * recall) / (precision + recall)
        f1s.append(f1)

    return mean(precisions), mean(recalls), mean(f1s)


def score_coref_predictions(
    preds: List[CoreferenceDocument], refs: List[CoreferenceDocument]
) -> Dict[
    Literal["MUC", "B3", "CEAF", "BLANC", "LEA"],
    Dict[Literal["precision", "recall", "f1"], float],
]:
    """Score coreference prediction according to MUC, B3, CEAF, BLANC and LEA

    .. note::

        the returned metrics are the macro-wise mean across documents

    :param preds: Predictions
    :param refs: References
    """
    muc_precision, muc_recall, muc_f1 = score_muc(preds, refs)
    b3_precision, b3_recall, b3_f1 = score_b_cubed(preds, refs)
    ceaf_precision, ceaf_recall, ceaf_f1 = score_ceaf(preds, refs)
    blanc_precision, blanc_recall, blanc_f1 = score_blanc(preds, refs)
    lea_precision, lea_recall, lea_f1 = score_lea(preds, refs)

    return {
        "MUC": {
            "precision": muc_precision,
            "recall": muc_recall,
            "f1": muc_f1,
        },
        "B3": {
            "precision": b3_precision,
            "recall": b3_recall,
            "f1": b3_f1,
        },
        "CEAF": {
            "precision": ceaf_precision,
            "recall": ceaf_recall,
            "f1": ceaf_f1,
        },
        "BLANC": {
            "precision": blanc_precision,
            "recall": blanc_recall,
            "f1": blanc_f1,
        },
        "LEA": {
            "precision": lea_precision,
            "recall": lea_recall,
            "f1": lea_f1,
        },
    }


def doc_mentions(doc: CoreferenceDocument) -> List[Mention]:
    return [mention for chain in doc.coref_chains for mention in chain]


def score_mention_detection(
    preds: List[CoreferenceDocument], refs: List[CoreferenceDocument]
) -> Tuple[float, float, float]:
    """Compute mention detection precision, recall and F1.

    :param preds: predictions
    :param refs: references

    :return: ``(precision, recall, f1)``
    """
    assert len(preds) > 0
    assert len(refs) > 0

    precision_l = []
    recall_l = []
    f1_l = []

    for pred, ref in zip(preds, refs):
        pred_mentions = doc_mentions(pred)
        ref_mentions = doc_mentions(ref)

        if len(pred_mentions) == 0:
            continue
        precision = len([m for m in pred_mentions if m in ref_mentions]) / len(
            pred_mentions
        )

        if len(ref_mentions) == 0:
            continue
        recall = len([m for m in ref_mentions if m in pred_mentions]) / len(
            ref_mentions
        )

        if precision + recall == 0:
            continue

        f1 = 2 * (precision * recall) / (precision + recall)

        precision_l.append(precision)
        recall_l.append(recall)
        f1_l.append(f1)

    if len(f1_l) == 0:
        print("[warning] undefined F1 for all samples")
        return (0.0, 0.0, 0.0)

    return (mean(precision_l), mean(recall_l), mean(f1_l))
