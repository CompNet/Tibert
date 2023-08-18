from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Literal, Callable, Set, Tuple
from statistics import mean
import numpy as np
from tibert.utils import spans_indexs

if TYPE_CHECKING:
    from tibert.bertcoref import CoreferenceDocument, Mention


def score_coref_predictions(
    preds: List[CoreferenceDocument], refs: List[CoreferenceDocument]
) -> Dict[
    Literal["MUC", "B3", "CEAF"],
    Dict[Literal["precision", "recall", "f1"], float],
]:
    """Score coreference prediction according to MUC, B3 and CEAF
    metrics

    .. note::

        Needs package ``neleval``

    :param preds: Predictions
    :param refs: References
    """
    assert len(preds) == len(refs)
    assert len(preds) > 0

    # neleval use np.int and np.bool, which are deprecated
    # (https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations). The
    # two following lines fix the resulting crash.
    np.int = int  # type: ignore
    np.bool = bool  # type: ignore

    from neleval.coref_metrics import muc, b_cubed, ceaf

    def coref_doc_to_neleval_format(doc: CoreferenceDocument, max_span_size: int):
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

    def precisions_recalls_f1s(
        preds: List[CoreferenceDocument],
        refs: List[CoreferenceDocument],
        neleval_fn: Callable[
            [Dict[int, Set[str]], Dict[int, Set[str]]],
            Tuple[float, float, float, float],
        ],
    ) -> Tuple[List[float], List[float], List[float]]:
        """Get precisions, recalls and f1s from a neleval metrics"""
        precisions = []
        recalls = []
        f1s = []
        for pred, ref in zip(preds, refs):
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
            max_span_size = max(pred_max_span_size, ref_max_span_size)
            # TODO max_span_size
            neleval_pred = coref_doc_to_neleval_format(pred, max_span_size + 1)
            neleval_ref = coref_doc_to_neleval_format(ref, max_span_size + 1)

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

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        return precisions, recalls, f1s

    muc_precisions, muc_recalls, muc_f1s = precisions_recalls_f1s(preds, refs, muc)
    b3_precisions, b3_recalls, b3_f1s = precisions_recalls_f1s(preds, refs, b_cubed)
    ceaf_precisions, ceaf_recalls, ceaf_f1s = precisions_recalls_f1s(preds, refs, ceaf)

    return {
        "MUC": {
            "precision": mean(muc_precisions),
            "recall": mean(muc_recalls),
            "f1": mean(muc_f1s),
        },
        "B3": {
            "precision": mean(b3_precisions),
            "recall": mean(b3_recalls),
            "f1": mean(b3_f1s),
        },
        "CEAF": {
            "precision": mean(ceaf_precisions),
            "recall": mean(ceaf_recalls),
            "f1": mean(ceaf_f1s),
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
