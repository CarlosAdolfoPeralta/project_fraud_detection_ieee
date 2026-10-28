"""Métricas para evaluar modelos de fraude (clases muy desbalanceadas)."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def recall_at_precision(y_true, y_score, min_precision: float) -> float:
    """Mayor recall alcanzable con una precisión >= min_precision (0 si no se alcanza)."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = precision >= min_precision
    return float(recall[mask].max()) if mask.any() else 0.0


def precision_at_recall(y_true, y_score, min_recall: float) -> float:
    """Mayor precisión alcanzable con un recall >= min_recall (0 si no se alcanza)."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = recall >= min_recall
    return float(precision[mask].max()) if mask.any() else 0.0


def evaluate(y_true, y_score, precisions=(0.5, 0.8)) -> dict:
    """Métricas principales: AUC-PR (principal), AUC-ROC y recall a precisión fija."""
    y_true = np.asarray(y_true)
    out = {
        "auc_pr": average_precision_score(y_true, y_score),
        "auc_roc": roc_auc_score(y_true, y_score),
        "tasa_base": y_true.mean(),  # AUC-PR de un modelo aleatorio
    }
    for p in precisions:
        out[f"recall@prec{int(p * 100)}"] = recall_at_precision(y_true, y_score, p)
    return out


def evaluate_by_segment(df: pd.DataFrame, y_score, segment_col: str, target: str = "isFraud") -> pd.DataFrame:
    """AUC-PR y AUC-ROC por segmento (omite segmentos sin ambas clases)."""
    tmp = pd.DataFrame(
        {"seg": df[segment_col].values, "y": df[target].values, "score": np.asarray(y_score)}
    )
    rows = []
    for seg, g in tmp.groupby("seg", observed=True):
        if g["y"].nunique() < 2:
            continue
        rows.append(
            {
                segment_col: seg,
                "filas": len(g),
                "fraudes": int(g["y"].sum()),
                "tasa_fraude_%": 100 * g["y"].mean(),
                "auc_pr": average_precision_score(g["y"], g["score"]),
                "auc_roc": roc_auc_score(g["y"], g["score"]),
            }
        )
    return pd.DataFrame(rows).set_index(segment_col).round(4)


# ---------------------------------------------------------------------------
# Costo de negocio
# ---------------------------------------------------------------------------
# Modelo de costo (supuestos explícitos, ver notebook 06):
# - Cada transacción con score >= umbral genera una alerta que se revisa manualmente.
#   Cada revisión cuesta `review_cost` (fijo).
# - La revisión detecta perfectamente el fraude: una alerta sobre un fraude evita la pérdida.
# - Un fraude que NO se alerta cuesta el monto completo de la transacción.
# - Un no fraude alertado solo cuesta la revisión.


def cost_curve(y_true, scores, amounts, review_cost: float) -> pd.DataFrame:
    """Costo total para cada cantidad k de alertas (las k transacciones de mayor score).

    Devuelve n_flagged, threshold (score de la k-ésima alerta; inf para k=0) y cost.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores)
    a = np.asarray(amounts, dtype="float64")

    order = np.argsort(-s, kind="stable")
    y_o, a_o, s_o = y[order], a[order], s[order]

    caught = np.concatenate([[0.0], np.cumsum(a_o * y_o)])
    total_fraud_amt = float((a * y).sum())
    k = np.arange(len(y) + 1)
    cost = review_cost * k + total_fraud_amt - caught
    threshold = np.concatenate([[np.inf], s_o])
    return pd.DataFrame({"n_flagged": k, "threshold": threshold, "cost": cost})


def best_threshold(y_true, scores, amounts, review_cost: float) -> float:
    """Umbral que minimiza el costo total. inf = no conviene generar ninguna alerta."""
    curve = cost_curve(y_true, scores, amounts, review_cost)
    return float(curve.loc[curve["cost"].idxmin(), "threshold"])


def cost_at_threshold(y_true, scores, amounts, review_cost: float, threshold: float) -> dict:
    """Costo y métricas operativas al aplicar un umbral fijo."""
    y = np.asarray(y_true)
    s = np.asarray(scores)
    a = np.asarray(amounts, dtype="float64")

    flagged = s >= threshold
    tp = int((flagged & (y == 1)).sum())
    n_flagged = int(flagged.sum())
    n_fraud = int((y == 1).sum())
    missed_amt = float(a[(~flagged) & (y == 1)].sum())
    cost = review_cost * n_flagged + missed_amt

    no_model = float(a[y == 1].sum())          # no revisar nada
    review_all = review_cost * len(y)          # revisar todo
    return {
        "umbral": threshold,
        "alertas_%": 100 * n_flagged / len(y),
        "precision": tp / n_flagged if n_flagged else 0.0,
        "recall": tp / n_fraud if n_fraud else 0.0,
        "costo": cost,
        "costo_sin_modelo": no_model,
        "costo_revisar_todo": review_all,
        "ahorro_vs_sin_modelo_%": 100 * (1 - cost / no_model) if no_model else 0.0,
    }
