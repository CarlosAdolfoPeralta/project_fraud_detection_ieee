"""División temporal de IEEE-CIS: train / validación / test sin mezclar pasado y futuro."""
import pandas as pd

SECONDS_PER_DAY = 24 * 3600

# Límites por defecto (en días relativos): train [0, 120), valid [120, 150), test [150, fin]
TRAIN_END_DAY = 120
VALID_END_DAY = 150


def add_day(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega la columna `day` (día relativo) a partir de TransactionDT."""
    df = df.copy()
    df["day"] = (df["TransactionDT"] // SECONDS_PER_DAY).astype("int16")
    return df


def temporal_split(
    df: pd.DataFrame,
    train_end_day: int = TRAIN_END_DAY,
    valid_end_day: int = VALID_END_DAY,
    gap_days: int = 0,
):
    """Divide el dataset por tiempo.

    - train: day < train_end_day
    - valid: train_end_day + gap_days <= day < valid_end_day
    - test:  day >= valid_end_day + gap_days

    `gap_days` deja un hueco entre tramos para reducir la fuga de información
    por comportamientos que se extienden en el tiempo (p. ej. una misma tarjeta
    comprometida que aparece antes y después del corte). Por defecto es 0.
    """
    if "day" not in df.columns:
        df = add_day(df)

    train = df[df["day"] < train_end_day]
    valid = df[(df["day"] >= train_end_day + gap_days) & (df["day"] < valid_end_day)]
    test = df[df["day"] >= valid_end_day + gap_days]
    return train, valid, test


def summarize_splits(**splits: pd.DataFrame) -> pd.DataFrame:
    """Tabla resumen por tramo: filas, rango de días, tasa de fraude y % con identidad."""
    rows = []
    for name, part in splits.items():
        rows.append(
            {
                "split": name,
                "filas": len(part),
                "% del total": 100 * len(part) / sum(len(p) for p in splits.values()),
                "día_min": part["day"].min(),
                "día_max": part["day"].max(),
                "fraudes": int(part["isFraud"].sum()),
                "tasa_fraude_%": 100 * part["isFraud"].mean(),
                "%_con_identidad": 100 * part["has_identity"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("split").round(2)
