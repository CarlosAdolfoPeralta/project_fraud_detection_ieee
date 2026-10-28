"""Feature engineering para IEEE-CIS.

Principios de diseño:
- Ninguna variable usa el target (isFraud).
- Las variables de comportamiento por cliente son CAUSALES: para cada transacción
  solo usan transacciones anteriores en el tiempo. Así no se filtra información del
  futuro entre train / validación / test, y se parece a lo que se podría calcular
  en producción.
"""
import numpy as np
import pandas as pd

CARD_COLS = ["card1", "card2", "card3", "card5"]

# Columnas auxiliares (claves de agrupación): se usan para calcular features,
# pero NO deben entrar al modelo (son identificadores de alta cardinalidad).
KEY_COLS = ["card_key", "uid"]

NEW_FEATURES = [
    "hour", "dow", "amt_log", "amt_frac", "n_missing",
    "P_email_provider", "R_email_provider", "email_match",
    "card_n_prev", "card_secs_since_prev", "card_amt_ratio",
    "card_n_1h", "card_n_24h",
    "uid_n_prev", "uid_secs_since_prev", "uid_amt_ratio",
]

# Variantes para definir el identificador `uid` (columnas que lo componen).
# `_d1n` = day - D1 (redondeado).
UID_VARIANTS = {
    "card_addr_d1n": ["card_key", "addr1", "_d1n"],
    "card_d1n": ["card_key", "_d1n"],
    "card_addr": ["card_key", "addr1"],
}


def _causal_stats(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """Estadísticas por clave usando SOLO transacciones anteriores.

    - {prefix}_n_prev:           cantidad de transacciones previas de la clave
    - {prefix}_secs_since_prev:  segundos desde la transacción anterior (NaN si es la primera)
    - {prefix}_amt_ratio:        monto actual / monto medio de las transacciones previas
    """
    order = np.argsort(df["TransactionDT"].values, kind="stable")
    tmp = pd.DataFrame(
        {
            "k": df[key].values[order],
            "dt": df["TransactionDT"].values[order].astype("float64"),
            "amt": df["TransactionAmt"].values[order].astype("float64"),
        }
    )
    g = tmp.groupby("k", sort=False)
    n_prev = g.cumcount()
    secs_since_prev = tmp["dt"] - g["dt"].shift(1)
    sum_prev = g["amt"].cumsum() - tmp["amt"]
    mean_prev = sum_prev / n_prev.replace(0, np.nan)
    amt_ratio = tmp["amt"] / mean_prev

    # volver al orden original del DataFrame
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    out = pd.DataFrame(
        {
            f"{prefix}_n_prev": n_prev.values[inv].astype("int32"),
            f"{prefix}_secs_since_prev": secs_since_prev.values[inv].astype("float32"),
            f"{prefix}_amt_ratio": amt_ratio.values[inv].astype("float32"),
        },
        index=df.index,
    )
    return out


def _velocity(df: pd.DataFrame, key: str, prefix: str, windows: dict) -> pd.DataFrame:
    """Cantidad de transacciones previas de la clave dentro de ventanas de tiempo.

    Para cada transacción cuenta las de la misma clave en [t - ventana, t), es decir,
    estrictamente anteriores (causal). `windows` = {"1h": 3600, "24h": 86400}.
    """
    dt = df["TransactionDT"].values.astype("int64")
    comp = df[key].values.astype("int64") * (2**32) + dt  # clave y tiempo en un solo número
    order = np.argsort(comp, kind="stable")
    comp_sorted = comp[order]
    first_same = np.searchsorted(comp_sorted, comp_sorted, side="left")  # excluye empates
    out = {}
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    for label, secs in windows.items():
        start = np.searchsorted(comp_sorted, comp_sorted - secs, side="left")
        out[f"{prefix}_n_{label}"] = (first_same - start)[inv].astype("int32")
    return pd.DataFrame(out, index=df.index)


def _email_provider(s: pd.Series) -> pd.Series:
    """'gmail.com' -> 'gmail' (primer tramo del dominio). Conserva NaN."""
    return s.astype("object").str.split(".").str[0].astype("category")


def build_features(df: pd.DataFrame, uid_variant: str = "card_addr_d1n") -> pd.DataFrame:
    """Devuelve una copia de df con las variables nuevas. Requiere la columna `day`.

    uid_variant: cómo se define el cliente aproximado (ver UID_VARIANTS).
    """
    df = df.copy()

    # --- Tiempo (relativo: el origen de TransactionDT es desconocido) ---
    df["hour"] = ((df["TransactionDT"] // 3600) % 24).astype("int8")
    df["dow"] = (df["day"] % 7).astype("int8")

    # --- Monto ---
    amt = df["TransactionAmt"].astype("float64")
    df["amt_log"] = np.log1p(amt).astype("float32")
    # parte decimal: montos con centavos "raros" suelen indicar conversión de moneda
    df["amt_frac"] = (amt - np.floor(amt)).astype("float32")

    # --- Nulos por fila ---
    df["n_missing"] = df.isna().sum(axis=1).astype("int16")

    # --- Email ---
    df["P_email_provider"] = _email_provider(df["P_emaildomain"])
    df["R_email_provider"] = _email_provider(df["R_emaildomain"])
    both = df["P_emaildomain"].notna() & df["R_emaildomain"].notna()
    same = df["P_emaildomain"].astype("object") == df["R_emaildomain"].astype("object")
    df["email_match"] = np.where(both, same.astype("float32"), np.nan).astype("float32")

    # --- Identificadores aproximados de cliente ---
    # card_key: combinación de atributos de tarjeta.
    df["card_key"] = df.groupby(CARD_COLS, dropna=False, observed=True).ngroup().astype("int32")
    # uid: card_key + dirección + "día de inicio" estimado (day - D1).
    # Hipótesis de la comunidad de la competencia: D1 se comporta como días desde el
    # inicio de la tarjeta, por lo que day - D1 es aproximadamente constante por cliente.
    d1n = (df["day"].astype("float32") - df["D1"]).round()
    df["_d1n"] = d1n
    df["uid"] = (
        df.groupby(UID_VARIANTS[uid_variant], dropna=False, observed=True)
        .ngroup()
        .astype("int32")
    )
    df = df.drop(columns="_d1n")

    # --- Comportamiento por cliente (causal) ---
    df = pd.concat(
        [
            df,
            _causal_stats(df, "card_key", "card"),
            _velocity(df, "card_key", "card", {"1h": 3600, "24h": 86400}),
            _causal_stats(df, "uid", "uid"),
        ],
        axis=1,
    )
    return df
