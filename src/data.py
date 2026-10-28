"""Carga, merge y reducción de memoria del dataset IEEE-CIS Fraud Detection."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Reduce el uso de memoria haciendo downcast de tipos numéricos.

    - Enteros: al tipo entero más chico que contenga el rango.
    - Floats: float32 como mínimo (float16 pierde precisión en montos).
    - Strings repetidos (object): category.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = df[col].astype(np.float32)
        elif col_type == object:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(
            f"Memoria: {start_mem:.1f} MB -> {end_mem:.1f} MB "
            f"({100 * (start_mem - end_mem) / start_mem:.0f}% menos)"
        )
    return df


def load_ieee(raw_dir: Path = RAW_DIR, merge: bool = True):
    """Carga train_transaction y train_identity.

    Si merge=True devuelve un único DataFrame (left join por TransactionID)
    con una columna extra `has_identity` que indica si la transacción
    tenía información de identidad. Si merge=False devuelve ambos por separado.
    """
    raw_dir = Path(raw_dir)
    print("Cargando train_transaction.csv ...")
    trans = reduce_mem_usage(pd.read_csv(raw_dir / "train_transaction.csv"))
    print("Cargando train_identity.csv ...")
    ident = reduce_mem_usage(pd.read_csv(raw_dir / "train_identity.csv"))

    if not merge:
        return trans, ident

    # No todas las transacciones tienen identidad -> left join
    ident["has_identity"] = 1
    df = trans.merge(ident, on="TransactionID", how="left")
    df["has_identity"] = df["has_identity"].fillna(0).astype(np.int8)
    print(f"Dataset final: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    return df


def save_processed(df: pd.DataFrame, name: str = "train_merged.parquet") -> Path:
    """Guarda el DataFrame en parquet (requiere pyarrow) para cargas rápidas."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / name
    df.to_parquet(path, index=False)
    print(f"Guardado en {path}")
    return path


def load_processed(name: str = "train_merged.parquet") -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / name)
