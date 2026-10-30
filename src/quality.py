"""Control de calidad: artefactos de señal, outliers estadísticos y datos faltantes.

Distinción metodológica central:
  - ARTEFACTO DE SEÑAL: el registro no refleja actividad cerebral (electrodo
    suelto -> señal plana, saturación del conversor -> señal recortada,
    valores no finitos). Se detecta con criterios *físicos* sobre la señal y se
    marca con `is_artifact` / `artifact_reason`. Las filas se CONSERVAN.
  - OUTLIER ESTADÍSTICO: valor extremo respecto de la distribución
    (regla de Tukey, 1.5·IQR). Puede ser fisiología real (p. ej. ondas lentas
    de gran amplitud en N3), por eso es solo informativo y NUNCA elimina filas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

ARTIFACT_NON_FINITE = "non_finite"
ARTIFACT_FLAT = "flat_signal"
ARTIFACT_CLIPPED = "clipped"

# Para potencias (fuertemente asimétricas a la derecha) la regla de Tukey se
# aplica sobre log10; en escala lineal casi toda la cola superior sería "outlier".
_LOG_SCALE_SUFFIXES = ("_abs", "total_power")
IQR_FEATURES = ["delta_abs", "total_power", "delta_rel", "alpha_rel",
                "sample_entropy", "spectral_entropy", "rms", "peak_to_peak"]
IQR_FACTOR = 1.5


def flag_artifacts(epochs_uv: np.ndarray, n_clipped: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Marca artefactos por (epoch, canal) a partir de la señal filtrada en µV.

    `n_clipped` son las muestras saturadas de cada (epoch, canal), contadas sobre
    la señal sin filtrar (ver count_clipped_samples).

    Criterios (umbrales en config):
      - non_finite: algún valor NaN/inf;
      - flat_signal: desvío estándar < ARTIFACT_MIN_STD_UV (electrodo desconectado);
      - clipped: al menos ARTIFACT_MIN_CLIPPED_SAMPLES muestras en el límite
        físico del EDF. Un umbral de pico a pico no sirve: el rango físico
        (como máximo ±248 µV) recorta la señal antes de alcanzarlo.

    Devuelve (is_artifact bool, artifact_reason str) de forma (n_epochs, n_canales).
    Si se cumplen varios criterios, los motivos se concatenan con ';'.
    """
    finite = np.isfinite(epochs_uv).all(axis=-1)
    with np.errstate(invalid="ignore"):
        checks = {
            ARTIFACT_NON_FINITE: ~finite,
            ARTIFACT_FLAT: finite & (np.std(epochs_uv, axis=-1) < config.ARTIFACT_MIN_STD_UV),
            ARTIFACT_CLIPPED: finite & (n_clipped >= config.ARTIFACT_MIN_CLIPPED_SAMPLES),
        }
    reasons = np.full(finite.shape, "", dtype=object)
    for name, mask in checks.items():
        reasons[mask] = np.where(reasons[mask] == "", name, reasons[mask] + ";" + name)
    is_artifact = reasons != ""
    reasons[~is_artifact] = None
    return is_artifact, reasons


def flag_statistical_outliers(features: pd.DataFrame,
                              group_cols: list[str] = ["recording_id", "channel", "stage"]) -> pd.DataFrame:
    """Agrega `outlier_features`: lista de features fuera de [Q1 - 1.5·IQR, Q3 + 1.5·IQR].

    Se calcula dentro de cada grupo registro × canal × etapa: las distribuciones
    difieren legítimamente entre etapas (N3 tiene más delta que REM por definición)
    y entre noches. Con una referencia entre registros, la regla marca personas o
    ganancias del equipo, no epochs raros. Excluye artefactos del cálculo de
    cuartiles. Solo informativo.
    """
    flags = pd.DataFrame(False, index=features.index, columns=IQR_FEATURES)
    clean = ~features["is_artifact"]
    for col in IQR_FEATURES:
        values = features[col]
        if col.endswith(_LOG_SCALE_SUFFIXES):
            values = np.log10(values.where(values > 0))
        # Cuartiles por grupo usando solo filas limpias; luego se asignan a todas las filas del grupo.
        quartiles = (values[clean]
                     .groupby([features.loc[clean, c] for c in group_cols])
                     .quantile([0.25, 0.75]).unstack())
        quartiles.columns = ["q1", "q3"]
        bounds = features[group_cols].join(quartiles, on=group_cols)
        iqr = bounds["q3"] - bounds["q1"]
        flags[col] = (values < bounds["q1"] - IQR_FACTOR * iqr) | (values > bounds["q3"] + IQR_FACTOR * iqr)
    out = features.copy()
    out["outlier_features"] = flags.apply(lambda row: ";".join(flags.columns[row]) or None, axis=1)
    out["is_stat_outlier"] = out["outlier_features"].notna()
    return out


def count_clipped_samples(epochs_uv: np.ndarray, low_uv: float, high_uv: float, tolerance_uv: float) -> np.ndarray:
    """Cuenta, por epoch, las muestras en el límite físico del EDF (saturación del conversor).

    `epochs_uv` debe ser la señal SIN filtrar de un canal, (n_epochs, n_muestras): el filtro
    redondea las mesetas. `tolerance_uv` es un paso de cuantización.
    """
    return ((epochs_uv >= high_uv - tolerance_uv) | (epochs_uv <= low_uv + tolerance_uv)).sum(axis=-1)


def data_profile(table: pd.DataFrame) -> pd.DataFrame:
    """Perfil por columna numérica: nulos, infinitos, ceros, negativos y rango."""
    numeric = table.select_dtypes("number")
    return pd.DataFrame({
        "nulos": numeric.isna().sum(),
        "infinitos": np.isinf(numeric).sum(),
        "ceros": (numeric == 0).sum(),
        "negativos": (numeric < 0).sum(),
        "mínimo": numeric.min(),
        "mediana": numeric.median(),
        "máximo": numeric.max(),
    })


def missing_data_report(epochs: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Resume datos faltantes y exclusiones con su causa y la decisión tomada.

    `epochs`: tabla de epochs de todos los registros (salida de preprocessing,
    con has_signal / in_analysis_window). `features`: tabla tidy.
    Porcentajes sobre el total de epochs del hipnograma o de filas de features.
    """
    n_epochs, n_rows = len(epochs), len(features)
    window = epochs["in_analysis_window"] & epochs["has_signal"]
    items = [
        ("epoch", "Epoch del hipnograma sin señal EEG", (~epochs["has_signal"]).sum(), n_epochs,
         "El hipnograma se extiende más allá del final del PSG",
         "Excluido: no hay señal de la cual extraer features"),
        ("epoch", "Fuera de la ventana nocturna (vigilia y siestas)", (epochs["has_signal"] & ~epochs["in_analysis_window"]).sum(), n_epochs,
         f"Registros de ~20 h; se conservan {config.WAKE_PADDING_MIN} min de W antes/después del sueño "
         f"nocturno ({config.NIGHT_START_HOUR}:00-{config.NIGHT_END_HOUR}:00)",
         "Excluido por diseño (no es un faltante): evita que W y las siestas diurnas dominen el análisis"),
        ("epoch", "Etapa desconocida ('?') en ventana", (window & (epochs["stage"] == "UNK")).sum(), n_epochs,
         "El experto no asignó etapa",
         "Se conserva con etapa UNK; no se imputa; se excluye de comparaciones entre etapas"),
        ("epoch", "Movement time en ventana", (window & (epochs["stage"] == "MOVE")).sum(), n_epochs,
         "Movimiento corporal: el experto no pudo asignar etapa",
         "Se conserva con etapa MOVE; no se imputa; se excluye de comparaciones entre etapas"),
        ("epoch", "Etapa sin mapeo", epochs["stage"].isna().sum(), n_epochs,
         "Descripción del hipnograma no contemplada en STAGE_MAPPING",
         "Debe ser 0 (la validación de cada registro lo impide)"),
    ]
    feature_cols = [c for c in features.columns if c.endswith(("_abs", "_rel")) or c in
                    ("total_power", "sample_entropy", "spectral_entropy", "rms", "peak_to_peak")]
    for col in feature_cols:
        items.append(("feature", f"{col} = NaN", features[col].isna().sum(), n_rows,
                      "No calculable (señal plana o no finita, o SampEn sin coincidencias)",
                      "Se deja NaN (no se imputa); la fila conserva su flag de artefacto"))
    report = pd.DataFrame(items, columns=["level", "item", "count", "denominator", "cause", "decision"])
    report["percent"] = (100 * report["count"] / report["denominator"]).round(2)
    return report[["level", "item", "count", "denominator", "percent", "cause", "decision"]]
