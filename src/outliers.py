"""Diagnóstico de outliers: compara reglas univariadas y detectores multivariados.

Aplica al EEG los métodos de la clase 05 (z-score, Tukey, Anderson-Darling, k-means, k-NN,
LOF, Isolation Forest). Es un análisis exploratorio: no modifica ninguna tabla ni elimina filas.

Todos los detectores multivariados marcan la misma proporción de filas para poder
compararlos. DBSCAN queda afuera porque no da un puntaje continuo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from src import config, quality

GROUP_BETWEEN = ["channel", "sleep_stage"]  # compara registros entre sí
METRICS = ["sample_entropy", "spectral_entropy", "delta_rel", "alpha_rel"]
MULTIVARIATE_FEATURES = [f"{band}_abs" for band in config.FREQ_BANDS] + [
    "sample_entropy", "spectral_entropy", "peak_to_peak"]
_LOG_FEATURES = [c for c in MULTIVARIATE_FEATURES if c.endswith("_abs")] + ["peak_to_peak"]
_ANDERSON_SAMPLE = 5_000


def _plain(tidy: pd.DataFrame) -> pd.DataFrame:
    """Copia con claves de texto, para agrupar sin combinaciones vacías de categorías."""
    return tidy.astype({"sleep_stage": str, "channel": str, "recording_id": str})


def compare_univariate_rules(tidy: pd.DataFrame, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Por variable: forma de la distribución y % de filas marcadas por cada regla.

    Los z se calculan por canal × etapa, sobre log10 en las potencias. Con colas pesadas,
    % |z| > 3 se aparta del 0.27 % esperado bajo normalidad. El estadístico A² usa una
    submuestra: con todas las filas cualquier desvío mínimo rechaza la normalidad.
    """
    data = _plain(tidy[~tidy["is_artifact"] & tidy["sleep_stage"].isin(config.SCORED_STAGES)])
    groups = [data[c] for c in GROUP_BETWEEN]
    rows = []
    for column in quality.IQR_FEATURES:
        values = np.log10(data[column]) if column.endswith(quality._LOG_SCALE_SUFFIXES) else data[column]
        grouped = values.groupby(groups)
        z = (values - grouped.transform("mean")) / grouped.transform("std")
        q1, q3 = grouped.transform(lambda s: s.quantile(0.25)), grouped.transform(lambda s: s.quantile(0.75))
        iqr = q3 - q1
        rows.append({
            "variable": column,
            "asimetría": stats.skew(z),
            "curtosis": stats.kurtosis(z),
            "A²": stats.anderson(z.sample(_ANDERSON_SAMPLE, random_state=seed), dist="norm").statistic,
            "% |z| > 3": 100 * (z.abs() > 3).mean(),
            "% Tukey 1.5": 100 * ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).mean(),
            "% Tukey 3": 100 * ((values < q1 - 3 * iqr) | (values > q3 + 3 * iqr)).mean(),
        })
    return pd.DataFrame(rows).set_index("variable")


def tukey_rate_by_recording(tidy: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Compara el Tukey de la tabla (dentro del registro) con una referencia entre registros.

    Devuelve (máscara entre registros, % de filas marcadas por registro y grupo).
    """
    data = _plain(tidy)
    between = quality.flag_statistical_outliers(data, GROUP_BETWEEN)["is_stat_outlier"]
    rates = (pd.DataFrame({"Entre registros": between, "Dentro del registro": data["is_stat_outlier"]})
             .groupby(data["recording_id"]).mean() * 100)
    return between, rates


def robust_zscores(tidy: pd.DataFrame) -> np.ndarray:
    """Z robusto por registro × canal: cada epoch se mide contra su propia noche."""
    data = _plain(tidy)
    x = data[MULTIVARIATE_FEATURES].copy()
    x[_LOG_FEATURES] = np.log10(x[_LOG_FEATURES])
    grouped = x.groupby([data["recording_id"], data["channel"]])
    iqr = (grouped.transform(lambda s: s.quantile(0.75)) - grouped.transform(lambda s: s.quantile(0.25))) / 1.349
    return ((x - grouped.transform("median")) / iqr).to_numpy()


def multivariate_scores(tidy: pd.DataFrame, k_neighbors: int = 10, n_clusters: int = 10,
                        seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Puntaje de cada detector (mayor = más anómalo)."""
    z = robust_zscores(tidy)
    kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=seed).fit(z)
    distances, _ = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(z).kneighbors(z)
    return pd.DataFrame({
        "Distancia al centro": np.linalg.norm(z - z.mean(axis=0), axis=1),
        "k-means": np.linalg.norm(z - kmeans.cluster_centers_[kmeans.labels_], axis=1),
        "k-NN": distances[:, 1:].mean(axis=1),
        "LOF": -LocalOutlierFactor(n_neighbors=k_neighbors).fit(z).negative_outlier_factor_,
        "Isolation Forest": -IsolationForest(n_estimators=200, random_state=seed).fit(z).score_samples(z),
    }, index=tidy.index)


def flag_top(scores: pd.DataFrame, fraction: float = config.OUTLIER_TOP_FRACTION) -> pd.DataFrame:
    """Marca en cada detector la misma proporción de filas con mayor puntaje."""
    return scores >= scores.quantile(1 - fraction)


def overlap(flags: pd.DataFrame) -> pd.DataFrame:
    """Índice de Jaccard entre los conjuntos de filas marcadas por cada detector."""
    return pd.DataFrame({a: {b: (flags[a] & flags[b]).sum() / (flags[a] | flags[b]).sum() for b in flags}
                         for a in flags})


def reference_auc(scores: pd.DataFrame, references: dict[str, pd.Series]) -> pd.DataFrame:
    """ROC-AUC de cada detector contra referencias parciales (0.5 = azar, 1 = perfecto)."""
    return pd.DataFrame({name: {detector: roc_auc_score(reference, scores[detector]) for detector in scores}
                         for name, reference in references.items()})


def exclusion_sensitivity(tidy: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    """Cuánto cambiarían las medianas por etapa y canal si se excluyera cada conjunto.

    Devuelve filas excluidas y el mayor cambio relativo (%) entre etapas y canales por métrica.
    """
    scored = tidy["sleep_stage"].isin(config.SCORED_STAGES)
    data = _plain(tidy[scored])
    base = data.groupby(GROUP_BETWEEN)[METRICS].median()
    rows = []
    for name, mask in masks.items():
        keep = ~mask[scored]
        change = (data[keep].groupby(GROUP_BETWEEN)[METRICS].median() / base - 1).abs().max() * 100
        rows.append({"criterio": name, "filas excluidas": int((~keep).sum()),
                     "% excluido": 100 * (~keep).mean(), **change.add_prefix("Δ% ")})
    return pd.DataFrame(rows).set_index("criterio")
