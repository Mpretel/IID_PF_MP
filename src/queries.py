"""Consultas analíticas sobre el Data Warehouse (sleep_dw.sqlite).

Todas las consultas comparten dos parámetros:
  :include_artifacts  1 = incluir epochs marcados como artefacto, 0 = excluirlos
y solo consideran etapas puntuadas (is_scored = 1), salvo las de calidad
(conteos y porcentaje de artefactos), que describen todo el conjunto.

Los resultados son descriptivos: una diferencia de medias no implica
significancia estadística (no se hacen tests ni se modela la dependencia
entre epochs consecutivos del mismo sujeto).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src import config

_FROM_STAR = """
FROM fact_eeg_epoch f
JOIN dim_subject     su ON su.subject_key   = f.subject_key
JOIN dim_recording   r  ON r.recording_key  = f.recording_key
JOIN dim_channel     c  ON c.channel_key    = f.channel_key
JOIN dim_sleep_stage st ON st.stage_key     = f.stage_key
"""
_ANALYSIS_FILTER = "WHERE st.is_scored = 1 AND (:include_artifacts = 1 OR f.is_artifact = 0)"

QUERIES: dict[str, str] = {
    # 1. Sample Entropy promedio por etapa (y canal: los canales no se mezclan).
    "sample_entropy_by_stage": f"""
        SELECT st.stage_code, c.short_name AS channel,
               COUNT(f.sample_entropy) AS n, AVG(f.sample_entropy) AS mean_sample_entropy,
               MIN(f.sample_entropy) AS min_value, MAX(f.sample_entropy) AS max_value
        {_FROM_STAR} {_ANALYSIS_FILTER}
        GROUP BY st.stage_code, st.stage_order, c.short_name
        ORDER BY st.stage_order, channel""",

    # 2. Spectral Entropy promedio por etapa.
    "spectral_entropy_by_stage": f"""
        SELECT st.stage_code, c.short_name AS channel,
               COUNT(f.spectral_entropy) AS n, AVG(f.spectral_entropy) AS mean_spectral_entropy,
               MIN(f.spectral_entropy) AS min_value, MAX(f.spectral_entropy) AS max_value
        {_FROM_STAR} {_ANALYSIS_FILTER}
        GROUP BY st.stage_code, st.stage_order, c.short_name
        ORDER BY st.stage_order, channel""",

    # 3. Potencia relativa de cada banda por etapa.
    "relative_band_power_by_stage": f"""
        SELECT st.stage_code, c.short_name AS channel, COUNT(*) AS n,
               AVG(f.delta_rel) AS delta, AVG(f.theta_rel) AS theta, AVG(f.alpha_rel) AS alpha,
               AVG(f.sigma_rel) AS sigma, AVG(f.beta_rel) AS beta
        {_FROM_STAR} {_ANALYSIS_FILTER}
        GROUP BY st.stage_code, st.stage_order, c.short_name
        ORDER BY st.stage_order, channel""",

    # 4. Comparación entre canales (métricas principales, por etapa).
    "channel_comparison": f"""
        SELECT c.short_name AS channel, st.stage_code, COUNT(*) AS n,
               AVG(f.sample_entropy) AS sample_entropy, AVG(f.spectral_entropy) AS spectral_entropy,
               AVG(f.delta_rel) AS delta_rel, AVG(f.alpha_rel) AS alpha_rel,
               AVG(f.total_power) AS total_power, AVG(f.rms) AS rms
        {_FROM_STAR} {_ANALYSIS_FILTER}
        GROUP BY c.short_name, st.stage_code, st.stage_order
        ORDER BY st.stage_order, channel""",

    # 5. Comparación entre sujetos.
    "subject_comparison": f"""
        SELECT su.subject_id, su.age, su.sex, st.stage_code, c.short_name AS channel, COUNT(*) AS n,
               AVG(f.sample_entropy) AS sample_entropy, AVG(f.spectral_entropy) AS spectral_entropy,
               AVG(f.delta_rel) AS delta_rel
        {_FROM_STAR} {_ANALYSIS_FILTER}
        GROUP BY su.subject_id, su.age, su.sex, st.stage_code, st.stage_order, c.short_name
        ORDER BY su.subject_id, st.stage_order, channel""",

    # 6. Cantidad de epochs por etapa (un epoch tiene una fila por canal: se cuentan
    #    pares registro-epoch distintos). Incluye etapas no puntuadas.
    "epochs_by_stage": f"""
        WITH epochs AS (
            SELECT DISTINCT f.recording_key, f.epoch_number, st.stage_code, st.stage_order
            {_FROM_STAR}
        )
        SELECT stage_code, COUNT(*) AS n_epochs,
               ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM epochs), 2) AS percent,
               ROUND(COUNT(*) * {config.EPOCH_DURATION_S} / 3600.0, 2) AS hours
        FROM epochs
        GROUP BY stage_code, stage_order
        ORDER BY stage_order""",

    # 7. Porcentaje de observaciones (epoch x canal) marcadas como artefacto.
    "artifact_percentage": f"""
        SELECT r.recording_id, c.short_name AS channel, COUNT(*) AS n_observations,
               SUM(f.is_artifact) AS n_artifacts,
               ROUND(100.0 * SUM(f.is_artifact) / COUNT(*), 2) AS artifact_percent,
               SUM(f.is_stat_outlier) AS n_stat_outliers,
               ROUND(100.0 * SUM(f.is_stat_outlier) / COUNT(*), 2) AS stat_outlier_percent
        {_FROM_STAR}
        GROUP BY r.recording_id, c.short_name
        ORDER BY r.recording_id, channel""",

    # 8. Evolución temporal: promedios por hora desde el inicio del sueño.
    "temporal_evolution": f"""
        SELECT r.recording_id, c.short_name AS channel,
               CAST(f.hours_since_sleep_onset + 1 AS INTEGER) - 1 AS hour_bin,
               COUNT(*) AS n,
               AVG(f.sample_entropy) AS sample_entropy, AVG(f.spectral_entropy) AS spectral_entropy,
               AVG(f.delta_rel) AS delta_rel,
               ROUND(100.0 * SUM(st.stage_code = 'N3') / COUNT(*), 1) AS pct_n3,
               ROUND(100.0 * SUM(st.stage_code = 'REM') / COUNT(*), 1) AS pct_rem
        {_FROM_STAR} {_ANALYSIS_FILTER}
        GROUP BY r.recording_id, c.short_name, hour_bin
        ORDER BY r.recording_id, channel, hour_bin""",
}

# Datos a nivel fila para gráficos de distribución, hipnograma y dashboard.
OBSERVATIONS_SQL = f"""
SELECT su.subject_id, r.recording_id, r.night, c.short_name AS channel,
       f.epoch_number, f.start_sec, f.hours_since_sleep_onset,
       st.stage_code, st.stage_order, st.is_scored,
       f.delta_abs, f.theta_abs, f.alpha_abs, f.sigma_abs, f.beta_abs,
       f.delta_rel, f.theta_rel, f.alpha_rel, f.sigma_rel, f.beta_rel, f.total_power,
       f.sample_entropy, f.spectral_entropy, f.rms, f.peak_to_peak,
       f.is_artifact, f.artifact_reason, f.is_stat_outlier, f.outlier_features
{_FROM_STAR}
ORDER BY r.recording_id, f.epoch_number, c.short_name
"""


def run_query(name: str, include_artifacts: bool = False, path: Path = config.DW_DB_PATH) -> pd.DataFrame:
    """Ejecuta una consulta nombrada de QUERIES y devuelve un DataFrame."""
    if name not in QUERIES:
        raise KeyError(f"Consulta desconocida '{name}'. Disponibles: {list(QUERIES)}")
    with sqlite3.connect(path) as conn:
        result = pd.read_sql(QUERIES[name], conn, params={"include_artifacts": int(include_artifacts)})
    conn.close()
    return result


def load_observations(path: Path = config.DW_DB_PATH) -> pd.DataFrame:
    """Todas las observaciones (epoch x canal) con su contexto, desde el DW."""
    with sqlite3.connect(path) as conn:
        obs = pd.read_sql(OBSERVATIONS_SQL, conn)
    conn.close()
    obs[["is_artifact", "is_scored", "is_stat_outlier"]] = obs[["is_artifact", "is_scored", "is_stat_outlier"]].astype(bool)
    return obs
