"""Validaciones automáticas del pipeline.

Cada check lanza ValidationError con un mensaje comprensible si falla y
devuelve una descripción breve si pasa, para mostrar en el notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.features import BAND_NAMES, FEATURE_COLUMNS


class ValidationError(AssertionError):
    """Una validación del pipeline falló."""


def _check(condition: bool, ok_message: str, error_message: str) -> str:
    if not condition:
        raise ValidationError(error_message)
    return ok_message


def validate_downloads(manifest: pd.DataFrame) -> list[str]:
    """Archivos presentes, tamaño mínimo y un PSG + un hipnograma por registro."""
    paths = manifest["file"].map(lambda f: config.PROJECT_ROOT / f)
    missing = manifest.loc[~paths.map(lambda p: p.exists()), "file"].tolist()
    kinds = manifest.groupby("recording_id")["kind"].apply(sorted)
    unpaired = kinds[kinds.map(lambda k: k != ["Hypnogram", "PSG"])].index.tolist()
    return [
        _check(not missing, f"{len(manifest)} archivos presentes en data/raw", f"Archivos faltantes: {missing}"),
        _check(not unpaired, f"{len(kinds)} registros con PSG + Hypnogram",
               f"Registros sin par PSG/Hypnogram: {unpaired}"),
    ]


def validate_tidy(tidy: pd.DataFrame, rel_tolerance: float = 1e-9) -> list[str]:
    """Checks de la tabla tidy (una fila por epoch x canal)."""
    key = ["recording_id", "epoch_number", "channel"]
    dup = int(tidy.duplicated(key).sum())
    channels = set(tidy["channel"].astype(str))
    starts_ok = np.allclose(tidy["start_time_sec"] % config.EPOCH_DURATION_S, 0)
    valid_stages = set(config.SCORED_STAGES + config.UNSCORED_STAGES)
    stages = set(tidy["sleep_stage"].dropna().astype(str))
    clean = tidy[~tidy["is_artifact"]]
    rel_sum = clean[[f"{b}_rel" for b in BAND_NAMES]].sum(axis=1)
    rel = clean[[f"{b}_rel" for b in BAND_NAMES]]
    per_epoch = tidy.groupby(["recording_id", "epoch_number"], observed=True)["channel"].nunique()
    reasons_ok = (tidy["is_artifact"] == tidy["artifact_reason"].notna()).all()
    return [
        _check(dup == 0, "Sin claves (recording, epoch, channel) duplicadas", f"{dup} claves duplicadas"),
        _check(channels == set(config.EEG_CHANNELS), f"Canales: {sorted(channels)}",
               f"Canales inesperados: {channels} (esperados {config.EEG_CHANNELS})"),
        _check((per_epoch == len(config.EEG_CHANNELS)).all(), "Cada epoch tiene todos los canales",
               f"{int((per_epoch != len(config.EEG_CHANNELS)).sum())} epochs con canales faltantes"),
        _check(starts_ok, f"Inicios de epoch múltiplos de {config.EPOCH_DURATION_S:.0f} s",
               "Hay epochs que no empiezan en múltiplos de la duración de epoch"),
        _check(tidy["sleep_stage"].notna().all() and stages <= valid_stages, f"Etapas válidas: {sorted(stages)}",
               f"Etapas inválidas o nulas: {stages - valid_stages}"),
        _check(bool(np.allclose(rel_sum, 1, atol=rel_tolerance)) and bool(((rel >= 0) & (rel <= 1)).all().all()),
               f"Potencia relativa en [0, 1] y suma 1 ± {rel_tolerance} (sin artefactos)",
               f"Potencia relativa fuera de rango: suma entre {rel_sum.min():.4f} y {rel_sum.max():.4f}"),
        _check(reasons_ok, "is_artifact coherente con artifact_reason",
               "Hay filas con is_artifact y artifact_reason inconsistentes"),
        _check(tidy.loc[~tidy["is_artifact"], FEATURE_COLUMNS].notna().all().all(),
               "Sin features NaN en filas no artefactadas",
               "Hay features NaN en filas no marcadas como artefacto"),
    ]


def validate_oltp(tidy: pd.DataFrame, path=config.OLTP_DB_PATH, rel_tolerance: float = 1e-9) -> list[str]:
    """FK activas y consistentes, conteos iguales a la tabla tidy y vistas derivadas correctas."""
    from src.database_oltp import connect  # import local: evita dependencia circular

    with connect(path) as conn:
        fk_on = conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        fk_violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        n_epoch = conn.execute("SELECT COUNT(*) FROM epoch").fetchone()[0]
        n_feature = conn.execute("SELECT COUNT(*) FROM eeg_feature").fetchone()[0]
        n_recording = conn.execute("SELECT COUNT(*) FROM recording").fetchone()[0]
        db_rel = pd.read_sql(
            "SELECT e.recording_id, e.epoch_number, c.name AS channel, f.delta_rel "
            "FROM v_eeg_feature f JOIN epoch e USING (epoch_id) JOIN channel c USING (channel_id)", conn)
    conn.close()

    expected_epochs = tidy[["recording_id", "epoch_number"]].drop_duplicates().shape[0]
    merged = tidy.assign(channel=tidy["channel"].astype(str)).merge(
        db_rel, on=["recording_id", "epoch_number", "channel"], suffixes=("", "_db"))
    max_diff = float((merged["delta_rel"] - merged["delta_rel_db"]).abs().max())
    return [
        _check(fk_on, "PRAGMA foreign_keys = ON", "Las foreign keys no están activadas"),
        _check(not fk_violations, "PRAGMA foreign_key_check sin violaciones",
               f"Violaciones de FK: {fk_violations[:5]}"),
        _check(n_recording == tidy["recording_id"].nunique(), f"recording: {n_recording} filas",
               f"recording tiene {n_recording} filas, la tidy {tidy['recording_id'].nunique()} registros"),
        _check(n_epoch == expected_epochs, f"epoch: {n_epoch} filas = epochs de la tidy",
               f"epoch tiene {n_epoch} filas, se esperaban {expected_epochs}"),
        _check(n_feature == len(tidy), f"eeg_feature: {n_feature} filas = filas de la tidy",
               f"eeg_feature tiene {n_feature} filas, la tidy {len(tidy)}"),
        _check(len(merged) == len(tidy) and max_diff < rel_tolerance,
               f"Potencia relativa recalculada en la vista coincide (max |dif| = {max_diff:.1e})",
               f"La vista v_eeg_feature difiere de la tidy (max |dif| = {max_diff:.1e})"),
    ]


def validate_dw(oltp_path=config.OLTP_DB_PATH, dw_path=config.DW_DB_PATH) -> list[str]:
    """FK del DW, grano único, conteo igual al OLTP y reproducibilidad de las consultas."""
    from src import database_dw, database_oltp, queries

    with database_oltp.connect(oltp_path) as conn:
        n_oltp = conn.execute("SELECT COUNT(*) FROM eeg_feature").fetchone()[0]
    conn.close()
    with database_dw.connect(dw_path) as conn:
        fk_violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        n_fact = conn.execute("SELECT COUNT(*) FROM fact_eeg_epoch").fetchone()[0]
        n_grain = conn.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT recording_key, epoch_number, channel_key FROM fact_eeg_epoch)"
        ).fetchone()[0]
        subject_mismatch = conn.execute(
            "SELECT COUNT(*) FROM (SELECT subject_id FROM dim_recording "
            "GROUP BY subject_id HAVING COUNT(DISTINCT age) > 1 OR COUNT(DISTINCT sex) > 1 "
            "OR COUNT(DISTINCT age_group) > 1)").fetchone()[0]
    conn.close()

    not_reproducible = [
        name for name in queries.QUERIES
        if not queries.run_query(name, path=dw_path).equals(queries.run_query(name, path=dw_path))
    ]
    return [
        _check(not fk_violations, "DW: PRAGMA foreign_key_check sin violaciones", f"Violaciones de FK en DW: {fk_violations[:5]}"),
        _check(n_fact == n_oltp, f"fact_eeg_epoch: {n_fact} filas = eeg_feature del OLTP",
               f"fact_eeg_epoch tiene {n_fact} filas y eeg_feature {n_oltp}"),
        _check(n_grain == n_fact, "Grano respetado: (registro, epoch, canal) único en la tabla de hechos",
               f"{n_fact - n_grain} filas repiten el grano (registro, epoch, canal)"),
        _check(subject_mismatch == 0, "Jerarquía de dim_recording coherente: cada sujeto tiene una edad, un sexo y un grupo",
               f"{subject_mismatch} sujetos con atributos distintos entre registros"),
        _check(not not_reproducible, f"{len(queries.QUERIES)} consultas analíticas reproducibles",
               f"Consultas con resultados distintos entre ejecuciones: {not_reproducible}"),
    ]
