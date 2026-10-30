"""Data Warehouse en esquema estrella: data/database/sleep_dw.sqlite.

    SQLite OLTP  --(ETL: extract SQL -> transform pandas -> load)-->  SQLite DW

Grano de FACT_EEG_EPOCH: una observación de features EEG correspondiente a
un epoch de 30 segundos, un canal, un sujeto y un registro.

                      dim_subject
                           │
        dim_recording ── fact_eeg_epoch ── dim_channel
                           │
                    dim_sleep_stage

A diferencia del OLTP, acá la redundancia es deliberada: se materializan
potencias total/relativas, tiempos y flags de outlier para que las consultas
analíticas sean simples (un JOIN por dimensión) y rápidas.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from src import config, database_oltp, quality
from src.database_oltp import _insert
from src.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE dim_subject (
    subject_key   INTEGER PRIMARY KEY,          -- surrogate key
    subject_id    INTEGER NOT NULL UNIQUE,      -- clave natural (Sleep-EDF)
    age           INTEGER NOT NULL,
    sex           TEXT    NOT NULL,
    age_group     TEXT    NOT NULL
);

CREATE TABLE dim_recording (
    recording_key         INTEGER PRIMARY KEY,
    recording_id          TEXT    NOT NULL UNIQUE,
    night                 INTEGER NOT NULL,
    start_datetime        TEXT    NOT NULL,
    lights_off            TEXT,
    sampling_frequency_hz REAL    NOT NULL,
    psg_duration_h        REAL    NOT NULL,
    analyzed_duration_h   REAL    NOT NULL
);

CREATE TABLE dim_channel (
    channel_key   INTEGER PRIMARY KEY,
    channel_name  TEXT NOT NULL UNIQUE,
    short_name    TEXT NOT NULL,
    location      TEXT NOT NULL
);

CREATE TABLE dim_sleep_stage (
    stage_key            INTEGER PRIMARY KEY,
    original_description TEXT    NOT NULL UNIQUE,   -- etiqueta R&K del hipnograma
    stage_code           TEXT    NOT NULL,          -- W, N1, N2, N3, REM, MOVE, UNK
    stage_name           TEXT    NOT NULL,
    is_scored            INTEGER NOT NULL,
    stage_order          INTEGER NOT NULL
);

CREATE TABLE fact_eeg_epoch (
    fact_id                 INTEGER PRIMARY KEY,
    subject_key             INTEGER NOT NULL REFERENCES dim_subject (subject_key),
    recording_key           INTEGER NOT NULL REFERENCES dim_recording (recording_key),
    channel_key             INTEGER NOT NULL REFERENCES dim_channel (channel_key),
    stage_key               INTEGER NOT NULL REFERENCES dim_sleep_stage (stage_key),
    epoch_number            INTEGER NOT NULL,       -- dimensión degenerada
    start_sec               REAL    NOT NULL,       -- desde el inicio del registro
    hours_since_sleep_onset REAL    NOT NULL,       -- negativo = vigilia previa al sueño
    delta_abs REAL, theta_abs REAL, alpha_abs REAL, sigma_abs REAL, beta_abs REAL,
    delta_rel REAL, theta_rel REAL, alpha_rel REAL, sigma_rel REAL, beta_rel REAL,
    total_power      REAL,
    sample_entropy   REAL,
    spectral_entropy REAL,
    rms              REAL,
    peak_to_peak     REAL,
    is_artifact      INTEGER NOT NULL,
    artifact_reason  TEXT,
    is_stat_outlier  INTEGER NOT NULL,
    outlier_features TEXT,
    UNIQUE (recording_key, epoch_number, channel_key)
);

CREATE INDEX idx_fact_subject   ON fact_eeg_epoch (subject_key);
CREATE INDEX idx_fact_channel   ON fact_eeg_epoch (channel_key);
CREATE INDEX idx_fact_stage     ON fact_eeg_epoch (stage_key);
"""

# Extract: una consulta sobre el OLTP con las vistas que calculan atributos derivados.
OLTP_EXTRACT_SQL = """
SELECT r.subject_id, s.age, s.sex,
       r.recording_id, r.night, r.start_datetime, r.lights_off, r.sampling_frequency_hz,
       r.psg_duration_s, r.analysis_start_sec, r.analysis_end_sec,
       c.name AS channel, c.location,
       e.epoch_number, e.start_sec, e.stage_original, e.stage_code AS stage, e.is_scored,
       f.delta_abs, f.theta_abs, f.alpha_abs, f.sigma_abs, f.beta_abs,
       f.delta_rel, f.theta_rel, f.alpha_rel, f.sigma_rel, f.beta_rel, f.total_power,
       f.sample_entropy, f.spectral_entropy, f.rms, f.peak_to_peak,
       f.is_artifact, f.artifact_reason
FROM v_eeg_feature f
JOIN v_epoch   e ON e.epoch_id   = f.epoch_id
JOIN channel   c ON c.channel_id = f.channel_id
JOIN recording r ON r.recording_id = e.recording_id
JOIN subject   s ON s.subject_id = r.subject_id
"""

AGE_BINS = [0, 40, 60, 80, 120]
AGE_LABELS = ["<40", "40-59", "60-79", "80+"]


def connect(path: Path = config.DW_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def extract_from_oltp(oltp_path: Path = config.OLTP_DB_PATH) -> pd.DataFrame:
    with database_oltp.connect(oltp_path) as conn:
        data = pd.read_sql(OLTP_EXTRACT_SQL, conn)
    conn.close()
    data["is_artifact"] = data["is_artifact"].astype(bool)
    return data


def _with_key(df: pd.DataFrame, key: str, order_by: list[str]) -> pd.DataFrame:
    """Asigna surrogate keys 1..n en un orden determinístico (reproducible)."""
    df = df.sort_values(order_by).reset_index(drop=True)
    df.insert(0, key, range(1, len(df) + 1))
    return df


def transform(data: pd.DataFrame, oltp_path: Path = config.OLTP_DB_PATH) -> dict[str, pd.DataFrame]:
    """Construye dimensiones y hechos a partir de la extracción del OLTP."""
    dim_subject = _with_key(
        data[["subject_id", "age", "sex"]].drop_duplicates()
        .assign(age_group=lambda d: pd.cut(d["age"], AGE_BINS, right=False, labels=AGE_LABELS).astype(str)),
        "subject_key", ["subject_id"])

    rec = data.drop_duplicates("recording_id")
    dim_recording = _with_key(pd.DataFrame({
        "recording_id": rec["recording_id"], "night": rec["night"], "start_datetime": rec["start_datetime"],
        "lights_off": rec["lights_off"], "sampling_frequency_hz": rec["sampling_frequency_hz"],
        "psg_duration_h": rec["psg_duration_s"] / 3600,
        "analyzed_duration_h": (rec["analysis_end_sec"] - rec["analysis_start_sec"]) / 3600,
    }), "recording_key", ["recording_id"])

    dim_channel = _with_key(
        data[["channel", "location"]].drop_duplicates()
        .rename(columns={"channel": "channel_name"})
        .assign(short_name=lambda d: d["channel_name"].str.replace("EEG ", "", regex=False)),
        "channel_key", ["channel_name"])[["channel_key", "channel_name", "short_name", "location"]]

    # Todas las etiquetas posibles (no solo las observadas), desde el catálogo del OLTP.
    with database_oltp.connect(oltp_path) as conn:
        stages = pd.read_sql(
            "SELECT l.description AS original_description, s.stage_code, s.stage_name, s.is_scored, s.stage_order "
            "FROM hypnogram_label l JOIN sleep_stage s USING (stage_id)", conn)
    conn.close()
    dim_sleep_stage = _with_key(stages, "stage_key", ["stage_order", "original_description"])

    # Hechos: outliers estadísticos dentro de cada noche (dependen de la distribución).
    facts = quality.flag_statistical_outliers(data, group_cols=["recording_id", "channel", "stage"])
    sleep_onset = (facts[facts["stage"].isin(["N1", "N2", "N3", "REM"])]
                   .groupby("recording_id")["start_sec"].min().rename("sleep_onset_sec"))
    facts = (facts
             .merge(sleep_onset, on="recording_id")
             .merge(dim_subject[["subject_key", "subject_id"]], on="subject_id")
             .merge(dim_recording[["recording_key", "recording_id"]], on="recording_id")
             .merge(dim_channel[["channel_key", "channel_name"]], left_on="channel", right_on="channel_name")
             .merge(dim_sleep_stage[["stage_key", "original_description"]],
                    left_on="stage_original", right_on="original_description"))
    if len(facts) != len(data):
        raise ValueError(f"El join con dimensiones cambió la cantidad de filas ({len(data)} -> {len(facts)}).")
    facts["hours_since_sleep_onset"] = (facts["start_sec"] - facts["sleep_onset_sec"]) / 3600
    facts[["is_artifact", "is_stat_outlier"]] = facts[["is_artifact", "is_stat_outlier"]].astype(int)
    fact_cols = (["subject_key", "recording_key", "channel_key", "stage_key", "epoch_number", "start_sec",
                  "hours_since_sleep_onset"] + FEATURE_COLUMNS
                 + ["is_artifact", "artifact_reason", "is_stat_outlier", "outlier_features"])
    fact = facts.sort_values(["recording_key", "epoch_number", "channel_key"])[fact_cols]

    return {"dim_subject": dim_subject, "dim_recording": dim_recording, "dim_channel": dim_channel,
            "dim_sleep_stage": dim_sleep_stage, "fact_eeg_epoch": fact}


def load(tables: dict[str, pd.DataFrame], path: Path = config.DW_DB_PATH) -> None:
    """(Re)crea el DW y carga dimensiones y luego hechos (orden requerido por las FK)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        for name in ["dim_subject", "dim_recording", "dim_channel", "dim_sleep_stage", "fact_eeg_epoch"]:
            _insert(conn, name, tables[name])
    conn.close()


def build_dw(oltp_path: Path = config.OLTP_DB_PATH, dw_path: Path = config.DW_DB_PATH) -> dict[str, int]:
    """ETL completo OLTP -> DW. Devuelve la cantidad de filas por tabla."""
    tables = transform(extract_from_oltp(oltp_path), oltp_path)
    load(tables, dw_path)
    logger.info("DW creado: %s", dw_path.relative_to(config.PROJECT_ROOT).as_posix())
    return {name: len(df) for name, df in tables.items()}
