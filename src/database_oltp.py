"""Base transaccional (OLTP) normalizada en 3NF: data/database/sleep_oltp.sqlite.

Modelo:
    subject 1──N recording 1──N epoch N──1 hypnogram_label N──1 sleep_stage
                                  │
                                  1
                                  │
                                  N
                             eeg_feature N──1 channel

Decisiones de normalización:
  - La etiqueta original del hipnograma ('Sleep stage 4') determina la etapa
    (N3). Guardar ambas en `epoch` crearía la dependencia transitiva
    epoch -> etiqueta -> etapa; por eso existe `hypnogram_label`.
  - start_sec = epoch_number * 30 no se guarda: dependería solo de parte de la
    clave candidata (recording_id, epoch_number). Se calcula en `v_epoch`.
  - Potencia total y relativa se derivan de las potencias absolutas
    (total = suma de bandas disjuntas, rel = abs/total). No se guardan; se
    calculan en `v_eeg_feature`.
  - Los outliers estadísticos dependen de toda la distribución, no de la fila:
    no son un atributo de la observación y se calculan en el ETL al DW.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

STAGE_NAMES = {
    "W": "Wake", "N1": "NREM 1", "N2": "NREM 2", "N3": "NREM 3 (slow-wave sleep)",
    "REM": "Rapid eye movement", "MOVE": "Movement time (unscored)", "UNK": "Unknown (unscored)",
}
CHANNEL_LOCATIONS = {
    "EEG Fpz-Cz": "Bipolar frontal-central (Fpz referenced to Cz)",
    "EEG Pz-Oz": "Bipolar parietal-occipital (Pz referenced to Oz)",
}
STORED_FEATURES = ["delta_abs", "theta_abs", "alpha_abs", "sigma_abs", "beta_abs",
                   "sample_entropy", "spectral_entropy", "rms", "peak_to_peak"]

SCHEMA_SQL = f"""
CREATE TABLE subject (
    subject_id      INTEGER PRIMARY KEY,                      -- ID de Sleep-EDF (SC4ss..)
    age             INTEGER NOT NULL CHECK (age BETWEEN 0 AND 120),
    sex             TEXT    NOT NULL CHECK (sex IN ('F', 'M'))
);

CREATE TABLE recording (
    recording_id          TEXT    PRIMARY KEY,                -- p. ej. 'SC4001'
    subject_id            INTEGER NOT NULL REFERENCES subject (subject_id),
    night                 INTEGER NOT NULL CHECK (night IN (1, 2)),
    start_datetime        TEXT    NOT NULL,                   -- ISO 8601, del header EDF
    lights_off            TEXT,                               -- HH:MM:SS (SC-subjects.xls)
    sampling_frequency_hz REAL    NOT NULL CHECK (sampling_frequency_hz > 0),
    psg_duration_s        REAL    NOT NULL CHECK (psg_duration_s > 0),
    analysis_start_sec    REAL    NOT NULL,
    analysis_end_sec      REAL    NOT NULL,
    psg_file              TEXT    NOT NULL UNIQUE,
    hypnogram_file        TEXT    NOT NULL UNIQUE,
    UNIQUE (subject_id, night),
    CHECK (analysis_start_sec >= 0 AND analysis_end_sec <= psg_duration_s
           AND analysis_start_sec < analysis_end_sec)
);

CREATE TABLE channel (
    channel_id      INTEGER PRIMARY KEY,
    name            TEXT    NOT NULL UNIQUE,                  -- etiqueta del EDF
    channel_type    TEXT    NOT NULL CHECK (channel_type IN ('EEG')),
    location        TEXT    NOT NULL
);

CREATE TABLE sleep_stage (
    stage_id        INTEGER PRIMARY KEY,
    stage_code      TEXT    NOT NULL UNIQUE,                  -- W, N1, N2, N3, REM, MOVE, UNK
    stage_name      TEXT    NOT NULL,
    is_scored       INTEGER NOT NULL CHECK (is_scored IN (0, 1)),  -- 0: no es una etapa de sueño
    stage_order     INTEGER NOT NULL UNIQUE
);

CREATE TABLE hypnogram_label (
    label_id        INTEGER PRIMARY KEY,
    description     TEXT    NOT NULL UNIQUE,                  -- texto original del hipnograma (R&K)
    stage_id        INTEGER NOT NULL REFERENCES sleep_stage (stage_id)
);

CREATE TABLE epoch (
    epoch_id        INTEGER PRIMARY KEY,
    recording_id    TEXT    NOT NULL REFERENCES recording (recording_id),
    epoch_number    INTEGER NOT NULL CHECK (epoch_number >= 0),   -- índice desde el inicio del EDF
    label_id        INTEGER NOT NULL REFERENCES hypnogram_label (label_id),
    UNIQUE (recording_id, epoch_number)
);

CREATE TABLE eeg_feature (
    epoch_id         INTEGER NOT NULL REFERENCES epoch (epoch_id),
    channel_id       INTEGER NOT NULL REFERENCES channel (channel_id),
    -- Potencias absolutas [µV²] (NULL = no calculable, ver artifact_reason)
    delta_abs        REAL CHECK (delta_abs >= 0),
    theta_abs        REAL CHECK (theta_abs >= 0),
    alpha_abs        REAL CHECK (alpha_abs >= 0),
    sigma_abs        REAL CHECK (sigma_abs >= 0),
    beta_abs         REAL CHECK (beta_abs >= 0),
    sample_entropy   REAL CHECK (sample_entropy >= 0),
    spectral_entropy REAL CHECK (spectral_entropy BETWEEN 0 AND 1),
    rms              REAL CHECK (rms >= 0),                    -- [µV]
    peak_to_peak     REAL CHECK (peak_to_peak >= 0),           -- [µV]
    is_artifact      INTEGER NOT NULL CHECK (is_artifact IN (0, 1)),
    artifact_reason  TEXT,
    PRIMARY KEY (epoch_id, channel_id),
    CHECK ((is_artifact = 0 AND artifact_reason IS NULL) OR (is_artifact = 1 AND artifact_reason IS NOT NULL))
);

-- Índices para las FK más consultadas (las PK/UNIQUE ya están indexadas).
CREATE INDEX idx_recording_subject ON recording (subject_id);
CREATE INDEX idx_epoch_label       ON epoch (label_id);
CREATE INDEX idx_feature_channel   ON eeg_feature (channel_id);

-- Vistas con atributos derivados (no almacenados para respetar 3NF).
CREATE VIEW v_epoch AS
SELECT e.epoch_id, e.recording_id, e.epoch_number,
       e.epoch_number * {config.EPOCH_DURATION_S} AS start_sec,
       s.stage_code, s.is_scored, l.description AS stage_original
FROM epoch e
JOIN hypnogram_label l ON l.label_id = e.label_id
JOIN sleep_stage s     ON s.stage_id = l.stage_id;

CREATE VIEW v_eeg_feature AS
SELECT f.*,
       f.delta_abs + f.theta_abs + f.alpha_abs + f.sigma_abs + f.beta_abs AS total_power,
       f.delta_abs / (f.delta_abs + f.theta_abs + f.alpha_abs + f.sigma_abs + f.beta_abs) AS delta_rel,
       f.theta_abs / (f.delta_abs + f.theta_abs + f.alpha_abs + f.sigma_abs + f.beta_abs) AS theta_rel,
       f.alpha_abs / (f.delta_abs + f.theta_abs + f.alpha_abs + f.sigma_abs + f.beta_abs) AS alpha_rel,
       f.sigma_abs / (f.delta_abs + f.theta_abs + f.alpha_abs + f.sigma_abs + f.beta_abs) AS sigma_rel,
       f.beta_abs  / (f.delta_abs + f.theta_abs + f.alpha_abs + f.sigma_abs + f.beta_abs) AS beta_rel
FROM eeg_feature f;
"""


def connect(path: Path = config.OLTP_DB_PATH) -> sqlite3.Connection:
    """Abre la base con foreign keys activadas (en SQLite están desactivadas por defecto)."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _records(df: pd.DataFrame) -> list[tuple]:
    """Filas como tuplas nativas de Python; NaN -> NULL."""
    clean = df.astype(object).where(df.notna(), None)
    return [tuple(v.item() if isinstance(v, np.generic) else v for v in row)
            for row in clean.itertuples(index=False)]


def _insert(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    cols = ", ".join(df.columns)
    marks = ", ".join("?" * len(df.columns))
    conn.executemany(f"INSERT INTO {table} ({cols}) VALUES ({marks})", _records(df))


def build_oltp(tidy: pd.DataFrame, recordings: pd.DataFrame, path: Path = config.OLTP_DB_PATH) -> Path:
    """(Re)crea la base OLTP desde las tablas procesadas.

    La base se reconstruye completa en cada ejecución: es un artefacto derivado
    de data/processed, y reconstruir garantiza que no queden filas obsoletas.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with connect(path) as conn:
        conn.executescript(SCHEMA_SQL)

        subjects = (recordings[["subject", "age", "sex"]].drop_duplicates()
                    .rename(columns={"subject": "subject_id"}))
        if subjects["subject_id"].duplicated().any():
            raise ValueError("Un mismo sujeto tiene edad/sexo distintos entre noches.")
        _insert(conn, "subject", subjects)

        _insert(conn, "recording", recordings.rename(columns={"subject": "subject_id"})[[
            "recording_id", "subject_id", "night", "start_datetime", "lights_off", "sampling_frequency_hz",
            "psg_duration_s", "analysis_start_sec", "analysis_end_sec", "psg_file", "hypnogram_file"]])

        _insert(conn, "channel", pd.DataFrame({
            "name": config.EEG_CHANNELS, "channel_type": "EEG",
            "location": [CHANNEL_LOCATIONS[c] for c in config.EEG_CHANNELS]}))

        stage_codes = config.SCORED_STAGES + config.UNSCORED_STAGES
        _insert(conn, "sleep_stage", pd.DataFrame({
            "stage_code": stage_codes, "stage_name": [STAGE_NAMES[s] for s in stage_codes],
            "is_scored": [int(s in config.SCORED_STAGES) for s in stage_codes],
            "stage_order": range(1, len(stage_codes) + 1)}))
        stage_ids = dict(conn.execute("SELECT stage_code, stage_id FROM sleep_stage"))
        _insert(conn, "hypnogram_label", pd.DataFrame({
            "description": list(config.STAGE_MAPPING),
            "stage_id": [stage_ids[s] for s in config.STAGE_MAPPING.values()]}))
        label_ids = dict(conn.execute("SELECT description, label_id FROM hypnogram_label"))

        epochs = (tidy[["recording_id", "epoch_number", "stage_original"]]
                  .drop_duplicates(["recording_id", "epoch_number"]))
        _insert(conn, "epoch", epochs.assign(label_id=epochs["stage_original"].map(label_ids))
                [["recording_id", "epoch_number", "label_id"]])

        epoch_ids = pd.read_sql("SELECT epoch_id, recording_id, epoch_number FROM epoch", conn)
        channel_ids = dict(conn.execute("SELECT name, channel_id FROM channel"))
        feats = (tidy.merge(epoch_ids, on=["recording_id", "epoch_number"], how="left", validate="many_to_one")
                 .assign(channel_id=lambda d: d["channel"].astype(str).map(channel_ids),
                         is_artifact=lambda d: d["is_artifact"].astype(int)))
        _insert(conn, "eeg_feature",
                feats[["epoch_id", "channel_id"] + STORED_FEATURES + ["is_artifact", "artifact_reason"]])
    conn.close()
    logger.info("OLTP creada: %s", path.relative_to(config.PROJECT_ROOT).as_posix())
    return path


def table_counts(path: Path = config.OLTP_DB_PATH) -> pd.Series:
    """Cantidad de filas por tabla."""
    with connect(path) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY rowid")]
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    conn.close()
    return pd.Series(counts, name="rows")
