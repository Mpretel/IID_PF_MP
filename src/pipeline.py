"""Orquestación del Transform: de los EDF descargados a las tablas procesadas.

Por cada registro (secuencialmente, para no mantener varias señales en RAM):
    inspección + validación -> epochs + etapas -> recorte + filtro
    -> array de epochs -> features -> flags de artefacto -> caché Parquet

Luego se combinan los registros, se calculan outliers estadísticos y se
escriben las tablas finales en data/processed/.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, features, preprocessing, quality
from src.bids import load_subjects_metadata

logger = logging.getLogger(__name__)

CACHE_DIR = config.PROCESSED_DIR / "by_recording"
TIDY_PATH = config.PROCESSED_DIR / "features_tidy.parquet"
EPOCHS_PATH = config.PROCESSED_DIR / "epochs.parquet"
RECORDINGS_PATH = config.PROCESSED_DIR / "recordings.parquet"
MISSING_REPORT_PATH = config.PROCESSED_DIR / "missing_report.csv"
ENVIRONMENT_PATH = config.PROCESSED_DIR / "environment.json"

# Incrementar cuando cambie el código de cálculo (no solo parámetros) para invalidar el caché.
TRANSFORM_CODE_VERSION = 4

TRACKED_PACKAGES = ["numpy", "pandas", "scipy", "mne", "mne-bids", "antropy", "pyarrow", "plotly", "streamlit",
                    "scikit-learn"]


def processing_params() -> dict:
    """Parámetros que determinan el resultado del Transform (para invalidar el caché)."""
    return {
        "code_version": TRANSFORM_CODE_VERSION,
        "eeg_channels": config.EEG_CHANNELS,
        "epoch_s": config.EPOCH_DURATION_S,
        "wake_padding_min": config.WAKE_PADDING_MIN,
        "night_hours": [config.NIGHT_START_HOUR, config.NIGHT_END_HOUR],
        "filter": [config.L_FREQ_HZ, config.H_FREQ_HZ, config.FILTER_EDGE_PAD_S],
        "stage_mapping": config.STAGE_MAPPING,
        "bands": config.FREQ_BANDS,
        "welch": [config.WELCH_WINDOW_S, config.WELCH_OVERLAP],
        "sampen_order": config.SAMPEN_ORDER,
        "artifact": [config.ARTIFACT_MIN_CLIPPED_SAMPLES, config.ARTIFACT_MIN_STD_UV],
    }


def params_hash() -> str:
    payload = json.dumps(processing_params(), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def _cache_paths(recording_id: str) -> dict[str, Path]:
    return {
        "features": CACHE_DIR / f"{recording_id}_features.parquet",
        "epochs": CACHE_DIR / f"{recording_id}_epochs.parquet",
        "meta": CACHE_DIR / f"{recording_id}_meta.json",
    }


def _load_cache(recording_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    paths = _cache_paths(recording_id)
    if not all(p.exists() for p in paths.values()):
        return None
    meta = json.loads(paths["meta"].read_text())
    if meta.get("params_hash") != params_hash():
        logger.info("%s: parámetros cambiaron, se recalcula", recording_id)
        return None
    return pd.read_parquet(paths["epochs"]), pd.read_parquet(paths["features"]), meta


def _clipped_counts(raw, psg_path: Path, start_secs: np.ndarray) -> np.ndarray:
    """Muestras en el límite físico del EDF por (epoch, canal), sobre la señal sin filtrar."""
    limits = preprocessing.read_edf_header(psg_path)[1].set_index("label")
    data = preprocessing.extract_epoch_array(raw, start_secs)
    counts = []
    for index, channel in enumerate(raw.ch_names):
        low, high = float(limits.loc[channel, "physical_min"]), float(limits.loc[channel, "physical_max"])
        step = (high - low) / (float(limits.loc[channel, "digital_max"]) - float(limits.loc[channel, "digital_min"]))
        counts.append(quality.count_clipped_samples(data[:, index], low, high, step))
    return np.stack(counts, axis=1)


def process_recording(recording_id: str, psg_path: Path, hyp_path: Path, force: bool = False
                      ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Procesa un registro completo. Devuelve (epochs, features, metadata del registro)."""
    if not force and (cached := _load_cache(recording_id)) is not None:
        logger.debug("%s: caché reutilizado", recording_id)
        return cached

    info = preprocessing.inspect_recording(psg_path, hyp_path)
    for warning in preprocessing.validate_recording(info):
        logger.warning(warning)

    epochs = preprocessing.annotations_to_epochs(preprocessing.read_hypnogram(hyp_path), info["psg_duration_s"])
    epochs = preprocessing.mark_analysis_window(epochs, info["start_datetime"])
    usable = epochs[epochs["in_analysis_window"] & epochs["has_signal"]]

    raw = preprocessing.load_eeg(psg_path, hyp_path)
    n_clipped = _clipped_counts(raw, psg_path, usable["start_sec"].to_numpy())
    raw = preprocessing.crop_and_filter(raw, epochs)
    sfreq, channels = raw.info["sfreq"], list(raw.ch_names)
    data = preprocessing.extract_epoch_array(raw, usable["start_sec"].to_numpy())
    del raw

    table = features.compute_features(data, sfreq, channels, usable["epoch_number"].to_numpy())
    is_artifact, reason = quality.flag_artifacts(data, n_clipped)
    table["is_artifact"] = is_artifact.reshape(-1)
    table["artifact_reason"] = reason.reshape(-1)
    del data
    gc.collect()

    meta = {
        "recording_id": recording_id,
        "psg_file": info["psg_file"],
        "hypnogram_file": info["hypnogram_file"],
        "start_datetime": info["start_datetime"].isoformat(),
        "psg_duration_s": info["psg_duration_s"],
        "sampling_frequency_hz": sfreq,
        "n_epochs_hypnogram": len(epochs),
        "n_epochs_analyzed": len(usable),
        "analysis_start_sec": float(usable["start_sec"].min()),
        "analysis_end_sec": float(usable["start_sec"].max() + config.EPOCH_DURATION_S),
        "params_hash": params_hash(),
    }
    paths = _cache_paths(recording_id)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    epochs.to_parquet(paths["epochs"], index=False)
    table.to_parquet(paths["features"], index=False)
    paths["meta"].write_text(json.dumps(meta, indent=2))
    logger.info("%s: %d épocas analizadas, %d filas de features", recording_id, len(usable), len(table))
    return epochs, table, meta


def build_tidy_table(epochs: pd.DataFrame, feats: pd.DataFrame, recordings: pd.DataFrame) -> pd.DataFrame:
    """Une features con contexto: una fila = un epoch de 30 s de un canal EEG."""
    context = epochs[["recording_id", "epoch_number", "start_sec", "stage", "stage_original"]]
    tidy = (feats
            .merge(context, on=["recording_id", "epoch_number"], how="left", validate="many_to_one")
            .merge(recordings[["recording_id", "subject", "night"]], on="recording_id", how="left",
                   validate="many_to_one")
            .rename(columns={"subject": "subject_id", "start_sec": "start_time_sec", "stage": "sleep_stage"}))
    tidy = quality.flag_statistical_outliers(tidy, group_cols=["recording_id", "channel", "sleep_stage"])
    key = ["subject_id", "recording_id", "night", "epoch_number", "start_time_sec",
           "sleep_stage", "stage_original", "channel"]
    flags = ["is_artifact", "artifact_reason", "is_stat_outlier", "outlier_features"]
    tidy = tidy[key + features.FEATURE_COLUMNS + flags]
    tidy["sleep_stage"] = pd.Categorical(tidy["sleep_stage"],
                                         categories=config.SCORED_STAGES + config.UNSCORED_STAGES)
    tidy["channel"] = tidy["channel"].astype("category")
    return tidy.sort_values(["recording_id", "epoch_number", "channel"]).reset_index(drop=True)


def library_versions() -> dict:
    return {"python": platform.python_version(), **{pkg: version(pkg) for pkg in TRACKED_PACKAGES}}


def run_pipeline(manifest: pd.DataFrame, force: bool = False) -> dict[str, pd.DataFrame]:
    """Procesa todos los registros del manifest de Extract y escribe las tablas en data/processed/."""
    files = manifest.pivot(index=["recording_id", "subject", "night"], columns="kind", values="file").reset_index()
    all_epochs, all_features, all_meta = [], [], []
    for rec in files.itertuples(index=False):
        epochs, feats, meta = process_recording(
            rec.recording_id, config.PROJECT_ROOT / rec.PSG, config.PROJECT_ROOT / rec.Hypnogram, force
        )
        all_epochs.append(epochs.assign(recording_id=rec.recording_id))
        all_features.append(feats.assign(recording_id=rec.recording_id))
        all_meta.append({**meta, "subject": rec.subject, "night": rec.night})

    epochs = pd.concat(all_epochs, ignore_index=True)
    recordings = (pd.DataFrame(all_meta)
                  .merge(load_subjects_metadata(), on=["subject", "night"], how="left", validate="one_to_one"))
    tidy = build_tidy_table(epochs, pd.concat(all_features, ignore_index=True), recordings)
    missing = quality.missing_data_report(epochs, tidy)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tidy.to_parquet(TIDY_PATH, index=False)
    epochs.to_parquet(EPOCHS_PATH, index=False)
    recordings.to_parquet(RECORDINGS_PATH, index=False)
    missing.to_csv(MISSING_REPORT_PATH, index=False)
    ENVIRONMENT_PATH.write_text(json.dumps(
        {"library_versions": library_versions(), "params_hash": params_hash(),
         "processing_params": processing_params()}, indent=2, ensure_ascii=False))
    logger.info("Tidy: %d filas x %d columnas (%.1f MB en memoria)",
                *tidy.shape, tidy.memory_usage(deep=True).sum() / 1e6)
    return {"tidy": tidy, "epochs": epochs, "recordings": recordings, "missing": missing}
