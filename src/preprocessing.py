"""Lectura e inspección de registros EDF (PSG) y sus hipnogramas.

Nada se asume: canales, frecuencias de muestreo, unidades, duración y
anotaciones se leen de cada archivo y se validan contra `config`.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# Tamaños de campo del header por señal en EDF (especificación EDF, Kemp et al. 1992).
_EDF_SIGNAL_FIELDS = [
    ("label", 16), ("transducer", 80), ("physical_dim", 8),
    ("physical_min", 8), ("physical_max", 8), ("digital_min", 8), ("digital_max", 8),
    ("prefilter", 80), ("n_samples_per_record", 8), ("reserved", 32),
]


class RecordingValidationError(ValueError):
    """Un registro no cumple lo que el pipeline necesita."""


def read_edf_header(path: Path) -> tuple[dict, pd.DataFrame]:
    """Lee el header EDF sin cargar la señal.

    Devuelve (header general, tabla con una fila por señal). Se usa para
    verificar unidades y frecuencias *originales* de cada canal, que MNE
    homogeneiza al cargar (sube todos los canales a la frecuencia máxima).
    """
    with open(path, "rb") as fh:
        head = fh.read(256).decode("latin-1")
        n_signals = int(head[252:256])
        block = fh.read(n_signals * 256).decode("latin-1")

    start = datetime.strptime(head[168:184], "%d.%m.%y%H.%M.%S")
    general = {
        "start_datetime": start,
        "n_records": int(head[236:244]),
        "record_duration_s": float(head[244:252]),
        "n_signals": n_signals,
    }

    columns, offset = {}, 0
    for name, width in _EDF_SIGNAL_FIELDS:
        columns[name] = [block[offset + i * width: offset + (i + 1) * width].strip() for i in range(n_signals)]
        offset += width * n_signals
    signals = pd.DataFrame(columns).drop(columns="reserved")
    signals["n_samples_per_record"] = signals["n_samples_per_record"].astype(int)
    signals["sfreq_hz"] = signals["n_samples_per_record"] / general["record_duration_s"]
    return general, signals


def read_hypnogram(path: Path) -> mne.Annotations:
    """Lee las anotaciones de etapas de sueño de un archivo *-Hypnogram.edf."""
    return mne.read_annotations(path)


def inspect_recording(psg_path: Path, hyp_path: Path) -> dict:
    """Resume las propiedades de un registro PSG + hipnograma (sin cargar la señal)."""
    psg_header, signals = read_edf_header(psg_path)
    hyp_header, _ = read_edf_header(hyp_path)
    annotations = read_hypnogram(hyp_path)

    psg_duration = psg_header["n_records"] * psg_header["record_duration_s"]
    hyp_end = float((annotations.onset + annotations.duration).max())
    eeg = signals[signals["label"].isin(config.EEG_CHANNELS)].set_index("label")

    return {
        "psg_file": psg_path.name,
        "hypnogram_file": hyp_path.name,
        "start_datetime": psg_header["start_datetime"],
        "hypnogram_start_datetime": hyp_header["start_datetime"],
        "psg_duration_s": psg_duration,
        "hypnogram_end_s": hyp_end,
        "channels": signals["label"].tolist(),
        "eeg_sfreq_hz": eeg["sfreq_hz"].to_dict(),
        "eeg_units": eeg["physical_dim"].to_dict(),
        "eeg_prefilter": eeg["prefilter"].to_dict(),
        "annotation_counts": pd.Series(annotations.description).value_counts().to_dict(),
        "annotation_durations_multiple_of_epoch": bool(
            np.allclose(annotations.duration % config.EPOCH_DURATION_S, 0)
        ),
    }


def validate_recording(info: dict) -> list[str]:
    """Valida un resumen de `inspect_recording`.

    Lanza RecordingValidationError ante problemas que impiden procesar el
    registro y devuelve una lista de advertencias para situaciones esperables
    que se tratan más adelante (p. ej. hipnograma más largo que la señal).
    """
    name = info["psg_file"]
    errors, warnings = [], []

    missing = set(config.EEG_CHANNELS) - set(info["channels"])
    if missing:
        errors.append(f"faltan canales EEG {sorted(missing)} (disponibles: {info['channels']})")
    for ch, sfreq in info["eeg_sfreq_hz"].items():
        if not np.isclose(sfreq, config.EXPECTED_SFREQ_HZ):
            errors.append(f"{ch}: fs = {sfreq} Hz, se esperaba {config.EXPECTED_SFREQ_HZ} Hz")
    for ch, unit in info["eeg_units"].items():
        if unit not in ("uV", "µV"):
            errors.append(f"{ch}: unidad '{unit}', se esperaba µV")
    if info["start_datetime"] != info["hypnogram_start_datetime"]:
        errors.append(
            f"PSG e hipnograma no comparten inicio ({info['start_datetime']} vs "
            f"{info['hypnogram_start_datetime']}): los onsets no serían comparables"
        )
    if not info["annotation_durations_multiple_of_epoch"]:
        errors.append(f"hay anotaciones cuya duración no es múltiplo de {config.EPOCH_DURATION_S} s")
    unknown = set(info["annotation_counts"]) - set(config.STAGE_MAPPING)
    if unknown:
        errors.append(f"descripciones de hipnograma no contempladas en STAGE_MAPPING: {sorted(unknown)}")

    if errors:
        raise RecordingValidationError(f"{name}: " + "; ".join(errors))

    if info["hypnogram_end_s"] > info["psg_duration_s"]:
        warnings.append(
            f"{name}: el hipnograma termina en {info['hypnogram_end_s']:.0f} s y la señal en "
            f"{info['psg_duration_s']:.0f} s; los epochs sin señal se descartan al segmentar"
        )
    return warnings


def load_eeg(psg_path: Path, hyp_path: Path) -> mne.io.Raw:
    """Carga solo los canales EEG (en memoria) y les asigna el hipnograma como anotaciones.

    Cargar solo EEG evita que MNE suba a 100 Hz los canales de 1 Hz
    (EMG, respiración, temperatura) y reduce la memoria a ~2/7.
    """
    raw = mne.io.read_raw_edf(psg_path, include=config.EEG_CHANNELS, preload=True, verbose="error")
    # Los onsets del hipnograma son relativos al inicio del registro (mismo start, validado).
    raw.set_annotations(read_hypnogram(hyp_path), emit_warning=False)
    return raw


# ---------------------------------------------------------------------------
# Epoching y etapas de sueño
# ---------------------------------------------------------------------------
def annotations_to_epochs(annotations: mne.Annotations, psg_duration_s: float) -> pd.DataFrame:
    """Divide los segmentos del hipnograma en epochs de `EPOCH_DURATION_S`.

    Cada fila es un epoch con:
      - epoch_number: índice desde el inicio del registro (trazable al EDF);
      - stage_original: descripción del hipnograma tal cual;
      - stage: etapa mapeada (W/N1/N2/N3/REM, o MOVE/UNK que NO se imputan);
      - has_signal: False si el epoch excede la duración del PSG.
    """
    epoch_len = config.EPOCH_DURATION_S
    onsets, durations = annotations.onset, annotations.duration
    gaps = ~np.isclose(onsets[1:], onsets[:-1] + durations[:-1])
    if gaps.any() or not np.isclose(onsets[0] % epoch_len, 0):
        raise RecordingValidationError(
            f"El hipnograma no es contiguo o no está alineado a {epoch_len} s "
            f"(huecos en onsets {onsets[1:][gaps].tolist()})"
        )

    n_per_segment = np.round(durations / epoch_len).astype(int)
    starts = np.concatenate([o + epoch_len * np.arange(n) for o, n in zip(onsets, n_per_segment)])
    descriptions = np.repeat(np.asarray(annotations.description, dtype=str), n_per_segment)

    epochs = pd.DataFrame({
        "epoch_number": np.round(starts / epoch_len).astype(int),
        "start_sec": starts,
        "stage_original": descriptions,
    })
    epochs["stage"] = epochs["stage_original"].map(config.STAGE_MAPPING)
    epochs["has_signal"] = epochs["start_sec"] + epoch_len <= psg_duration_s
    return epochs


def mark_analysis_window(epochs: pd.DataFrame, start_datetime: datetime,
                         padding_min: float = config.WAKE_PADDING_MIN) -> pd.DataFrame:
    """Marca la ventana de análisis del período de sueño nocturno.

    1. Se consideran solo los epochs de sueño (N1-N3, REM) cuya hora del día cae en la
       franja nocturna [NIGHT_START_HOUR, NIGHT_END_HOUR): así una siesta diurna no
       extiende la "noche" a todo el registro de ~20 h.
    2. La ventana va desde `padding_min` antes del primer epoch de sueño nocturno hasta
       `padding_min` después del último (recorte de la vigilia extensa).

    No elimina filas: agrega `in_analysis_window` para que el recorte quede documentado.
    """
    clock = pd.Timestamp(start_datetime) + pd.to_timedelta(epochs["start_sec"], unit="s")
    hour = clock.dt.hour + clock.dt.minute / 60
    nocturnal = (hour >= config.NIGHT_START_HOUR) | (hour < config.NIGHT_END_HOUR)
    sleep = epochs["stage"].isin(["N1", "N2", "N3", "REM"]) & nocturnal & epochs["has_signal"]
    if not sleep.any():
        raise RecordingValidationError("El hipnograma no contiene sueño dentro de la franja nocturna.")
    padding_s = padding_min * 60
    first_sleep = epochs.loc[sleep, "start_sec"].min()
    last_sleep = epochs.loc[sleep, "start_sec"].max()
    return epochs.assign(in_analysis_window=epochs["start_sec"].between(first_sleep - padding_s, last_sleep + padding_s))


def crop_and_filter(raw: mne.io.Raw, epochs: pd.DataFrame) -> mne.io.Raw:
    """Recorta la señal a los epochs analizables (+ margen) y aplica el pasa-banda.

    Se filtra la señal continua (no cada epoch) para evitar efectos de borde
    por epoch. Filtro FIR de fase cero (default de MNE, ventana Hamming):
      - pasa-altos 0.5 Hz: elimina deriva lenta/DC residual (el registro ya
        tiene HP 0.5 Hz de hardware; esto garantiza el mismo corte en todos);
      - pasa-bajos 30 Hz: límite superior de la banda beta analizada; atenúa
        actividad muscular y ruido de red (50 Hz, cerca de Nyquist).
    Modifica `raw` in place y lo devuelve.
    """
    usable = epochs[epochs["in_analysis_window"] & epochs["has_signal"]]
    duration = raw.n_times / raw.info["sfreq"]
    tmin = max(0.0, usable["start_sec"].min() - config.FILTER_EDGE_PAD_S)
    tmax = min(duration - 1 / raw.info["sfreq"],
               usable["start_sec"].max() + config.EPOCH_DURATION_S + config.FILTER_EDGE_PAD_S)
    raw.crop(tmin=tmin, tmax=tmax)
    raw.filter(config.L_FREQ_HZ, config.H_FREQ_HZ, picks="eeg", verbose="error")
    return raw


def extract_epoch_array(raw: mne.io.Raw, start_secs: np.ndarray) -> np.ndarray:
    """Devuelve un array (n_epochs, n_canales, n_muestras) en µV, float32.

    `start_secs` son tiempos desde el inicio del registro original; se
    convierten a índices de muestra considerando el recorte (`raw.first_time`).
    float32 alcanza para EEG (precisión ~7 dígitos) y reduce la memoria a la mitad.
    """
    sfreq = raw.info["sfreq"]
    n_samples = int(round(config.EPOCH_DURATION_S * sfreq))
    first = np.round((np.asarray(start_secs) - raw.first_time) * sfreq).astype(int)
    if first.min() < 0 or first.max() + n_samples > raw.n_times:
        raise RecordingValidationError("Algún epoch cae fuera de la señal recortada.")
    data = raw.get_data(units="uV")
    out = np.stack([data[:, s:s + n_samples] for s in first]).astype(np.float32)
    del data
    return out
