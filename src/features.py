"""Extracción de features por epoch y canal.

Entrada: array (n_epochs, n_canales, n_muestras) en µV, ya filtrado 0.5-30 Hz.
Salida: DataFrame tidy, una fila por (epoch, canal).

Definiciones (x = señal del epoch, N muestras, fs = 100 Hz):
  - PSD: Welch, ventana Hann de 4 s (400 muestras), 50 % de solapamiento,
    detrend constante, densidad espectral en µV²/Hz. Resolución: 0.25 Hz.
  - Potencia absoluta de banda [f1, f2) = Σ PSD(f)·Δf, con f1 <= f < f2  [µV²].
    Se usa la regla del rectángulo con intervalos semiabiertos para que las
    bandas sean disjuntas, su suma sea exactamente la potencia total y la
    potencia total sea consistente con la varianza (Parseval). Nota: Simpson
    (p. ej. YASA) pondera menos el extremo de 0.5 Hz, donde la PSD es máxima,
    y da delta absoluta ~10 % menor; los valores absolutos dependen entonces de
    la convención y no deben compararse entre métodos distintos.
  - Potencia total = potencia en [0.5, 30) Hz  [µV²].
  - Potencia relativa = potencia absoluta / potencia total  [adimensional].
  - Spectral Entropy: entropía de Shannon de la PSD de Welch normalizada
    como distribución de probabilidad, dividida por log2(n_frecuencias)
    -> rango [0, 1] (antropy.spectral_entropy, normalize=True).
    Mide cuán "repartida" está la potencia entre frecuencias.
  - Sample Entropy: -ln(A/B), con B = pares de subsecuencias de largo m que
    coinciden con tolerancia r y A = las que siguen coincidiendo con m+1
    (antropy.sample_entropy, m = 2, r = 0.2·SD, distancia de Chebyshev).
    Mide la (im)predecibilidad temporal de la señal.
  - RMS = sqrt(mean(x²))  [µV].
  - Peak-to-peak = max(x) - min(x)  [µV].
"""

from __future__ import annotations

import logging

import antropy as ant
import numpy as np
import pandas as pd
from scipy.signal import welch

from src import config

logger = logging.getLogger(__name__)

BAND_NAMES = list(config.FREQ_BANDS)
FEATURE_COLUMNS = (
    [f"{b}_abs" for b in BAND_NAMES]
    + [f"{b}_rel" for b in BAND_NAMES]
    + ["total_power", "sample_entropy", "spectral_entropy", "rms", "peak_to_peak"]
)


def compute_psd(epochs_uv: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """PSD de Welch sobre el último eje. Devuelve (frecuencias, psd en µV²/Hz)."""
    nperseg = int(round(config.WELCH_WINDOW_S * sfreq))
    return welch(
        epochs_uv, fs=sfreq, window="hann", nperseg=nperseg,
        noverlap=int(nperseg * config.WELCH_OVERLAP), detrend="constant",
        scaling="density", axis=-1,
    )


def band_powers(freqs: np.ndarray, psd: np.ndarray) -> dict[str, np.ndarray]:
    """Potencia absoluta por banda y potencia total (µV²), integrando la PSD."""
    df = freqs[1] - freqs[0]

    def integrate(low: float, high: float) -> np.ndarray:
        mask = (freqs >= low) & (freqs < high)
        return psd[..., mask].sum(axis=-1) * df

    powers = {f"{name}_abs": integrate(lo, hi) for name, (lo, hi) in config.FREQ_BANDS.items()}
    powers["total_power"] = integrate(*config.TOTAL_POWER_RANGE)
    return powers


def _sample_entropy(x: np.ndarray) -> float:
    """Sample Entropy de un epoch; NaN si no está definida.

    Casos indefinidos (no se imputan): señal plana (r = 0.2·SD = 0, antropy
    devolvería 0 = "regularidad perfecta", lo que sería engañoso) y ausencia de
    coincidencias (antropy devuelve inf).
    """
    if not np.all(np.isfinite(x)) or np.std(x) < config.ARTIFACT_MIN_STD_UV:
        return np.nan
    value = ant.sample_entropy(x, order=config.SAMPEN_ORDER)
    return float(value) if np.isfinite(value) else np.nan


def compute_features(
    epochs_uv: np.ndarray, sfreq: float, channels: list[str], epoch_numbers: np.ndarray
) -> pd.DataFrame:
    """Calcula todas las features para un registro.

    Devuelve un DataFrame con una fila por (epoch_number, channel).
    """
    n_epochs, n_channels, _ = epochs_uv.shape
    # Las señales se guardan en float32 (memoria), pero los cálculos se hacen en
    # float64: acumular sumas espectrales en float32 introduce errores ~1e-6.
    epochs_uv = epochs_uv.astype(np.float64)
    freqs, psd = compute_psd(epochs_uv, sfreq)
    powers = band_powers(freqs, psd)

    with np.errstate(divide="ignore", invalid="ignore"):
        # total_power = 0 (señal plana) -> potencia relativa indefinida (NaN), no 0.
        relative = {f"{b}_rel": powers[f"{b}_abs"] / powers["total_power"] for b in BAND_NAMES}
        spectral_entropy = ant.spectral_entropy(
            epochs_uv, sfreq, method="welch", nperseg=int(round(config.WELCH_WINDOW_S * sfreq)),
            normalize=True, axis=-1,
        )
    # Señal plana: la distribución espectral no está definida (antropy devolvería 0).
    flat = epochs_uv.std(axis=-1) < config.ARTIFACT_MIN_STD_UV
    spectral_entropy = np.where(flat, np.nan, spectral_entropy)

    sample_entropy = np.array(
        [[_sample_entropy(epochs_uv[e, c]) for c in range(n_channels)] for e in range(n_epochs)]
    )

    features = {
        **powers,
        **relative,
        "sample_entropy": sample_entropy,
        "spectral_entropy": spectral_entropy,
        "rms": np.sqrt(np.mean(epochs_uv ** 2, axis=-1)),
        "peak_to_peak": np.ptp(epochs_uv, axis=-1),
    }
    table = pd.DataFrame({
        "epoch_number": np.repeat(epoch_numbers, n_channels),
        "channel": np.tile(channels, n_epochs),
        # Matrices (n_epochs, n_channels) aplanadas en el mismo orden (epoch, canal).
        **{name: np.asarray(values, dtype=np.float64).reshape(-1) for name, values in features.items()},
    })
    return table[["epoch_number", "channel"] + FEATURE_COLUMNS]
