"""Configuración central del proyecto.

Todas las rutas se calculan relativamente a la raíz del repositorio, por lo que
el proyecto funciona igual en VS Code, en un clon de GitHub o en Google Colab.
Los parámetros metodológicos (bandas, umbrales, filtros) se definen acá para
que no aparezcan "números mágicos" en el resto del código.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BIDS_DIR = DATA_DIR / "bids"
PROCESSED_DIR = DATA_DIR / "processed"
DATABASE_DIR = DATA_DIR / "database"

OLTP_DB_PATH = DATABASE_DIR / "sleep_oltp.sqlite"
DW_DB_PATH = DATABASE_DIR / "sleep_dw.sqlite"

DOWNLOAD_MANIFEST_PATH = RAW_DIR / "download_manifest.csv"

# ---------------------------------------------------------------------------
# Dataset: Sleep-EDF Expanded (PhysioNet)
# ---------------------------------------------------------------------------
PHYSIONET_BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/"
# Mirror oficial de PhysioNet en AWS Open Data (mismo contenido, ~50x más rápido
# en pruebas). Se intenta primero; la integridad se verifica igual contra los
# SHA256 oficiales, de modo que la fuente no altera los datos.
PHYSIONET_MIRROR_URLS = [
    "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/",
    PHYSIONET_BASE_URL,
]
PHYSIONET_SUBSET = "sleep-cassette"           # estudio SC: sujetos sanos, sin medicación
CHECKSUMS_FILENAME = "SHA256SUMS.txt"         # lista oficial de archivos + hashes
SUBJECTS_METADATA_FILENAME = "SC-subjects.xls"

# Selección de registros. Para escalar basta con cambiar estos valores.
#   - N_SUBJECTS: se toman los primeros N sujetos disponibles (orden por ID).
#   - SUBJECT_IDS: lista explícita de IDs (tiene prioridad sobre N_SUBJECTS).
#   - NIGHTS: noches a incluir (algunos sujetos tienen solo una).
N_SUBJECTS = 2
SUBJECT_IDS: list[int] | None = None
NIGHTS: tuple[int, ...] = (1, 2)

# Validación de tamaño mínimo de descarga (bytes). Un PSG de SC pesa ~40-50 MB
# y un hipnograma unos pocos KB; valores menores indican descarga truncada.
MIN_PSG_SIZE_BYTES = 1_000_000
MIN_HYPNOGRAM_SIZE_BYTES = 1_000

DOWNLOAD_TIMEOUT_S = 120
DOWNLOAD_RETRIES = 3

# ---------------------------------------------------------------------------
# Señal EEG
# ---------------------------------------------------------------------------
EEG_CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]
EXPECTED_SFREQ_HZ = 100.0
EPOCH_DURATION_S = 30.0

# Recorte de vigilia: se conservan solo estos minutos de W antes del primer
# epoch de sueño y después del último (los registros SC duran ~20 h).
WAKE_PADDING_MIN = 30

# Filtro pasa-banda aplicado a la señal continua antes de segmentar.
L_FREQ_HZ = 0.5
H_FREQ_HZ = 30.0
# Señal extra conservada a cada lado de la ventana de análisis para que el
# transitorio del filtro FIR (~6.6 s para 0.5 Hz) no afecte epochs analizados.
FILTER_EDGE_PAD_S = 30.0

# ---------------------------------------------------------------------------
# Etapas de sueño (R&K -> AASM simplificado)
# ---------------------------------------------------------------------------
# Las descripciones exactas se verifican en cada hipnograma.
STAGE_MAPPING = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",   # R&K S3 + S4 = AASM N3
    "Sleep stage R": "REM",
    # No son etapas de sueño: se conservan pero NO se imputan.
    "Movement time": "MOVE",
    "Sleep stage ?": "UNK",
}
SCORED_STAGES = ["W", "N1", "N2", "N3", "REM"]
UNSCORED_STAGES = ["MOVE", "UNK"]

# ---------------------------------------------------------------------------
# Features espectrales
# ---------------------------------------------------------------------------
FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}
TOTAL_POWER_RANGE = (L_FREQ_HZ, H_FREQ_HZ)
WELCH_WINDOW_S = 4.0       # resolución espectral 0.25 Hz
WELCH_OVERLAP = 0.5

SAMPEN_ORDER = 2           # m; tolerancia r = 0.2 * SD (default de antropy)

# ---------------------------------------------------------------------------
# Control de calidad (flags de artefacto, NO eliminación)
# ---------------------------------------------------------------------------
ARTIFACT_MAX_PTP_UV = 500.0    # amplitud pico a pico excesiva
ARTIFACT_MIN_STD_UV = 0.5      # señal plana / electrodo desconectado

RANDOM_SEED = 42
