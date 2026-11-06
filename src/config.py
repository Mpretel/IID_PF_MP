"""Configuración del proyecto.
En este archivo se definen todos los parámetros metodológicos como bandas de 
frecuencias, umbrales y filtros.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directorios de datos y bases de datos. Se crean automáticamente si no existen.
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
# Mirror oficial de PhysioNet en AWS Open Data.
# La integridad se verifica igual contra los SHA256.
PHYSIONET_MIRROR_URLS = [
    "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/",
    PHYSIONET_BASE_URL,
]
PHYSIONET_SUBSET = "sleep-cassette"           # estudio SC: sujetos sanos, sin medicación
CHECKSUMS_FILENAME = "SHA256SUMS.txt"         # lista oficial de archivos + hashes
SUBJECTS_METADATA_FILENAME = "SC-subjects.xls"

# Selección de registros
#   - N_SUBJECTS: se toman los primeros N sujetos disponibles (orden por ID).
#   - SUBJECT_IDS: lista explícita de IDs (tiene prioridad sobre N_SUBJECTS).
#   - NIGHTS: noches a incluir (algunos sujetos tienen solo una).
N_SUBJECTS = 18
# Muestra estratificada de 18 sujetos: 3 mujeres y 3 hombres en cada grupo etario.
# Regla: para cada sexo, los sujetos más cercanos a los cuantiles 10/30/50/70/90 % de edad; luego cada grupo y
# sexo se completa hasta 3 con la edad más alejada de las ya elegidas (en empate, el ID más bajo).
# Siempre sujetos con 2 noches, y sin mirar sus señales.
#   <40:   F 5 (28 años), 6 (31), 4 (34)      M 10 (26), 14 (27), 16 (32)
#   40-64: F 20 (51), 24 (54), 22 (56)        M 38 (51), 31 (54), 33 (60)
#   65+:   F 40 (67), 65 (88), 61 (101)       M 50 (71), 76 (90), 73 (97)
# Con SUBJECT_IDS = None se usan los primeros N_SUBJECTS sujetos por ID.
SUBJECT_IDS: list[int] | None = [4, 5, 6, 10, 14, 16, 20, 22, 24, 31, 33, 38, 40, 50, 61, 65, 73, 76]
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

# Recorte de vigilia (W): se conservan solo estos minutos de W antes de la primera
# época de sueño y después del último (los registros SC duran ~20 h).
WAKE_PADDING_MIN = 30
# Se utiliza la siguiente franja horria de sueño (hora local del registro):
NIGHT_START_HOUR = 21
NIGHT_END_HOUR = 12

# Filtro pasa-banda aplicado a la señal continua antes de segmentar.
L_FREQ_HZ = 0.5
H_FREQ_HZ = 30.0
# Señal extra conservada a cada lado de la ventana de análisis para que el
# transitorio del filtro FIR (~6.6 s para 0.5 Hz) no afecte épocas analizadas.
FILTER_EDGE_PAD_S = 30.0

# ---------------------------------------------------------------------------
# Etapas de sueño (R&K -> AASM simplificado)
# ---------------------------------------------------------------------------
STAGE_MAPPING = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",   # R&K S3 + S4 = AASM N3
    "Sleep stage R": "REM",
    # No son etapas de sueño: se conservan pero NO se imputan.
    "Movement time": "MOVE",
    "Sleep stage ?": "UNK", # Sin marcar por parte del experto (no confundir con artefacto)
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
ARTIFACT_MIN_CLIPPED_SAMPLES = 10    # muestras en el límite físico del EDF por epoch (0.1 s): saturación
ARTIFACT_MIN_STD_UV = 0.5            # señal plana / electrodo desconectado

# Diagnóstico de outliers
OUTLIER_TOP_FRACTION = 0.01    # cada detector multivariado marca la misma proporción de filas
OUTLIER_MIN_VOTES = 4          # detectores (de 5) que deben coincidir para el consenso

RANDOM_SEED = 42
