"""Organización de los registros según BIDS (EEG-BIDS) usando MNE-BIDS.

data/raw/  -> archivos originales de PhysioNet (no se modifican)
data/bids/ -> misma información organizada como:
    dataset_description.json, participants.tsv, README
    sub-XX/ses-N/eeg/sub-XX_ses-N_task-sleep_eeg.edf  (+ channels.tsv, events.tsv, eeg.json)

MNE-BIDS copia el EDF original sin reescribir la señal; las etapas del
hipnograma se exportan como eventos en *_events.tsv.
"""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path

import mne
import pandas as pd
from mne_bids import BIDSPath, make_dataset_description, write_raw_bids

from src import config
from src.preprocessing import read_edf_header, read_hypnogram

logger = logging.getLogger(__name__)

BIDS_TASK = "sleep"
LINE_FREQ_HZ = 50  # Sleep-EDF se registró en Países Bajos (red eléctrica de 50 Hz)

# Tipo BIDS de cada canal del PSG (los nombres se verifican en preprocessing).
CHANNEL_TYPES = {
    "EEG Fpz-Cz": "eeg",
    "EEG Pz-Oz": "eeg",
    "EOG horizontal": "eog",
    "EMG submental": "emg",
    "Resp oro-nasal": "resp",
    "Temp rectal": "temperature",
    "Event marker": "misc",
}

# Unidades del header EDF -> notación BIDS (CMIXF).
_UNIT_TO_BIDS = {"uV": "µV", "DegC": "oC", "": "n/a"}

# Códigos enteros para events.tsv (BIDS exige un valor numérico por tipo de evento).
EVENT_ID = {description: code for code, description in enumerate(config.STAGE_MAPPING, start=1)}


def load_subjects_metadata() -> pd.DataFrame:
    """Lee SC-subjects.xls y lo normaliza (una fila por sujeto y noche).

    El archivo codifica el sexo como 'sex (F=1)'; se traduce a 'F'/'M'.
    """
    meta = pd.read_excel(config.RAW_DIR / config.SUBJECTS_METADATA_FILENAME)
    meta = meta.rename(columns={"sex (F=1)": "sex_code", "LightsOff": "lights_off"})
    unexpected = set(meta["sex_code"]) - {1, 2}
    if unexpected:
        raise ValueError(f"Códigos de sexo inesperados en SC-subjects.xls: {unexpected}")
    meta["sex"] = meta["sex_code"].map({1: "F", 2: "M"})
    meta["lights_off"] = meta["lights_off"].astype(str)
    return meta[["subject", "night", "age", "sex", "lights_off"]]


def bids_subject_label(subject: int) -> str:
    """Etiqueta BIDS del sujeto: dos dígitos, como en los nombres originales (SC4ss...)."""
    return f"{subject:02d}"


def make_bids_path(subject: int, night: int) -> BIDSPath:
    return BIDSPath(
        subject=bids_subject_label(subject),
        session=str(night),
        task=BIDS_TASK,
        datatype="eeg",
        suffix="eeg",
        extension=".edf",
        root=config.BIDS_DIR,
    )


def convert_recording(psg_path: Path, hyp_path: Path, subject: int, night: int, overwrite: bool = False) -> BIDSPath:
    """Escribe un registro (PSG + hipnograma) en formato BIDS. Reutiliza si ya existe."""
    bids_path = make_bids_path(subject, night)
    if bids_path.fpath.exists() and not overwrite:
        logger.info("BIDS existente, se reutiliza: %s", bids_path.basename)
        return bids_path

    raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="error")
    with warnings.catch_warnings():
        # MNE avisa que cambia la unidad interna al reasignar tipos (esperable).
        warnings.filterwarnings("ignore", message="The unit for channel")
        raw.set_channel_types({ch: t for ch, t in CHANNEL_TYPES.items() if ch in raw.ch_names})
    raw.info["line_freq"] = LINE_FREQ_HZ
    raw.set_annotations(read_hypnogram(hyp_path), emit_warning=False)

    write_raw_bids(raw, bids_path, event_id=EVENT_ID, overwrite=True, verbose="error")
    _restore_channel_metadata(bids_path, psg_path)
    logger.info("BIDS escrito: %s", bids_path.basename)
    return bids_path


def _parse_prefilter(prefilter: str, key: str) -> float | str:
    """Extrae la frecuencia de corte 'HP:' o 'LP:' del campo prefilter del EDF."""
    match = re.search(rf"{key}:\s*([\d.]+)\s*Hz", prefilter)
    return float(match.group(1)) if match else "n/a"


def _restore_channel_metadata(bids_path: BIDSPath, psg_path: Path) -> None:
    """Corrige channels.tsv con los valores originales del header EDF.

    MNE remuestrea internamente todos los canales a la frecuencia máxima, por lo
    que MNE-BIDS reporta 100 Hz y cortes 0-50 Hz para todos. El EDF copiado
    conserva las frecuencias originales (p. ej. EMG a 1 Hz), y channels.tsv
    debe describir ese archivo, no la representación interna de MNE.
    """
    _, header = read_edf_header(psg_path)
    header = header.set_index("label")
    tsv_path = bids_path.copy().update(suffix="channels", extension=".tsv").fpath
    channels = pd.read_csv(tsv_path, sep="	", keep_default_na=False)
    names = channels["name"]
    channels["sampling_frequency"] = names.map(header["sfreq_hz"])
    channels["units"] = names.map(header["physical_dim"]).map(lambda u: _UNIT_TO_BIDS.get(u, u))
    channels["low_cutoff"] = names.map(header["prefilter"]).map(lambda f: _parse_prefilter(f, "HP"))
    channels["high_cutoff"] = names.map(header["prefilter"]).map(lambda f: _parse_prefilter(f, "LP"))
    channels.to_csv(tsv_path, sep="	", index=False)


def write_participants(subjects: list[int]) -> pd.DataFrame:
    """Completa participants.tsv con edad y sexo desde SC-subjects.xls.

    MNE-BIDS crea el archivo con columnas vacías ('n/a') porque el EDF de
    Sleep-EDF no trae datos demográficos; acá se agregan desde la fuente oficial.
    """
    path = config.BIDS_DIR / "participants.tsv"
    participants = pd.read_csv(path, sep="\t")
    demo = (load_subjects_metadata()
            .drop_duplicates("subject")
            .assign(participant_id=lambda d: "sub-" + d["subject"].map(bids_subject_label)))
    participants = (participants[["participant_id"]]
                    .merge(demo[["participant_id", "age", "sex"]], on="participant_id", how="left"))
    missing = participants[participants["age"].isna()]
    if not missing.empty:
        raise ValueError(f"Sin datos demográficos para: {missing['participant_id'].tolist()}")
    participants.to_csv(path, sep="\t", index=False, na_rep="n/a")
    return participants


def write_dataset_description() -> None:
    make_dataset_description(
        path=config.BIDS_DIR,
        name="Sleep-EDF Expanded (subset, sleep-cassette) - TP Introducción a la Ingeniería de Datos",
        dataset_type="raw",
        authors=["Kemp B", "Zwinderman AH", "Tuk B", "Kamphuisen HAC", "Oberyé JJL"],
        how_to_acknowledge="Citar Kemp et al. (2000) IEEE-BME 47(9):1185-1194 y Goldberger et al. (2000) PhysioNet.",
        references_and_links=["https://physionet.org/content/sleep-edfx/1.0.0/"],
        doi="doi:10.13026/C2X676",
        overwrite=True,
        verbose="error",
    )


def run_bids(manifest: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    """Convierte a BIDS todos los registros del manifest de Extract.

    Devuelve una tabla registro -> archivo BIDS.
    """
    files = manifest.pivot(index=["recording_id", "subject", "night"], columns="kind", values="file").reset_index()
    rows = []
    for rec in files.itertuples(index=False):
        bids_path = convert_recording(
            config.PROJECT_ROOT / rec.PSG, config.PROJECT_ROOT / rec.Hypnogram, rec.subject, rec.night, overwrite
        )
        rows.append({
            "recording_id": rec.recording_id,
            "subject": rec.subject,
            "night": rec.night,
            "bids_file": bids_path.fpath.relative_to(config.PROJECT_ROOT).as_posix(),
        })
    write_dataset_description()
    write_participants(sorted(files["subject"].unique()))
    return pd.DataFrame(rows)
