"""Etapa EXTRACT: descarga programática de Sleep-EDF Expanded desde PhysioNet.

Flujo:
    PhysioNet (SHA256SUMS.txt) -> catálogo de registros -> selección
    -> descarga PSG + Hypnogram -> validación (tamaño + SHA256) -> data/raw/

Los archivos ya presentes y válidos no se vuelven a descargar.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# SC4ssNEx-PSG.edf  /  SC4ssNxx-Hypnogram.edf  (ss = sujeto, N = noche)
_SC_FILE_PATTERN = re.compile(
    r"^SC4(?P<subject>\d{2})(?P<night>\d)(?P<suffix>[A-Z0-9]{2})-(?P<kind>PSG|Hypnogram)\.edf$"
)


# ---------------------------------------------------------------------------
# Utilidades de red y archivos
# ---------------------------------------------------------------------------
def _download_to_path(relpath: str, dest: Path) -> str:
    """Descarga `relpath` del dataset a `dest` de forma atómica (archivo .part + rename).

    Prueba cada mirror de `config.PHYSIONET_MIRROR_URLS` con reintentos y
    devuelve la URL efectivamente utilizada.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for base_url in config.PHYSIONET_MIRROR_URLS:
        url = base_url + relpath
        for attempt in range(1, config.DOWNLOAD_RETRIES + 1):
            try:
                with urllib.request.urlopen(url, timeout=config.DOWNLOAD_TIMEOUT_S) as resp, open(tmp, "wb") as fh:
                    while chunk := resp.read(1 << 20):
                        fh.write(chunk)
                tmp.replace(dest)
                return url
            except OSError as exc:
                logger.warning("Intento %d/%d falló para %s: %s", attempt, config.DOWNLOAD_RETRIES, url, exc)
                tmp.unlink(missing_ok=True)
                time.sleep(2 * attempt)
    raise RuntimeError(f"No se pudo descargar {relpath} desde ningún mirror.")


def sha256_of(path: Path) -> str:
    """Calcula el hash SHA256 de un archivo leyendo por bloques."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _is_valid_file(path: Path, expected_sha256: str, min_size: int) -> bool:
    return path.exists() and path.stat().st_size >= min_size and sha256_of(path) == expected_sha256


# ---------------------------------------------------------------------------
# Catálogo de registros
# ---------------------------------------------------------------------------
def fetch_checksums(force: bool = False) -> pd.DataFrame:
    """Descarga (o reutiliza) SHA256SUMS.txt y lo devuelve como DataFrame [sha256, relpath]."""
    path = config.RAW_DIR / config.CHECKSUMS_FILENAME
    if force or not path.exists():
        _download_to_path(config.CHECKSUMS_FILENAME, path)
    rows = [line.split(maxsplit=1) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows, columns=["sha256", "relpath"])


def build_catalog(checksums: pd.DataFrame) -> pd.DataFrame:
    """Construye el catálogo de registros SC emparejando PSG e hipnograma por sujeto y noche.

    El sufijo del hipnograma (p. ej. 'EC', 'EH') difiere del del PSG ('E0'),
    por eso el emparejamiento se hace por (subject, night) y no por nombre.
    """
    rows = []
    prefix = config.PHYSIONET_SUBSET + "/"
    for sha, relpath in checksums.itertuples(index=False):
        if not relpath.startswith(prefix):
            continue
        match = _SC_FILE_PATTERN.match(relpath[len(prefix):])
        if match:
            rows.append({
                "subject": int(match["subject"]),
                "night": int(match["night"]),
                "kind": match["kind"],
                "relpath": relpath,
                "sha256": sha,
            })
    files = pd.DataFrame(rows)

    psg = files[files.kind == "PSG"].drop(columns="kind")
    hyp = files[files.kind == "Hypnogram"].drop(columns="kind")
    if psg.duplicated(["subject", "night"]).any() or hyp.duplicated(["subject", "night"]).any():
        raise ValueError("El catálogo contiene más de un PSG o hipnograma por (sujeto, noche).")

    catalog = psg.merge(hyp, on=["subject", "night"], how="outer", suffixes=("_psg", "_hyp"), indicator=True)
    unpaired = catalog[catalog["_merge"] != "both"]
    if not unpaired.empty:
        logger.warning("Registros sin par PSG/Hypnogram (se excluyen): %s",
                       unpaired[["subject", "night"]].to_dict("records"))
    catalog = catalog[catalog["_merge"] == "both"].drop(columns="_merge")
    catalog["recording_id"] = catalog["relpath_psg"].str.extract(r"(SC4\d{3})", expand=False)
    return catalog.sort_values(["subject", "night"]).reset_index(drop=True)


def select_records(
    catalog: pd.DataFrame,
    n_subjects: int | None = config.N_SUBJECTS,
    subject_ids: list[int] | None = config.SUBJECT_IDS,
    nights: tuple[int, ...] = config.NIGHTS,
) -> pd.DataFrame:
    """Selecciona registros por lista explícita de sujetos o por los primeros N sujetos."""
    subset = catalog[catalog["night"].isin(nights)]
    if subject_ids is not None:
        missing = set(subject_ids) - set(subset["subject"])
        if missing:
            raise ValueError(f"Sujetos no disponibles en {config.PHYSIONET_SUBSET}: {sorted(missing)}")
        chosen = list(subject_ids)
    else:
        chosen = sorted(subset["subject"].unique())[:n_subjects]
    return subset[subset["subject"].isin(chosen)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------
def _ensure_file(relpath: str, sha256: str, min_size: int) -> tuple[Path, str, str | None]:
    """Garantiza que el archivo exista y sea válido. Devuelve (ruta, estado, url usada)."""
    dest = config.RAW_DIR / relpath
    if _is_valid_file(dest, sha256, min_size):
        return dest, "cached", None
    if dest.exists():
        logger.warning("Archivo inválido (tamaño o hash), se vuelve a descargar: %s", dest.name)
    url = _download_to_path(relpath, dest)
    if not _is_valid_file(dest, sha256, min_size):
        raise RuntimeError(
            f"{dest.name}: la descarga no pasó la validación (tamaño >= {min_size} B y SHA256 oficial)."
        )
    return dest, "downloaded", url


def download_records(records: pd.DataFrame) -> pd.DataFrame:
    """Descarga PSG + hipnograma de cada registro y escribe/actualiza el manifest."""
    entries = []
    for rec in records.itertuples(index=False):
        for kind, relpath, sha, min_size in [
            ("PSG", rec.relpath_psg, rec.sha256_psg, config.MIN_PSG_SIZE_BYTES),
            ("Hypnogram", rec.relpath_hyp, rec.sha256_hyp, config.MIN_HYPNOGRAM_SIZE_BYTES),
        ]:
            path, status, url = _ensure_file(relpath, sha, min_size)
            # Solo se informa lo que implicó trabajo; lo que ya estaba en caché va a DEBUG.
            (logger.debug if status == "cached" else logger.info)("%-9s %-24s %s", kind, path.name, status)
            entries.append({
                "recording_id": rec.recording_id,
                "subject": rec.subject,
                "night": rec.night,
                "kind": kind,
                "file": path.relative_to(config.PROJECT_ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha,
                "status": status,
                "source_url": url,
                "checked_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            })
    manifest = pd.DataFrame(entries)
    _update_manifest(manifest)
    status = manifest["status"].value_counts()
    logger.info("Archivos: %d (%d en caché, %d descargados)", len(manifest),
                status.get("cached", 0), status.get("downloaded", 0))
    return manifest


def _update_manifest(new: pd.DataFrame) -> None:
    """Agrega las entradas nuevas al manifest histórico (una fila por archivo, la más reciente).

    Si un archivo se reutiliza ('cached'), se conserva la URL de la descarga original.
    """
    path = config.DOWNLOAD_MANIFEST_PATH
    if path.exists():
        old = pd.read_csv(path)
        previous_url = new["file"].map(old.set_index("file")["source_url"])
        new = new.assign(source_url=new["source_url"].where(new["source_url"].notna(), previous_url))
        new = pd.concat([old, new]).drop_duplicates("file", keep="last")
    new.sort_values("file").to_csv(path, index=False)


def download_subjects_metadata() -> Path:
    """Descarga SC-subjects.xls (edad y sexo de cada sujeto) si no existe."""
    dest = config.RAW_DIR / config.SUBJECTS_METADATA_FILENAME
    if not dest.exists():
        _download_to_path(config.SUBJECTS_METADATA_FILENAME, dest)
    return dest


def run_extract(
    n_subjects: int | None = config.N_SUBJECTS,
    subject_ids: list[int] | None = config.SUBJECT_IDS,
    nights: tuple[int, ...] = config.NIGHTS,
) -> pd.DataFrame:
    """Punto de entrada de la etapa Extract. Devuelve el manifest de esta ejecución."""
    catalog = build_catalog(fetch_checksums())
    records = select_records(catalog, n_subjects, subject_ids, nights)
    logger.info("Registros seleccionados: %d (%d sujetos)", len(records), records["subject"].nunique())
    download_subjects_metadata()
    return download_records(records)
