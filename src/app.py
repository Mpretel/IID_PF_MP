"""Dashboard Streamlit: dinámica espectral y complejidad del EEG por etapa de sueño.

Lee exclusivamente el Data Warehouse (data/database/sleep_dw.sqlite), nunca los EDF.

Ejecutar desde la raíz del repositorio:
    streamlit run src/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # permite `import src` con `streamlit run`

from src import config, plots, queries  # noqa: E402

st.set_page_config(page_title="EEG de sueño", layout="wide")


@st.cache_data(show_spinner="Leyendo el Data Warehouse...")
def load_data(db_mtime: float) -> pd.DataFrame:
    """Observaciones (epoch × canal) del DW. `db_mtime` invalida el caché si la base cambia."""
    return queries.load_observations()


def relative_power_table(obs: pd.DataFrame) -> pd.DataFrame:
    """Potencia relativa media por etapa y canal, con la misma forma que la consulta SQL."""
    cols = {f"{band}_rel": band for band in plots.BANDS}
    return (obs.groupby(["stage_code", "channel"], as_index=False)[list(cols)].mean()
            .rename(columns=cols))


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
if not config.DW_DB_PATH.exists():
    st.error("No se encontró `data/database/sleep_dw.sqlite`. Ejecutá primero el notebook "
             "`notebooks/TP_final.ipynb` (o el pipeline) para generar las bases.")
    st.stop()

obs = load_data(config.DW_DB_PATH.stat().st_mtime)

# ---------------------------------------------------------------------------
# Filtros (barra lateral)
# ---------------------------------------------------------------------------
st.sidebar.header("Filtros")
subjects = sorted(obs["subject_id"].unique())
chosen_subjects = st.sidebar.multiselect("Sujeto", subjects, default=subjects,
                                         format_func=lambda s: f"Sujeto {s}")
recordings = sorted(obs.loc[obs["subject_id"].isin(chosen_subjects), "recording_id"].unique())
chosen_recordings = st.sidebar.multiselect("Noche / registro", recordings, default=recordings)
channel = st.sidebar.radio("Canal", sorted(obs["channel"].unique()))
chosen_stages = st.sidebar.multiselect("Etapas de sueño", plots.STAGE_ORDER, default=plots.STAGE_ORDER)
include_artifacts = st.sidebar.checkbox("Incluir epochs marcados como artefacto", value=False)
st.sidebar.caption(f"Umbral de artefacto: pico a pico > {config.ARTIFACT_MAX_PTP_UV:.0f} µV, "
                   f"señal plana (SD < {config.ARTIFACT_MIN_STD_UV} µV) o valores no finitos.")

base = obs[obs["recording_id"].isin(chosen_recordings) & (obs["channel"] == channel)]
if not include_artifacts:
    base = base[~base["is_artifact"]]
data = base[base["stage_code"].isin(chosen_stages)]

# ---------------------------------------------------------------------------
# Encabezado e indicadores
# ---------------------------------------------------------------------------
st.title("EEG de sueño: espectro y complejidad por etapa")
st.markdown("¿Cómo cambian la **dinámica espectral** y la **complejidad** del EEG entre las etapas del sueño? "
            "Datos: Sleep-EDF Expanded (PhysioNet), epochs de 30 s.")

if data.empty:
    st.warning("No hay observaciones para la combinación de filtros elegida.")
    st.stop()

n_artifacts = obs.loc[obs["recording_id"].isin(chosen_recordings) & (obs["channel"] == channel), "is_artifact"]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Epochs analizados", f"{len(data):,}".replace(",", "."))
k2.metric("Registros", data["recording_id"].nunique())
k3.metric("Horas", f"{len(data) * config.EPOCH_DURATION_S / 3600:.1f}")
k4.metric("Epochs con artefacto", f"{100 * n_artifacts.mean():.1f} %")

# ---------------------------------------------------------------------------
# 1. Hipnograma
# ---------------------------------------------------------------------------
st.subheader("1 · Estructura de la noche")
hyp_recording = st.selectbox("Registro para el hipnograma", sorted(base["recording_id"].unique()))
st.plotly_chart(plots.hypnogram(base, hyp_recording), width="stretch")

# ---------------------------------------------------------------------------
# 2. Complejidad
# ---------------------------------------------------------------------------
st.subheader("2 · Complejidad por etapa")
c1, c2 = st.columns(2)
c1.plotly_chart(plots.metric_by_stage(data, "sample_entropy"), width="stretch")
c2.plotly_chart(plots.metric_by_stage(data, "spectral_entropy"), width="stretch")
st.caption("Sample Entropy mide la impredecibilidad temporal de la señal; Spectral Entropy, cuán repartida "
           "está la potencia entre frecuencias. Son métricas distintas: no se comparan en la misma escala.")

# ---------------------------------------------------------------------------
# 3. Espectro
# ---------------------------------------------------------------------------
st.subheader("3 · Potencia relativa por banda")
rel = relative_power_table(data)
st.plotly_chart(plots.band_power_by_stage(rel, channel), width="stretch")
st.plotly_chart(plots.stage_band_heatmap(rel, channel), width="stretch")

# ---------------------------------------------------------------------------
# 4. Relación espectro–complejidad
# ---------------------------------------------------------------------------
st.subheader("4 · Actividad delta vs complejidad")
st.plotly_chart(plots.delta_vs_entropy(data, channel), width="stretch")
st.caption("Asociación descriptiva, no causal: ambas variables dependen de la etapa de sueño. "
           "Las diferencias observadas no implican significancia estadística.")

with st.expander("Ver datos filtrados"):
    st.dataframe(data.drop(columns=["stage_order", "is_scored"]), width="stretch", hide_index=True)
