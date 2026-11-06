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
    """Observaciones (época × canal) del DW. `db_mtime` invalida el caché si la base cambia."""
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
age_groups = [g for g in ["<40", "40-64", "65+"] if g in set(obs["age_group"])]
chosen_ages = st.sidebar.multiselect("Grupo etario", age_groups, default=age_groups)
subjects = sorted(obs.loc[obs["age_group"].isin(chosen_ages), "subject_id"].unique())
chosen_subjects = st.sidebar.multiselect("Sujeto", subjects, default=subjects,
                                         format_func=lambda s: f"Sujeto {s}")
recordings = sorted(obs.loc[obs["subject_id"].isin(chosen_subjects), "recording_id"].unique())
chosen_recordings = st.sidebar.multiselect("Noche / registro", recordings, default=recordings)
channel = st.sidebar.radio("Canal", sorted(obs["channel"].unique()))
chosen_stages = st.sidebar.multiselect("Etapas de sueño", plots.STAGE_ORDER, default=plots.STAGE_ORDER)
include_artifacts = st.sidebar.checkbox("Incluir épocas marcadas como artefacto", value=False)
st.sidebar.caption(f"Artefactos: ≥ {config.ARTIFACT_MIN_CLIPPED_SAMPLES} muestras saturadas, "
                   f"señal plana (SD < {config.ARTIFACT_MIN_STD_UV} µV) o valores no finitos.")

base = obs[obs["recording_id"].isin(chosen_recordings) & (obs["channel"] == channel)]
if not include_artifacts:
    base = base[~base["is_artifact"]]
data = base[base["stage_code"].isin(chosen_stages)]

# ---------------------------------------------------------------------------
# Encabezado e indicadores
# ---------------------------------------------------------------------------
st.title("EEG de sueño: qué cambia entre etapas y con la edad")
st.markdown("En toda la muestra, **el EEG se vuelve más regular a medida que el sueño NREM se profundiza**, y "
            "**la potencia sigma de N2 es menor a mayor edad**. Datos: Sleep-EDF Expanded (PhysioNet), épocas de 30 s, "
            "18 sujetos de 26 a 101 años. Los resultados son descriptivos.")

if data.empty:
    st.warning("No hay observaciones para la combinación de filtros elegida.")
    st.stop()

n_artifacts = obs.loc[obs["recording_id"].isin(chosen_recordings) & (obs["channel"] == channel), "is_artifact"]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Épocas analizadas", f"{len(data):,}".replace(",", "."))
k2.metric("Registros", data["recording_id"].nunique())
k3.metric("Horas", f"{len(data) * config.EPOCH_DURATION_S / 3600:.1f}")
k4.metric("Épocas con artefacto", f"{100 * n_artifacts.mean():.1f} %")

# ---------------------------------------------------------------------------
# Pregunta 1: etapas
# ---------------------------------------------------------------------------
st.header("Pregunta 1 · Cuanto más profundo el sueño NREM, más regular el EEG")
st.caption("¿Cómo varían la complejidad y la actividad espectral del EEG entre las etapas de sueño?")

c1, c2 = st.columns(2)
c1.plotly_chart(plots.metric_by_stage(data, "sample_entropy"), width="stretch")
c2.plotly_chart(plots.metric_by_stage(data, "spectral_entropy"), width="stretch")
st.caption("Sample Entropy mide la impredecibilidad temporal de la señal; Spectral Entropy, cuán repartida "
           "está la potencia entre frecuencias. Son métricas distintas: no se comparan en la misma escala.")

st.subheader("La potencia se concentra en delta al llegar a N3")
rel = relative_power_table(data)
st.plotly_chart(plots.band_power_by_stage(rel, channel), width="stretch")
st.plotly_chart(plots.stage_band_heatmap(rel, channel), width="stretch")

st.subheader("Más delta, menos complejidad")
st.plotly_chart(plots.delta_vs_entropy(data, channel), width="stretch")
st.caption("Asociación descriptiva, no causal: ambas variables dependen de la etapa de sueño. "
           "Las diferencias observadas no implican significancia estadística.")

# ---------------------------------------------------------------------------
# Pregunta 2: edad
# ---------------------------------------------------------------------------
st.header("Pregunta 2 · Con la edad cae la actividad sigma de N2")
st.caption("¿Cómo cambia ese perfil con la edad? Cada punto es un sujeto. Elegí otra etapa o métrica para "
           "hacer otro corte del cubo.")
s1, s2 = st.columns(2)
slice_stage = s1.selectbox("Etapa", plots.STAGE_ORDER, index=plots.STAGE_ORDER.index("N2"))
slice_metric = s2.selectbox("Métrica", ["sigma_rel", "delta_rel", "sample_entropy", "spectral_entropy"],
                            format_func=lambda m: plots.METRIC_LABELS.get(m, m))
by_subject = (base[base["stage_code"] == slice_stage]
              .groupby(["subject_id", "age", "channel"], as_index=False)[slice_metric].mean())
if by_subject.empty:
    st.info("La selección no tiene épocas de esta etapa.")
else:
    st.plotly_chart(plots.sigma_vs_age(by_subject, slice_metric, slice_stage), width="stretch")
st.caption("Son 18 sujetos: es una tendencia descriptiva, no una estimación del efecto de la edad.")

st.subheader("Cómo se distribuye la noche")
hyp_recording = st.selectbox("Registro para el hipnograma", sorted(base["recording_id"].unique()))
st.plotly_chart(plots.hypnogram(base, hyp_recording), width="stretch")

with st.expander("Ver datos filtrados"):
    st.dataframe(data.drop(columns=["stage_order", "is_scored"]), width="stretch", hide_index=True)
