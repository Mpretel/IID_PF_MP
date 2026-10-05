"""Figuras Plotly compartidas por el notebook y el dashboard.

Las funciones reciben DataFrames obtenidos del Data Warehouse (src.queries),
nunca de los EDF. Cada figura responde una pregunta concreta (ver docstrings).

Paleta de etapas (validada: CVD protan/deutan ΔE ≥ 9.2 y visión normal ΔE ≥ 19.5
en todos los pares, OKLab×100): W naranja; N1→N3 rampa azul de claro a oscuro
(profundidad del sueño); REM aqua. N1 y REM tienen contraste < 3:1 contra el
fondo, por eso todas las figuras llevan leyenda y tooltip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import config

STAGE_ORDER = config.SCORED_STAGES  # W, N1, N2, N3, REM
STAGE_COLORS = {"W": "#eb6834", "N1": "#86b6ef", "N2": "#2a78d6", "N3": "#104281", "REM": "#1baf7a"}
BANDS = list(config.FREQ_BANDS)

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
DIVERGING = [[0.0, "#256abf"], [0.5, "#f0efec"], [1.0, "#e34948"]]  # azul ↔ gris ↔ rojo
CHANNEL_INK = {"Fpz-Cz": INK, "Pz-Oz": INK_MUTED}

METRIC_LABELS = {
    "sample_entropy": "Sample Entropy",
    "spectral_entropy": "Spectral Entropy (normalizada)",
    "delta_rel": "Potencia relativa delta",
    "alpha_rel": "Potencia relativa alfa",
    "delta_abs": "Potencia absoluta delta [µV²]",
    "total_power": "Potencia total 0.5–30 Hz [µV²]",
    "peak_to_peak": "Amplitud pico a pico [µV]",
    "rms": "RMS [µV]",
}


def _style(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    """Estilo común: fondo claro, grilla y ejes recesivos, tipografía del sistema."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=INK), x=0, xanchor="left", y=0.98, yanchor="top"),
        template="plotly_white", height=height,
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12, color=INK_SECONDARY),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        margin=dict(l=60, r=20, t=80, b=50),
        hoverlabel=dict(bgcolor="white", font_color=INK),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="outside", tickcolor=AXIS)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="outside", tickcolor=AXIS)
    return fig


def _with_alpha(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _scored(obs: pd.DataFrame) -> pd.DataFrame:
    return obs[obs["stage_code"].isin(STAGE_ORDER)]


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def stage_distribution(epochs_by_stage: pd.DataFrame) -> go.Figure:
    """¿Cuánto tiempo pasa en cada etapa? (resultado de la consulta epochs_by_stage)."""
    data = epochs_by_stage[epochs_by_stage["stage_code"].isin(STAGE_ORDER)]
    fig = go.Figure(go.Bar(
        x=data["stage_code"], y=data["n_epochs"],
        marker=dict(color=[STAGE_COLORS[s] for s in data["stage_code"]], cornerradius=4),
        text=[f"{p:.1f}%" for p in data["percent"]], textposition="outside",
        customdata=data[["hours", "percent"]],
        hovertemplate="%{x}: %{y} epochs<br>%{customdata[0]:.2f} h (%{customdata[1]:.1f}%)<extra></extra>",
    ))
    fig.update_layout(bargap=0.35, showlegend=False)
    fig.update_yaxes(title="Epochs de 30 s")
    return _style(fig, "Distribución de etapas de sueño (ventana de análisis)")


def hypnogram(obs: pd.DataFrame, recording_id: str) -> go.Figure:
    """¿Cómo se organiza el sueño a lo largo de la noche? Hipnograma de un registro."""
    rec = obs[(obs["recording_id"] == recording_id)].drop_duplicates("epoch_number").sort_values("epoch_number")
    y_order = ["N3", "N2", "N1", "REM", "W"]  # convención: vigilia arriba, sueño profundo abajo
    rec = rec.assign(level=rec["stage_code"].map({s: i for i, s in enumerate(y_order)}),
                     hour=rec["hours_since_sleep_onset"])
    scored = rec[rec["level"].notna()]
    fig = go.Figure(go.Scatter(
        x=scored["hour"], y=scored["level"], mode="lines", line=dict(color=INK_MUTED, width=1, shape="hv"),
        hoverinfo="skip", showlegend=False))
    for stage in STAGE_ORDER:
        part = scored[scored["stage_code"] == stage]
        fig.add_trace(go.Scatter(
            x=part["hour"], y=part["level"], mode="markers", name=stage,
            marker=dict(color=STAGE_COLORS[stage], size=6, symbol="square"),
            customdata=part[["epoch_number"]],
            hovertemplate=f"{stage}<br>%{{x:.2f}} h desde el inicio del sueño<br>epoch %{{customdata[0]}}<extra></extra>"))
    unscored = rec[rec["level"].isna()]
    if not unscored.empty:
        fig.add_trace(go.Scatter(
            x=unscored["hour"], y=[len(y_order) - 0.5] * len(unscored), mode="markers", name="Sin etapa (MOVE/?)",
            marker=dict(color=INK, size=7, symbol="x-thin", line=dict(width=1.5, color=INK)),
            hovertemplate="%{x:.2f} h · sin etapa asignada<extra></extra>"))
    fig.update_yaxes(tickvals=list(range(len(y_order))), ticktext=y_order, range=[-0.5, len(y_order)])
    fig.update_xaxes(title="Horas desde el inicio del sueño")
    return _style(fig, f"Hipnograma · {recording_id}", height=340)


def metric_by_stage(obs: pd.DataFrame, metric: str) -> go.Figure:
    """¿Cómo cambia la distribución de una métrica entre etapas? Violin + box por canal."""
    data = _scored(obs)
    channels = sorted(data["channel"].unique())
    fig = make_subplots(rows=1, cols=len(channels), shared_yaxes=True, subplot_titles=channels,
                        horizontal_spacing=0.04)
    for col, channel in enumerate(channels, start=1):
        for stage in STAGE_ORDER:
            values = data.loc[(data["channel"] == channel) & (data["stage_code"] == stage), metric]
            fig.add_trace(go.Violin(
                x=[stage] * len(values), y=values, name=stage, legendgroup=stage, showlegend=col == 1,
                line=dict(color=STAGE_COLORS[stage], width=1.5), fillcolor=_with_alpha(STAGE_COLORS[stage], 0.45),
                box=dict(visible=True, fillcolor=SURFACE, line=dict(color=INK_SECONDARY, width=1)),
                meanline=dict(visible=False), points=False, spanmode="hard",
                hovertemplate=f"{stage} · {channel}<br>%{{y:.3f}}<extra></extra>"), row=1, col=col)
    fig.update_yaxes(title=METRIC_LABELS.get(metric, metric), row=1, col=1)
    fig = _style(fig, f"{METRIC_LABELS.get(metric, metric)} por etapa de sueño", height=450)
    fig.update_layout(legend_y=1.1, margin_t=110)  # leyenda por encima de los títulos de panel
    return fig


def band_power_by_stage(rel_power: pd.DataFrame, channel: str) -> go.Figure:
    """¿Qué bandas dominan cada etapa? Potencia relativa media, un panel por banda
    (ejes Y independientes: delta ~0.8 y beta ~0.02 no son legibles en una misma escala)."""
    data = rel_power[(rel_power["channel"] == channel) & rel_power["stage_code"].isin(STAGE_ORDER)]
    fig = make_subplots(rows=1, cols=len(BANDS), subplot_titles=BANDS, horizontal_spacing=0.05)
    for col, band in enumerate(BANDS, start=1):
        fig.add_trace(go.Bar(
            x=data["stage_code"], y=data[band], showlegend=False,
            marker=dict(color=[STAGE_COLORS[s] for s in data["stage_code"]], cornerradius=3),
            hovertemplate=f"{band} · %{{x}}<br>%{{y:.3f}}<extra></extra>"), row=1, col=col)
    fig.update_layout(bargap=0.3)
    fig.update_yaxes(title="Potencia relativa media", row=1, col=1)
    return _style(fig, f"Potencia relativa por banda y etapa · {channel}", height=360)


def stage_band_heatmap(rel_power: pd.DataFrame, channel: str) -> go.Figure:
    """¿En qué etapa es cada banda relativamente más fuerte?

    Color: z-score de la potencia relativa media dentro de cada banda (columna),
    para que bandas de distinta magnitud sean comparables. Texto: valor real.
    """
    data = (rel_power[(rel_power["channel"] == channel) & rel_power["stage_code"].isin(STAGE_ORDER)]
            .set_index("stage_code").loc[STAGE_ORDER, BANDS])
    z = (data - data.mean()) / data.std(ddof=0)
    fig = go.Figure(go.Heatmap(
        z=z.values, x=BANDS, y=STAGE_ORDER, colorscale=DIVERGING, zmid=0,
        text=data.values, texttemplate="%{text:.3f}", textfont=dict(color=INK, size=12),
        xgap=2, ygap=2,
        colorbar=dict(title=dict(text="z (por banda)", side="right"), thickness=12, outlinewidth=0),
        hovertemplate="%{y} · %{x}<br>media relativa %{text:.3f}<br>z = %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    return _style(fig, f"Etapa × banda: potencia relativa · {channel}", height=380)


def delta_vs_entropy(obs: pd.DataFrame, channel: str, x: str = "delta_rel", y: str = "sample_entropy",
                     max_points_per_stage: int = 3000) -> go.Figure:
    """¿Se asocian la actividad delta y la complejidad? Scatter coloreado por etapa.

    Se submuestrea por etapa (semilla fija) solo para la visualización.
    Asociación ≠ causalidad: ambas variables dependen de la etapa.
    """
    data = _scored(obs[obs["channel"] == channel]).dropna(subset=[x, y])
    data = pd.concat([g.sample(min(len(g), max_points_per_stage), random_state=config.RANDOM_SEED)
                      for _, g in data.groupby("stage_code")])
    fig = go.Figure()
    for stage in STAGE_ORDER:
        part = data[data["stage_code"] == stage]
        fig.add_trace(go.Scattergl(
            x=part[x], y=part[y], mode="markers", name=stage,
            marker=dict(color=STAGE_COLORS[stage], size=5, opacity=0.55, line=dict(width=0)),
            customdata=part[["recording_id", "epoch_number", "channel"]],
            hovertemplate=(f"{stage}<br>{METRIC_LABELS.get(x, x)}: %{{x:.3f}}<br>{METRIC_LABELS.get(y, y)}: %{{y:.3f}}"
                           "<br>%{customdata[0]} · epoch %{customdata[1]} · %{customdata[2]}<extra></extra>")))
    fig.update_xaxes(title=METRIC_LABELS.get(x, x))
    fig.update_yaxes(title=METRIC_LABELS.get(y, y))
    return _style(fig, f"{METRIC_LABELS.get(x, x)} vs {METRIC_LABELS.get(y, y)} · {channel}", height=480)


def channel_comparison(channel_stats: pd.DataFrame, metric: str) -> go.Figure:
    """¿Difieren los canales frontal y posterior en cada etapa? (consulta channel_comparison)."""
    fig = go.Figure()
    for channel, part in channel_stats.groupby("channel"):
        part = part.set_index("stage_code").loc[[s for s in STAGE_ORDER if s in part["stage_code"].values]]
        color = CHANNEL_INK.get(channel, INK_SECONDARY)
        fig.add_trace(go.Scatter(
            x=part.index, y=part[metric], mode="lines+markers", name=channel,
            line=dict(color=color, width=2), marker=dict(size=9, color=color, line=dict(color=SURFACE, width=2)),
            hovertemplate=f"{channel} · %{{x}}<br>%{{y:.3f}}<extra></extra>"))
    fig.update_yaxes(title=METRIC_LABELS.get(metric, metric))
    return _style(fig, f"Comparación entre canales · {METRIC_LABELS.get(metric, metric)} (media)", height=380)


def subject_comparison(subject_stats: pd.DataFrame, metric: str, channel: str) -> go.Figure:
    """¿El patrón entre etapas se repite en cada sujeto? Una línea por sujeto, la media resaltada."""
    data = subject_stats[subject_stats["channel"] == channel]
    fig = go.Figure()
    for subject, part in data.groupby("subject_id"):
        part = part.set_index("stage_code").reindex(STAGE_ORDER)
        fig.add_trace(go.Scatter(
            x=STAGE_ORDER, y=part[metric], mode="lines+markers", name=f"Sujeto {subject}",
            line=dict(color=AXIS, width=1.5), marker=dict(size=6, color=AXIS), legendgroup="subjects",
            showlegend=False, hovertemplate=f"Sujeto {subject} · %{{x}}<br>%{{y:.3f}}<extra></extra>"))
    mean = data.groupby("stage_code")[metric].mean().reindex(STAGE_ORDER)
    fig.add_trace(go.Scatter(
        x=STAGE_ORDER, y=mean, mode="lines+markers", name="Media de sujetos",
        line=dict(color=INK, width=2.5), marker=dict(size=9, color=INK, line=dict(color=SURFACE, width=2)),
        hovertemplate="Media · %{x}<br>%{y:.3f}<extra></extra>"))
    fig.update_yaxes(title=METRIC_LABELS.get(metric, metric))
    return _style(fig, f"Comparación entre sujetos · {METRIC_LABELS.get(metric, metric)} · {channel} "
                       f"(gris: cada sujeto)", height=380)


def temporal_evolution(evolution: pd.DataFrame, metric: str, channel: str,
                       min_recording_fraction: float = 0.5) -> go.Figure:
    """¿Cómo evoluciona la métrica durante la noche? Promedio por hora desde el inicio del sueño.

    Solo se muestran las horas cubiertas por al menos `min_recording_fraction`
    de los registros: al final de la noche quedan pocos registros y su "media"
    sería en realidad el valor de una sola noche.
    """
    data = evolution[evolution["channel"] == channel]
    coverage = data.groupby("hour_bin")["recording_id"].nunique() / data["recording_id"].nunique()
    data = data[data["hour_bin"].isin(coverage[coverage >= min_recording_fraction].index)]
    fig = go.Figure()
    for rec, part in data.groupby("recording_id"):
        fig.add_trace(go.Scatter(
            x=part["hour_bin"], y=part[metric], mode="lines", name=rec, line=dict(color=AXIS, width=1.5),
            showlegend=False, hovertemplate=f"{rec} · hora %{{x}}<br>%{{y:.3f}}<extra></extra>"))
    mean = data.groupby("hour_bin")[metric].mean()
    fig.add_trace(go.Scatter(
        x=mean.index, y=mean.values, mode="lines+markers", name="Media de registros",
        line=dict(color=INK, width=2.5), marker=dict(size=8, color=INK, line=dict(color=SURFACE, width=2)),
        hovertemplate="Media · hora %{x}<br>%{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="Hora desde el inicio del sueño (−1 = vigilia previa)", dtick=1)
    fig.update_yaxes(title=METRIC_LABELS.get(metric, metric))
    return _style(fig, f"Evolución nocturna · {METRIC_LABELS.get(metric, metric)} · {channel} (gris: cada registro)",
                  height=380)


def amplitude_vs_threshold(obs: pd.DataFrame) -> go.Figure:
    """¿Dónde caen los valores extremos respecto del umbral de artefacto?

    Box plot de pico a pico por etapa y canal con el umbral configurado: muestra
    que los outliers estadísticos (puntos) quedan muy por debajo del umbral físico.
    """
    data = _scored(obs)
    fig = go.Figure()
    for stage in STAGE_ORDER:
        part = data[data["stage_code"] == stage]
        fig.add_trace(go.Box(
            x=part["channel"], y=part["peak_to_peak"], name=stage, marker=dict(color=STAGE_COLORS[stage], size=4),
            line=dict(width=1.2), boxpoints="outliers",
            hovertemplate=f"{stage} · %{{x}}<br>%{{y:.0f}} µV<extra></extra>"))
    fig.add_hline(y=config.ARTIFACT_MAX_PTP_UV, line=dict(color="#d03b3b", width=1.5),
                  annotation_text=f"Umbral de artefacto: {config.ARTIFACT_MAX_PTP_UV:.0f} µV",
                  annotation_position="top left", annotation_font_color=INK_SECONDARY)
    fig.update_layout(boxmode="group")
    fig.update_yaxes(title=METRIC_LABELS["peak_to_peak"], rangemode="tozero")
    return _style(fig, "Amplitud pico a pico por etapa y umbral de artefacto", height=420)
