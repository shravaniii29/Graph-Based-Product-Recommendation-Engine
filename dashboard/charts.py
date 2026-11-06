"""Plotly figure builders. Every chart takes real data (saved results or live output)."""
import math

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from theme import (ACCENT, BORDER, EVENT_COLORS, GRID, METHOD_COLORS, MUTED, NODE_COLORS,
                   SHORT_NAMES, TEXT)

FONT = "Inter, 'Segoe UI', system-ui, sans-serif"


def style(fig, height=380, title=None, legend=True):
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(size=15, color=TEXT), x=0.01, xanchor="left") if title else None,
        font=dict(family=FONT, size=13, color=TEXT),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", bordercolor=BORDER, font=dict(family=FONT, color=TEXT)),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER)
    return fig


def _color(method):
    return METHOD_COLORS.get(method, ACCENT)


# ------------------------------------------------------------------ home
def event_distribution(event_counts):
    names = list(event_counts.keys())
    vals = [event_counts[k] for k in names]
    total = sum(vals)
    labels = {"view": "view", "addtocart": "add to cart", "transaction": "transaction"}
    fig = go.Figure(go.Bar(
        x=[labels.get(n, n) for n in names], y=vals,
        marker_color=[EVENT_COLORS.get(labels.get(n, n), ACCENT) for n in names],
        text=[f"{v:,}<br>({100*v/total:.1f}%)" for v in vals], textposition="outside",
        hovertemplate="%{x}: %{y:,}<extra></extra>"))
    fig.update_yaxes(type="log", title="events (log scale)")
    return style(fig, 340, legend=False)


def data_funnel(stages):
    """stages: list of (label, value)."""
    fig = go.Figure(go.Funnel(
        y=[s[0] for s in stages], x=[s[1] for s in stages], textinfo="value",
        marker=dict(color=["#D9D1FA", "#C6DDF4", "#F9D3C3", "#C7EAD6", "#F6E3A8"][:len(stages)]),
        connector=dict(line=dict(color=BORDER))))
    return style(fig, 340, legend=False)


# ------------------------------------------------------------------ user
def history_bars(history, top=20):
    h = history.head(top).iloc[::-1]
    fig = go.Figure()
    for ev, color in EVENT_COLORS.items():
        sub = h[h["strongest_event"] == ev]
        if len(sub):
            fig.add_trace(go.Bar(
                y=[f"item {i}" for i in sub["itemid"]], x=sub["weight"], orientation="h", name=ev,
                marker_color=color,
                customdata=np.stack([sub["n_events"], sub["raw_weight"]], axis=1),
                hovertemplate="%{y}<br>weight %{x} (raw %{customdata[1]})<br>%{customdata[0]} events<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="interaction weight (capped at 15)")
    fig.update_yaxes(categoryorder="array", categoryarray=[f"item {i}" for i in h["itemid"]])
    return style(fig, max(300, 22 * len(h) + 80), title=f"Strongest {len(h)} products in the history")


def event_mix(history):
    counts = history["strongest_event"].value_counts()
    fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.6,
                           marker=dict(colors=[EVENT_COLORS.get(e, ACCENT) for e in counts.index]),
                           textinfo="label+value", hovertemplate="%{label}: %{value} products<extra></extra>"))
    return style(fig, 300, title="Strongest action per product", legend=False)


# ------------------------------------------------------------------ recommendations
def rec_scores(df, method):
    d = df.iloc[::-1]
    colors = ["#2E8B57" if hit else _color(method) for hit in d["in_future_interactions"]]
    fig = go.Figure(go.Bar(
        y=[f"#{r} · item {i}" for r, i in zip(d["rank"], d["itemid"])], x=d["score"], orientation="h",
        marker_color=colors, hovertemplate="%{y}<br>score %{x:.4g}<extra></extra>"))
    fig.update_xaxes(title="score")
    return style(fig, 280, title=f"{SHORT_NAMES.get(method, method)} scores", legend=False)


def similar_users_bar(sim):
    d = sim.iloc[::-1]
    fig = go.Figure(go.Bar(
        y=[f"user {u}" for u in d["user"]], x=d["shared_products"], orientation="h",
        marker_color=METHOD_COLORS["Common Neighbors"],
        customdata=d["their_products"], hovertemplate="%{y}<br>%{x} shared products<br>"
                                                       "%{customdata} products in total<extra></extra>"))
    fig.update_xaxes(title="common neighbours = shared products")
    return style(fig, max(260, 26 * len(d) + 70), title="Most similar users", legend=False)


# ------------------------------------------------------------------ comparison
def metric_bars(summary, metrics, methods):
    fig = go.Figure()
    s = summary.set_index("method")
    for m in methods:
        if m not in s.index:
            continue
        vals = [s.loc[m, k] for k in metrics]
        lo = [s.loc[m, f"{k}_ci_low"] if f"{k}_ci_low" in s.columns else np.nan for k in metrics]
        hi = [s.loc[m, f"{k}_ci_high"] if f"{k}_ci_high" in s.columns else np.nan for k in metrics]
        fig.add_trace(go.Bar(
            name=SHORT_NAMES.get(m, m), x=metrics, y=vals, marker_color=_color(m),
            error_y=dict(type="data", symmetric=False, array=np.subtract(hi, vals), arrayminus=np.subtract(vals, lo),
                         color=MUTED, thickness=1.2, width=4),
            hovertemplate="%{x}: %{y:.4f}<extra>" + SHORT_NAMES.get(m, m) + "</extra>"))
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(title="mean over evaluated users (95% CI)")
    return style(fig, 420)


def metric_ci_dots(summary, metric, methods):
    s = summary.set_index("method")
    ms = [m for m in methods if m in s.index]
    vals = [s.loc[m, metric] for m in ms]
    lo = [s.loc[m, f"{metric}_ci_low"] for m in ms]
    hi = [s.loc[m, f"{metric}_ci_high"] for m in ms]
    fig = go.Figure(go.Scatter(
        x=vals, y=[SHORT_NAMES.get(m, m) for m in ms], mode="markers",
        marker=dict(size=16, color=[_color(m) for m in ms], line=dict(width=1.5, color="white")),
        error_x=dict(type="data", symmetric=False, array=np.subtract(hi, vals), arrayminus=np.subtract(vals, lo),
                     color=MUTED, thickness=2, width=8),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title=f"{metric} with 95% bootstrap CI", rangemode="tozero")
    fig.update_yaxes(autorange="reversed")
    return style(fig, 300, legend=False)


def time_bars(summary, methods):
    s = summary.set_index("method")
    ms = [m for m in methods if m in s.index]
    vals = [s.loc[m, "avg_time_ms"] for m in ms]
    fig = go.Figure(go.Bar(
        x=[SHORT_NAMES.get(m, m) for m in ms], y=vals, marker_color=[_color(m) for m in ms],
        text=[f"{v:.2f} ms" if v < 10 else f"{v:,.0f} ms" for v in vals], textposition="outside",
        hovertemplate="%{x}: %{y:.3f} ms<extra></extra>"))
    fig.update_yaxes(type="log", title="avg ms per user (log scale)")
    return style(fig, 360, title="Average recommendation time", legend=False)


def time_boxes(per_user, methods):
    fig = go.Figure()
    for m in methods:
        sub = per_user[per_user["method"] == m]
        if len(sub):
            fig.add_trace(go.Box(y=sub["time_ms"], name=SHORT_NAMES.get(m, m), marker_color=_color(m),
                                 boxpoints=False, line=dict(width=1.5)))
    fig.update_yaxes(type="log", title="ms per user (log scale)")
    return style(fig, 360, title="Per-user time distribution", legend=False)


def overlap_heatmap(mat):
    labels = [SHORT_NAMES.get(m, m) for m in mat.index]
    z = mat.values.astype(float)
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, zmin=0, zmax=1,
        colorscale=[[0, "#FFFFFF"], [0.5, "#D9D1FA"], [1, "#8B7FD9"]],
        text=[[f"{v:.2f}" if not math.isnan(v) else "" for v in row] for row in z],
        texttemplate="%{text}", hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>", showscale=False))
    fig.update_yaxes(autorange="reversed")
    return style(fig, 340, title="Average overlap of top-5 lists (Jaccard)", legend=False)


def hit_distribution(per_user, methods):
    fig = go.Figure()
    for m in methods:
        sub = per_user[per_user["method"] == m]
        if len(sub):
            hits = (sub["precision@5"] * 5).round().astype(int).value_counts().sort_index()
            fig.add_trace(go.Bar(name=SHORT_NAMES.get(m, m), x=[str(i) for i in hits.index], y=hits.values,
                                 marker_color=_color(m)))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="correct items in the top-5")
    fig.update_yaxes(title="users", type="log")
    return style(fig, 340, title="How many of the 5 recommendations were correct")


# ------------------------------------------------------------------ graph
def network(H, center, rec_nodes, full_degree, seed=42):
    pos = nx.spring_layout(H, seed=seed, weight="weight")
    rec_set = set(rec_nodes)
    fig = go.Figure()
    # edges: user history edges darker, others light; width ~ weight
    for a, b, d in H.edges(data=True):
        is_center = center in (a, b)
        is_rec = a in rec_set or b in rec_set
        color = "#9B8CE6" if is_center else ("#6CC497" if is_rec else "#D5D2E3")
        fig.add_trace(go.Scatter(x=[pos[a][0], pos[b][0]], y=[pos[a][1], pos[b][1]], mode="lines",
                                 line=dict(width=0.8 + d["weight"] / 4, color=color),
                                 hoverinfo="skip", showlegend=False))

    def group(nodes, name, color, symbol, size):
        nodes = list(nodes)
        if not nodes:
            return
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text", name=name,
            marker=dict(size=size, color=color, symbol=symbol, line=dict(width=1.5, color="white")),
            text=[n for n in nodes], textposition="top center", textfont=dict(size=9, color=MUTED),
            customdata=[full_degree(n) for n in nodes],
            hovertemplate="%{text}<br>connections in full graph: %{customdata}<extra>" + name + "</extra>"))

    users = [n for n in H if H.nodes[n]["node_type"] == "user" and n != center]
    prods = [n for n in H if H.nodes[n]["node_type"] == "product" and n not in rec_set]
    group(users, "Other users", NODE_COLORS["user"], "circle", 13)
    group(prods, "Products", NODE_COLORS["product"], "square", 13)
    group([n for n in rec_nodes if n in H], "Recommended products", NODE_COLORS["recommended"], "star", 20)
    group([center], "Selected user", NODE_COLORS["center"], "circle", 24)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, 620)


def degree_scatter(u_deg, p_deg):
    fig = go.Figure()
    for deg, name, color in [(u_deg, "Users", NODE_COLORS["user"]), (p_deg, "Products", NODE_COLORS["product"])]:
        if len(deg):
            vc = pd.Series(deg).value_counts().sort_index()
            fig.add_trace(go.Scatter(x=vc.index, y=vc.values, mode="markers", name=name,
                                     marker=dict(size=7, color=color, line=dict(width=0.5, color="#888")),
                                     hovertemplate="degree %{x}: %{y} nodes<extra>" + name + "</extra>"))
    fig.update_xaxes(type="log", title="degree (connections)")
    fig.update_yaxes(type="log", title="number of nodes")
    return style(fig, 360, title="Degree distribution (log-log)")


# ------------------------------------------------------------------ scalability
def scal_lines(long_df, x, y, x_title, y_title, title, log_y=False):
    fig = go.Figure()
    for m, sub in long_df.groupby("method", sort=False):
        sub = sub.sort_values(x)
        fig.add_trace(go.Scatter(x=sub[x], y=sub[y], mode="lines+markers", name=SHORT_NAMES.get(m, m),
                                 line=dict(color=_color(m), width=3), marker=dict(size=9),
                                 customdata=sub["dataset_pct"],
                                 hovertemplate="%{customdata}% of users<br>%{x:,}<br>%{y:.4g}<extra></extra>"))
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title, type="log" if log_y else "linear")
    return style(fig, 360, title=title)


def scal_pair(scal, x, series, x_title, y_title, title):
    """series: list of (column, name, colour, dash)."""
    fig = go.Figure()
    s = scal.sort_values(x)
    for col, name, color, dash in series:
        if col in s.columns:
            fig.add_trace(go.Scatter(x=s[x], y=s[col], mode="lines+markers", name=name,
                                     line=dict(color=color, width=3, dash=dash), marker=dict(size=9),
                                     customdata=s["dataset_pct"],
                                     hovertemplate="%{customdata}% of users<br>%{x:,}<br>%{y:.4g}<extra>"
                                                   + name + "</extra>"))
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)
    return style(fig, 360, title=title)
