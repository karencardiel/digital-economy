import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ==========================================
# 1. DATA PREPARATION
# ==========================================
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "digital_economy_drs.csv")
df = pd.read_csv(DATA_PATH)

# Translate countries to English
df["country"] = df["country"].replace({"México": "Mexico", "Reino Unido": "United Kingdom"})

INDICATOR_NAMES_EN = {
    "N_IND_01": "Fixed Broadband ('24)", "N_IND_02": "4G Coverage ('24)", "N_IND_03": "Internet Users ('24)",
    "N_IND_04": "Affordability ('24)", "N_IND_05": "ICT Exports ('24)", "N_IND_06": "Digital Services ('24)",
    "N_IND_07": "Patents ('21)", "N_IND_08": "R&D Spend ('20-24)"
}
norm_cols = list(INDICATOR_NAMES_EN.keys())

# Clustering PCA
X = df[norm_cols].fillna(0)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)
leader_cluster = df.groupby("Cluster")["DRS_100"].mean().idxmax()
df["Profile"] = df["Cluster"].apply(lambda c: "Leaders" if c == leader_cluster else "In Transition")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df["PCA_1"], df["PCA_2"] = X_pca[:, 0], X_pca[:, 1]
df_sorted = df.sort_values("DRS_100", ascending=False).reset_index(drop=True)

# ==========================================
# 2. STYLES & THEME
# ==========================================
COLORS = {
    "bg": "#0B1121", "card": "#1E293B", "card_hover": "#334155",
    "accent": "#0EA5E9", "accent_light": "#38BDF8", "accent_purple": "#8B5CF6", "accent_green": "#10B981",
    "text_main": "#F8FAFC", "text_muted": "#94A3B8", "border": "#334155", "red": "#EF4444", "orange": "#F59E0B"
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "HealthTech Digital Readiness"
server = app.server  # Necesario para deployment (Render/Gunicorn)

def icon(name, color=COLORS["text_muted"], size=20):
    return DashIconify(icon=name, color=color, width=size, height=size, className="me-2")

def get_color_intensity(val):
    if val >= 0.8: return COLORS["accent_green"]
    if val >= 0.5: return COLORS["accent"]
    if val >= 0.2: return COLORS["accent_purple"]
    return COLORS["red"]

card_style = {"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}", "borderRadius": "12px", "padding": "20px"}
layout_bg = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=COLORS["text_muted"])

# ==========================================
# 3. LAYOUT STRUCTURE
# ==========================================
app.layout = html.Div([
    # HEADER
    html.Div([
        html.Div([
            html.Div([
                DashIconify(icon="healthicons:health-worker", color=COLORS["accent_light"], width=32, className="me-2"),
                html.H4("HealthTech Digital Readiness Intelligence", style={"color": COLORS["text_main"], "margin": 0, "fontWeight": "700"})
            ], className="d-flex align-items-center"),
            html.Div([
                html.Span("Target Economy:", style={"color": COLORS["text_muted"], "marginRight": "10px"}),
                dcc.Dropdown(
                    id="country-selector",
                    options=[{"label": c, "value": c} for c in df_sorted["country"]],
                    value="Mexico",
                    clearable=False,
                    style={"minWidth": "200px", "color": "#000"} 
                )
            ], className="d-flex align-items-center")
        ], className="d-flex justify-content-between align-items-center w-100")
    ], style={"padding": "20px 40px", "borderBottom": f"1px solid {COLORS['border']}", "backgroundColor": COLORS["bg"]}),

    # MAIN CONTENT
    html.Div([
        # TOP ROW CARDS
        dbc.Row([
            dbc.Col(id="focus-card-container", md=5),
            dbc.Col(id="benchmark-card-container", md=4),
            dbc.Col(id="ai4c-card-container", md=3)
        ], className="mb-4 align-items-stretch"),
        
        # BOTTOM ROW: MATRIX & TABS
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5([icon("ph:grid-four-fill", color=COLORS["text_main"]), "HealthTech Readiness Matrix"], style={"color": COLORS["text_main"], "marginBottom": "20px", "fontWeight": "600"}),
                    
                    dbc.Tabs([
                        dbc.Tab(html.Div(id="heatmap-container", className="mt-4"), label="Heatmap Matrix", tab_style={"backgroundColor": "transparent", "border": "none"}, active_label_style={"color": COLORS["accent"], "borderBottom": f"2px solid {COLORS['accent']}", "backgroundColor": "transparent"}),
                        dbc.Tab(dcc.Graph(id="radar-chart", style={"height": "350px"}), label="Radar Benchmark", tab_style={"backgroundColor": "transparent", "border": "none"}, active_label_style={"color": COLORS["accent"], "borderBottom": f"2px solid {COLORS['accent']}", "backgroundColor": "transparent"}),
                        dbc.Tab(dcc.Graph(id="stacked-bar-chart", style={"height": "350px"}), label="Pillar Breakdown", tab_style={"backgroundColor": "transparent", "border": "none"}, active_label_style={"color": COLORS["accent"], "borderBottom": f"2px solid {COLORS['accent']}", "backgroundColor": "transparent"}),
                        dbc.Tab(dcc.Graph(id="scatter-chart", style={"height": "350px"}), label="R&D vs Exports", tab_style={"backgroundColor": "transparent", "border": "none"}, active_label_style={"color": COLORS["accent"], "borderBottom": f"2px solid {COLORS['accent']}", "backgroundColor": "transparent"}),
                        dbc.Tab(dcc.Graph(id="pca-chart", style={"height": "350px"}), label="Clustering PCA", tab_style={"backgroundColor": "transparent", "border": "none"}, active_label_style={"color": COLORS["accent"], "borderBottom": f"2px solid {COLORS['accent']}", "backgroundColor": "transparent"}),
                    ]),
                    
                ], style=card_style)
            ], md=8),
            
            # INSIGHTS TEXT BOXES
            dbc.Col(id="insights-container", md=4)
        ])
    ], style={"padding": "30px 40px"})

], style={"backgroundColor": COLORS["bg"], "minHeight": "100vh", "fontFamily": "'Segoe UI', Roboto, Helvetica, Arial, sans-serif"})

# ==========================================
# 4. CALLBACKS FOR DYNAMIC CONTENT
# ==========================================
@app.callback(
    [Output("focus-card-container", "children"),
     Output("benchmark-card-container", "children"),
     Output("ai4c-card-container", "children"),
     Output("heatmap-container", "children"),
     Output("radar-chart", "figure"),
     Output("stacked-bar-chart", "figure"),
     Output("scatter-chart", "figure"),
     Output("pca-chart", "figure"),
     Output("insights-container", "children")],
    [Input("country-selector", "value")]
)
def update_dashboard(selected_country):
    
    country_data = df_sorted[df_sorted["country"] == selected_country].iloc[0]
    rank = df_sorted[df_sorted["country"] == selected_country].index[0] + 1
    
    # Calculate Dynamic Strengths and Gaps
    scores = {INDICATOR_NAMES_EN[col]: country_data[col] for col in norm_cols}
    strength_name = max(scores, key=scores.get)
    gap_name = min(scores, key=scores.get)
    strength_val = scores[strength_name] * 100
    gap_val = scores[gap_name] * 100
    
    # ----------------------------------------
    # 1. FOCUS CARD (LEFT)
    # ----------------------------------------
    focus_card = html.Div([
        html.Div([
            html.Div([
                icon("ph:flag-fill", color=COLORS["accent"], size=32),
                html.H2(selected_country, style={"color": COLORS["text_main"], "margin": "0 10px", "fontWeight": "bold"}),
                html.Span(f"Rank #{rank}", className="badge", style={"backgroundColor": f"{COLORS['accent']}30", "color": COLORS["accent"], "border": f"1px solid {COLORS['accent']}50"})
            ], className="d-flex align-items-center"),
            html.Div([
                html.H1(f"{country_data['DRS_100']:.2f}", style={"color": COLORS["accent_green"], "margin": 0, "fontWeight": "bold"}),
                html.Small("DRS SCORE", style={"color": COLORS["text_muted"], "textAlign": "right", "display": "block"})
            ])
        ], className="d-flex justify-content-between align-items-start mb-4"),
        
        html.Div([
            html.Div([html.H5(f"{int(country_data['N_IND_03']*100)}%", style={"color": COLORS["text_main"], "margin": 0}), html.Small("Internet Users ('24)", style={"color": COLORS["text_muted"], "fontSize": "11px"})]),
            html.Div([html.H5(f"{int(country_data['N_IND_07']*100)}", style={"color": COLORS["text_main"], "margin": 0}), html.Small("Patents ('21)", style={"color": COLORS["text_muted"], "fontSize": "11px"})]),
            html.Div([html.H5(f"{int(country_data['N_IND_08']*100)}%", style={"color": COLORS["text_main"], "margin": 0}), html.Small("R&D Spend ('20-24)", style={"color": COLORS["text_muted"], "fontSize": "11px"})]),
            html.Div([html.H5(f"{int(country_data['N_IND_06']*100)}%", style={"color": COLORS["text_main"], "margin": 0}), html.Small("Digital Services ('24)", style={"color": COLORS["text_muted"], "fontSize": "11px"})])
        ], className="d-flex justify-content-between mb-4", style={"backgroundColor": COLORS["bg"], "padding": "15px", "borderRadius": "8px"}),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([icon("ph:check-circle-fill", color=COLORS["accent_green"]), html.Span("Primary Strength", style={"color": COLORS["text_main"], "fontWeight": "600", "fontSize": "13px"})]),
                    html.P(f"Strong performance in {strength_name} ({strength_val:.1f}%). Enables digital health reach.", style={"color": COLORS["text_muted"], "fontSize": "12px", "marginTop": "8px"})
                ], style={"backgroundColor": f"{COLORS['accent_green']}10", "padding": "12px", "borderRadius": "8px", "border": f"1px solid {COLORS['accent_green']}30", "height": "100%"})
            ], width=6),
            dbc.Col([
                html.Div([
                    html.Div([icon("ph:warning-fill", color=COLORS["red"]), html.Span("Critical Gap", style={"color": COLORS["text_main"], "fontWeight": "600", "fontSize": "13px"})]),
                    html.P(f"Low metrics in {gap_name} ({gap_val:.1f}%). Limits local medical innovation and HealthTech.", style={"color": COLORS["text_muted"], "fontSize": "12px", "marginTop": "8px"})
                ], style={"backgroundColor": f"{COLORS['red']}10", "padding": "12px", "borderRadius": "8px", "border": f"1px solid {COLORS['red']}30", "height": "100%"})
            ], width=6)
        ])
    ], style=card_style, className="h-100")

    # ----------------------------------------
    # 2. BENCHMARK CARD (MIDDLE)
    # ----------------------------------------
    b_rows = []
    for i, row in df_sorted.iterrows():
        r = i + 1
        color = COLORS["accent_green"] if r == 1 else (COLORS["accent"] if row["country"] == selected_country else COLORS["accent_purple"])
        b_rows.append(
            html.Div([
                html.Div([
                    html.Small(f"#{r}", className="text-muted fw-bold me-2"),
                    html.Span(row["country"], style={"color": COLORS["text_main"], "fontWeight": "600", "fontSize": "14px"}),
                ], className="d-flex align-items-center mb-1"),
                html.Div([
                    html.Div(style={"width": f"{row['DRS_100']}%", "backgroundColor": color, "height": "8px", "borderRadius": "4px", "boxShadow": f"0 0 8px {color}80"}),
                    html.Span(f"{row['DRS_100']:.1f} pts", style={"fontSize": "11px", "color": COLORS["text_muted"], "marginLeft": "8px", "minWidth": "45px"})
                ], className="d-flex align-items-center mb-3")
            ])
        )
        
    benchmark_card = html.Div([
        html.Div([
            icon("ph:trophy-fill", color=COLORS["text_main"], size=24),
            html.Div([
                html.H5("DRS Overall Benchmark", style={"color": COLORS["text_main"], "margin": 0, "fontWeight": "600"}),
                html.Small("Synthetic readiness score (8 normalized dims)", style={"color": COLORS["text_muted"]})
            ], className="ms-2")
        ], className="d-flex align-items-center mb-4"),
        html.Div(b_rows)
    ], style=card_style, className="h-100")

    # ----------------------------------------
    # 3. AI 4C CARD (RIGHT) - Compute removed visually
    # ----------------------------------------
    conn_val = (country_data["N_IND_01"] + country_data["N_IND_02"]) / 2
    comp_val = (country_data["N_IND_07"] + country_data["N_IND_08"]) / 2
    ctx_val = country_data["N_IND_06"]
    
    def get_4c_status(val):
        if val > 0.75: return "High", COLORS["accent_green"]
        if val > 0.4: return "Moderate", COLORS["accent"]
        return "Low", COLORS["red"]
        
    st_conn, col_conn = get_4c_status(conn_val)
    st_ctx, col_ctx = get_4c_status(ctx_val)
    st_comp, col_comp = get_4c_status(comp_val)

    ai4c_card = html.Div([
        html.Div([
            icon("ph:brain-fill", color=COLORS["accent_purple"], size=24),
            html.H5("AI 4C Diagnostic", style={"color": COLORS["text_main"], "margin": "0 0 0 8px", "fontWeight": "600"})
        ], className="d-flex align-items-center mb-4"),
        
        html.Div([
            html.Div([html.Span("Connectivity (Networks)", style={"color": COLORS["text_main"]}), html.Span(st_conn, style={"color": col_conn, "fontSize": "12px"})], className="d-flex justify-content-between mb-1"),
            html.Div(style={"width": "100%", "backgroundColor": COLORS["bg"], "height": "6px", "borderRadius": "3px", "marginBottom": "20px"}, children=[html.Div(style={"width": f"{conn_val*100}%", "backgroundColor": col_conn, "height": "100%", "borderRadius": "3px"})]),
            
            html.Div([html.Span("Context (Digital Services)", style={"color": COLORS["text_main"]}), html.Span(st_ctx, style={"color": col_ctx, "fontSize": "12px"})], className="d-flex justify-content-between mb-1"),
            html.Div(style={"width": "100%", "backgroundColor": COLORS["bg"], "height": "6px", "borderRadius": "3px", "marginBottom": "20px"}, children=[html.Div(style={"width": f"{ctx_val*100}%", "backgroundColor": col_ctx, "height": "100%", "borderRadius": "3px"})]),
            
            html.Div([html.Span("Competency (R&D & IP)", style={"color": COLORS["text_main"]}), html.Span(st_comp, style={"color": col_comp, "fontSize": "12px"})], className="d-flex justify-content-between mb-1"),
            html.Div(style={"width": "100%", "backgroundColor": COLORS["bg"], "height": "6px", "borderRadius": "3px", "marginBottom": "20px"}, children=[html.Div(style={"width": f"{comp_val*100}%", "backgroundColor": col_comp, "height": "100%", "borderRadius": "3px"})]),
        ]),
        
        html.Div([
            icon("ph:lightbulb-fill", color=COLORS["accent_green"]),
            html.Span("Takeaway: Evaluated against the 4C AI Framework, the dataset lacks 'Compute' metrics (server/cloud capacity). Without it, assessing true AI readiness for medical applications remains incomplete.", style={"color": COLORS["text_main"], "fontSize": "11px", "marginLeft": "8px", "lineHeight": "1.2"})
        ], className="d-flex mt-4", style={"backgroundColor": f"{COLORS['accent_green']}10", "padding": "10px", "borderRadius": "6px"})
        
    ], style=card_style, className="h-100")

    # ----------------------------------------
    # 4. HEATMAP TABLE (BOTTOM LEFT)
    # ----------------------------------------
    header = html.Tr([html.Th("Economy", style={"color": COLORS["text_muted"], "fontWeight": "normal", "padding": "12px", "fontSize": "13px"})] + 
                     [html.Th(name, style={"color": COLORS["text_muted"], "fontWeight": "normal", "fontSize": "10px", "padding": "12px", "textAlign": "center"}) for name in INDICATOR_NAMES_EN.values()] + 
                     [html.Th("DRS", style={"color": COLORS["accent_green"], "padding": "12px", "textAlign": "center"})],
                     style={"borderBottom": f"1px solid {COLORS['border']}"})
    
    h_rows = []
    for _, row in df_sorted.iterrows():
        cells = [html.Td(html.Strong(row["country"]), style={"padding": "12px", "color": COLORS["text_main"], "fontSize": "14px"})]
        for col in norm_cols:
            val = row[col]
            val_100 = int(val * 100) if pd.notna(val) else 0
            bg_color = get_color_intensity(val) if pd.notna(val) else COLORS["card_hover"]
            badge = html.Div(f"{val_100}", style={
                "backgroundColor": bg_color + "30", "color": bg_color, "border": f"1px solid {bg_color}50",
                "borderRadius": "6px", "padding": "4px 8px", "textAlign": "center", "fontWeight": "600", "fontSize": "12px",
                "width": "35px", "margin": "0 auto"
            })
            cells.append(html.Td(badge, style={"padding": "6px"}))
        cells.append(html.Td(f"{row['DRS_100']:.1f}", style={"color": COLORS["accent_green"], "fontWeight": "bold", "textAlign": "center"}))
        h_rows.append(html.Tr(cells, style={"borderBottom": f"1px solid {COLORS['border']}" if row['country'] != df_sorted.iloc[-1]['country'] else "none"}))
        
    heatmap_html = html.Table([html.Thead(header), html.Tbody(h_rows)], style={"width": "100%", "borderCollapse": "collapse"})

    # ----------------------------------------
    # 5. RADAR CHART (PLOTLY)
    # ----------------------------------------
    categories = list(INDICATOR_NAMES_EN.values())
    fig_radar = go.Figure()
    
    for _, row in df_sorted.iterrows():
        is_focus = row['country'] == selected_country
        c_color = COLORS["accent"] if is_focus else COLORS["text_muted"]
        
        # We fillna(0) to avoid broken polygons, then we append the first element to close the loop
        filled_row = row.fillna(0)
        r_vals = [filled_row[c] for c in norm_cols]
        r_vals.append(r_vals[0])
        
        fig_radar.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=categories + [categories[0]],
            fill='toself' if is_focus else 'none',
            name=row['country'],
            line=dict(color=c_color, width=3 if is_focus else 1),
            opacity=1 if is_focus else 0.3
        ))
    
    # We set range to [-0.1, 1.0] to prevent zero-values from completely collapsing into an invisible center dot
    fig_radar.update_layout(**layout_bg, polar=dict(radialaxis=dict(visible=False, range=[-0.1, 1.0]), bgcolor="rgba(0,0,0,0)"), margin=dict(l=50, r=50, t=30, b=30), legend=dict(orientation="h", y=-0.1))

    # ----------------------------------------
    # 6. STACKED BAR CHART (PLOTLY)
    # ----------------------------------------
    w_cols = [c.replace("N_", "W_") for c in norm_cols]
    df_stack = df_sorted[["country"] + w_cols].copy()
    df_stack.columns = ["Economy"] + categories
    
    bar_colors = [COLORS["accent_green"], COLORS["accent"], COLORS["accent_purple"], COLORS["accent_light"], "#F59E0B", "#EC4899", "#3B82F6", "#8B5CF6"]
    
    fig_stacked = px.bar(df_stack, y="Economy", x=categories, orientation="h", color_discrete_sequence=bar_colors)
    fig_stacked.update_layout(**layout_bg, margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h", y=-0.2, title=""), xaxis=dict(showgrid=False, zeroline=False, visible=False), yaxis=dict(title=""))

    # ----------------------------------------
    # 7. SCATTER CHART (R&D VS EXPORTS)
    # ----------------------------------------
    fig_scatter = px.scatter(df, x="N_IND_08", y="N_IND_06", color="Profile", text="country", size="DRS_100",
                             color_discrete_map={"Leaders": COLORS["accent_green"], "In Transition": COLORS["accent"]},
                             labels={"N_IND_08": "Normalized R&D Spend", "N_IND_06": "Normalized Digital Services Exports"},
                             size_max=25)
    fig_scatter.update_traces(textposition='top right', marker=dict(line=dict(width=1, color=COLORS["bg"])))
    fig_scatter.update_layout(**layout_bg, margin=dict(l=0, r=0, t=20, b=0), xaxis=dict(showgrid=True, gridcolor=COLORS["border"]), yaxis=dict(showgrid=True, gridcolor=COLORS["border"]))

    # ----------------------------------------
    # 8. PCA CLUSTERING
    # ----------------------------------------
    fig_pca = px.scatter(df, x="PCA_1", y="PCA_2", color="Profile", text="country",
                         color_discrete_map={"Leaders": COLORS["accent_green"], "In Transition": COLORS["accent"]}, size_max=15)
    fig_pca.update_traces(marker=dict(size=14, line=dict(width=1, color=COLORS["bg"])), textposition='top right')
    
    # Add dynamic annotations explaining the positions
    try:
        mex_col_x = df[df['country'].isin(['Mexico', 'Colombia'])]['PCA_1'].mean()
        mex_col_y = df[df['country'].isin(['Mexico', 'Colombia'])]['PCA_2'].max() + 0.1
        fig_pca.add_annotation(x=mex_col_x, y=mex_col_y, text="Focus: High 4G, Low IP/Adoption", showarrow=False, font=dict(color=COLORS['text_muted'], size=11))
        
        chile_x = df[df['country']=='Chile']['PCA_1'].values[0]
        chile_y = df[df['country']=='Chile']['PCA_2'].values[0] - 0.1
        fig_pca.add_annotation(x=chile_x, y=chile_y, text="Focus: High Adoption, Low Relative 4G", showarrow=False, font=dict(color=COLORS['text_muted'], size=11))
    except Exception:
        pass

    fig_pca.update_layout(
        **layout_bg, 
        margin=dict(l=20, r=20, t=40, b=40), 
        xaxis=dict(title="← Lower General Readiness ------ [ PCA Component 1 ] ------ Higher General Readiness →", showgrid=True, gridcolor=COLORS["border"], zeroline=True, zerolinecolor=COLORS["border"]), 
        yaxis=dict(title="Structural Variance (PCA 2)", showgrid=True, gridcolor=COLORS["border"], zeroline=True, zerolinecolor=COLORS["border"])
    )

    # ----------------------------------------
    # 9. INSIGHTS TEXT (DYNAMIC HEALTH FOCUS)
    # ----------------------------------------
    insight_text_1 = f"The selected economy, **{selected_country}**, shows a primary strength in **{strength_name}** but critically lags in **{gap_name}**. "
    if country_data["Profile"] == "Leaders":
        insight_text_1 += "It is positioned as a digital leader, demonstrating the robust structural readiness required to deploy nationwide e-health portals and telemedicine networks."
    else:
        insight_text_1 += "It remains in transition. While its basic connectivity is expanding, the foundation for advanced digital healthcare relies on importing technology rather than creating it."

    insights = html.Div([
        html.Div([
            html.H6([icon("ph:eye-fill", color=COLORS["accent"]), "What do we observe?"], style={"color": COLORS["text_main"], "fontWeight": "600"}),
            html.P(f"When examining {selected_country}, the heatmap gradients reveal its structural capacity. {insight_text_1}", style={"color": COLORS["text_muted"], "fontSize": "13px"})
        ], style={"backgroundColor": COLORS["bg"], "padding": "20px", "borderRadius": "8px", "marginBottom": "15px"}),
        
        html.Div([
            html.H6([icon("ph:heartbeat-fill", color=COLORS["accent_purple"]), "HealthTech Implication"], style={"color": COLORS["text_main"], "fontWeight": "600"}),
            html.P("These macro gaps mean that the capacity to deploy cutting-edge HealthTech—such as AI diagnostics, remote patient monitoring, and unified Electronic Health Records (EHR)—remains highly concentrated in Northern Europe (like Estonia). Emerging economies show high mobile adoption, but lack the R&D and affordability to sustain native digital healthcare innovations.", style={"color": COLORS["text_muted"], "fontSize": "13px"})
        ], style={"backgroundColor": COLORS["bg"], "padding": "20px", "borderRadius": "8px"})
    ], style=card_style, className="h-100")

    return focus_card, benchmark_card, ai4c_card, heatmap_html, fig_radar, fig_stacked, fig_scatter, fig_pca, insights

if __name__ == '__main__':
    app.run(debug=True, port=8050)
