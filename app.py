"""Autonomous Perception & Variable-Resolution Grid Dashboard.

DRDO Problem Statement PS26053:
Real-time LiDAR perception pipeline with adaptive spatial resolution.
Designed for SIH demonstration, DRDO technical review, and engineering evaluation.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from data_gen import generate_frame
from terrain_segmentation import segment_ground
from object_clustering import cluster_objects, classify_static_vs_dynamic
from variable_resolution_grid import build_variable_grid, memory_savings, TIERS

# ---------------------------------------------------------
# Page Configuration & Technical Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Autonomous Perception | Variable-Resolution LiDAR Grid",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Technical CSS for Aerospace / Defense Console Appearance
st.markdown(
    """
    <style>
    /* Global Background & Font */
    .stApp {
        background-color: #070a12;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Header Container */
    .telemetry-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 10px;
        margin-bottom: 8px;
    }
    .telemetry-title {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        color: #f8fafc;
        text-transform: uppercase;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    .telemetry-subtitle {
        font-size: 0.95rem;
        font-weight: 600;
        color: #38bdf8;
        letter-spacing: 0.02em;
    }
    .telemetry-tagline {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 2px;
    }
    .status-badge-online {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid #10b981;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        color: #34d399;
        letter-spacing: 0.08em;
        font-family: ui-monospace, monospace;
    }
    .status-dot-online {
        width: 7px;
        height: 7px;
        background-color: #10b981;
        border-radius: 50%;
        margin-right: 6px;
        box-shadow: 0 0 8px #10b981;
    }
    .status-badge-error {
        display: inline-flex;
        align-items: center;
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        color: #f87171;
        letter-spacing: 0.08em;
        font-family: ui-monospace, monospace;
    }
    .pipeline-breadcrumb {
        font-size: 0.7rem;
        color: #475569;
        font-family: ui-monospace, monospace;
        letter-spacing: 0.04em;
        margin-bottom: 12px;
    }
    
    /* Primary Telemetry Cards */
    .metric-card {
        background: #0c1222;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .metric-card.highlight {
        background: #091a18;
        border: 1px solid #059669;
    }
    .metric-label {
        font-size: 0.68rem;
        font-weight: 700;
        color: #94a3b8;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-family: ui-monospace, monospace;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #f8fafc;
        font-family: ui-monospace, monospace;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .metric-value.green {
        color: #10b981;
    }
    .metric-value.cyan {
        color: #38bdf8;
    }
    
    /* Secondary Diagnostics Strip */
    .diag-strip {
        display: flex;
        gap: 12px;
        background: #080d1a;
        border: 1px solid #162032;
        border-radius: 4px;
        padding: 6px 12px;
        margin-bottom: 14px;
        font-family: ui-monospace, monospace;
        font-size: 0.72rem;
        color: #64748b;
    }
    .diag-item {
        display: flex;
        gap: 6px;
    }
    .diag-item span.val {
        color: #cbd5e1;
        font-weight: 600;
    }
    
    /* Custom Legend Bar */
    .legend-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        background: #090e1a;
        border: 1px solid #1e293b;
        border-radius: 4px;
        padding: 6px 12px;
        margin-top: 6px;
        margin-bottom: 12px;
        font-size: 0.74rem;
        font-family: ui-monospace, monospace;
        color: #94a3b8;
        align-items: center;
    }
    .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Collapsible & Technical Sections */
    .streamlit-expanderHeader {
        background-color: #0c1222 !important;
        border: 1px solid #1e293b !important;
        color: #cbd5e1 !important;
        font-family: ui-monospace, monospace !important;
        font-size: 0.8rem !important;
    }
    
    /* Sidebar Tightening */
    section[data-testid="stSidebar"] {
        background-color: #080c16;
        border-right: 1px solid #162032;
    }
    section[data-testid="stSidebar"] .stMarkdown h1, 
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #94a3b8;
        font-family: ui-monospace, monospace;
        font-size: 0.82rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "pedestrian_offset" not in st.session_state:
    st.session_state.pedestrian_offset = 1.2
if "frame_id" not in st.session_state:
    st.session_state.frame_id = 2
if "centroid_history" not in st.session_state:
    st.session_state.centroid_history = [(20.0, 5.0)]


# ---------------------------------------------------------
# Decoupled Perception Interface (Phase 16 Architecture)
# ---------------------------------------------------------
@dataclass
class PerceptionResult:
    """Standardized perception frame result.

    This abstraction decouples data ingestion (synthetic now, real LiDAR later)
    from downstream clustering, tracking, variable-grid mapping, and UI display.
    """
    frame_id: int
    points_raw: np.ndarray
    ground_mask: np.ndarray
    clusters: List[np.ndarray]
    cluster_labels: List[str]
    final_points: np.ndarray
    final_labels: np.ndarray
    grid: Dict[Tuple[float, int, int], Dict[str, Any]]
    uniform_cells: int
    adaptive_cells: int
    memory_saved_pct: float
    dynamic_centroid: Optional[Tuple[float, float]]
    dynamic_vector: Optional[Tuple[float, float]]
    execution_time: float


def run_perception_pipeline(
    pts_live: np.ndarray,
    reference_clusters: List[np.ndarray],
    ground_threshold: float,
    dbscan_eps: float,
    dbscan_min_points: int,
    frame_id: int,
    reference_dynamic_centroid: Optional[Tuple[float, float]] = None,
) -> PerceptionResult:
    """Execute the core perception and variable-resolution grid pipeline.

    Operates purely on 3D point arrays. Independent of synthetic or real data sources.
    """
    t_start = time.time()

    # 1. Ground plane segmentation (RANSAC)
    ground_mask = segment_ground(pts_live, distance_threshold=ground_threshold)

    # 2. Object clustering (DBSCAN) on non-ground points
    clusters = cluster_objects(
        pts_live,
        ~ground_mask,
        eps=dbscan_eps,
        min_points=dbscan_min_points,
    )

    # 3. Static vs Dynamic obstacle classification
    cluster_labels = classify_static_vs_dynamic(
        reference_clusters,
        clusters,
        move_threshold=0.3,
    )

    # 4. Assemble labeled point set from ground and classified clusters
    ground_pts = pts_live[ground_mask]
    ground_lbls = np.full(len(ground_pts), "ground", dtype=object)

    cluster_pts_list = []
    cluster_lbl_list = []
    dyn_centroid = None
    dyn_vector = None

    for cluster, label in zip(clusters, cluster_labels):
        cluster_pts_list.append(cluster)
        cluster_lbl_list.append(np.full(len(cluster), label, dtype=object))
        if label == "dynamic_object":
            c_xy = cluster[:, :2].mean(axis=0)
            dyn_centroid = (float(c_xy[0]), float(c_xy[1]))
            if reference_dynamic_centroid is not None:
                dyn_vector = (
                    dyn_centroid[0] - reference_dynamic_centroid[0],
                    dyn_centroid[1] - reference_dynamic_centroid[1],
                )

    if cluster_pts_list:
        final_points = np.vstack([ground_pts] + cluster_pts_list)
        final_labels = np.concatenate([ground_lbls] + cluster_lbl_list)
    else:
        final_points = ground_pts
        final_labels = ground_lbls

    # 5. Build variable-resolution grid from full live point cloud
    grid = build_variable_grid(final_points, final_labels)

    # 6. Compute memory savings
    uniform_cells, adaptive_cells, memory_saved_pct = memory_savings(
        grid, radius=100.0, finest_cell=0.05
    )

    elapsed = time.time() - t_start

    return PerceptionResult(
        frame_id=frame_id,
        points_raw=pts_live,
        ground_mask=ground_mask,
        clusters=clusters,
        cluster_labels=cluster_labels,
        final_points=final_points,
        final_labels=final_labels,
        grid=grid,
        uniform_cells=uniform_cells,
        adaptive_cells=adaptive_cells,
        memory_saved_pct=memory_saved_pct,
        dynamic_centroid=dyn_centroid,
        dynamic_vector=dyn_vector,
        execution_time=elapsed,
    )


# ---------------------------------------------------------
# Cached Reference Frame Processing (Performance Strategy)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_reference_frame(
    seed: int, distance_threshold: float, eps: float, min_points: int
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], Optional[Tuple[float, float]]]:
    """Cache static reference frame 1 to avoid redundant RANSAC/DBSCAN recomputation."""
    pts1, _ = generate_frame(num_ground=8000, seed=seed, pedestrian_offset=0.0)
    g_mask1 = segment_ground(pts1, distance_threshold=distance_threshold)
    clusters1 = cluster_objects(pts1, ~g_mask1, eps=eps, min_points=min_points)

    dyn_c1 = None
    # Identify initial dynamic object near (20, 5)
    for c in clusters1:
        c_mean = c[:, :2].mean(axis=0)
        if np.hypot(c_mean[0] - 20.0, c_mean[1] - 5.0) < 3.0:
            dyn_c1 = (float(c_mean[0]), float(c_mean[1]))
            break

    return pts1, g_mask1, clusters1, dyn_c1


# ---------------------------------------------------------
# Sidebar: Perception Controls
# ---------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="font-family: ui-monospace, monospace; font-size: 0.95rem; font-weight: 800; color: #f8fafc; letter-spacing: 0.08em; padding: 4px 0 12px 0; border-bottom: 1px solid #1e293b; margin-bottom: 12px;">
      PERCEPTION CONTROL
    </div>
    """,
    unsafe_allow_html=True,
)

# Collapsible Section 1: SIMULATION
with st.sidebar.expander("SIMULATION", expanded=True):
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("Step Frame ⏭️", use_container_width=True):
            st.session_state.pedestrian_offset += 1.0
            st.session_state.frame_id += 1
    with col_s2:
        if st.button("Reset 🔄", use_container_width=True):
            st.session_state.pedestrian_offset = 1.2
            st.session_state.frame_id = 2
            st.session_state.centroid_history = [(20.0, 5.0)]

    pedestrian_offset = st.slider(
        "Pedestrian X Offset (m)",
        min_value=0.0,
        max_value=15.0,
        value=float(st.session_state.pedestrian_offset),
        step=0.2,
        help="Simulates dynamic pedestrian motion along the X axis.",
    )
    st.session_state.pedestrian_offset = pedestrian_offset

# Collapsible Section 2: PERCEPTION
with st.sidebar.expander("PERCEPTION", expanded=False):
    ground_thresh = st.slider(
        "Ground RANSAC Threshold (m)",
        min_value=0.02,
        max_value=0.20,
        value=0.08,
        step=0.01,
        help="Max orthogonal distance to fitted ground plane.",
    )
    dbscan_eps = st.slider(
        "DBSCAN Epsilon (m)",
        min_value=0.20,
        max_value=1.20,
        value=0.60,
        step=0.05,
        help="Spatial clustering radius.",
    )
    dbscan_min_pts = st.slider(
        "DBSCAN Min Points",
        min_value=5,
        max_value=30,
        value=15,
        step=1,
        help="Minimum core points per cluster.",
    )

# Collapsible Section 3: VISUALIZATION
with st.sidebar.expander("VISUALIZATION", expanded=True):
    show_ground = st.checkbox("Ground Plane", value=True)
    show_static = st.checkbox("Static Obstacles", value=True)
    show_dynamic = st.checkbox("Dynamic Objects", value=True)
    show_rings = st.checkbox("Resolution Zones", value=True)
    show_trail = st.checkbox("Motion Trail", value=True)

# Collapsible Section 4: ADVANCED / DEBUG
with st.sidebar.expander("ADVANCED / DEBUG", expanded=False):
    ground_downsample = st.checkbox(
        "Fast Ground Render (Visual Only)",
        value=True,
        help="Decoupled visual optimization. Renders a representative subset of ground points to preserve high UI FPS without altering algorithmic calculations.",
    )
    st.caption("Point cloud pipeline processes 100% of data.")

# ---------------------------------------------------------
# Execution & Telemetry Pipeline
# ---------------------------------------------------------
system_error: Optional[str] = None
result: Optional[PerceptionResult] = None

try:
    # 1. Fetch cached reference frame (Frame 1)
    pts1, ground_mask1, clusters_frame1, dyn_ref_centroid = get_cached_reference_frame(
        seed=1,
        distance_threshold=ground_thresh,
        eps=dbscan_eps,
        min_points=dbscan_min_pts,
    )

    # 2. Ingest live point cloud (Frame 2)
    # [Future note: Real LiDAR reader replaces generate_frame here]
    pts2, _ = generate_frame(
        num_ground=8000,
        seed=1,
        pedestrian_offset=pedestrian_offset,
    )

    # 3. Execute perception pipeline
    result = run_perception_pipeline(
        pts_live=pts2,
        reference_clusters=clusters_frame1,
        ground_threshold=ground_thresh,
        dbscan_eps=dbscan_eps,
        dbscan_min_points=dbscan_min_pts,
        frame_id=st.session_state.frame_id,
        reference_dynamic_centroid=dyn_ref_centroid,
    )

    # Update centroid trail
    if result.dynamic_centroid is not None:
        curr_pt = result.dynamic_centroid
        if not st.session_state.centroid_history or np.hypot(
            curr_pt[0] - st.session_state.centroid_history[-1][0],
            curr_pt[1] - st.session_state.centroid_history[-1][1],
        ) > 0.05:
            st.session_state.centroid_history.append(curr_pt)
            if len(st.session_state.centroid_history) > 12:
                st.session_state.centroid_history.pop(0)

except Exception as ex:
    system_error = str(ex)

# ---------------------------------------------------------
# Header & Status Indicator
# ---------------------------------------------------------
status_html = (
    """
    <div class="status-badge-online">
      <div class="status-dot-online"></div>
      SYSTEM ONLINE
    </div>
    """
    if system_error is None
    else f"""
    <div class="status-badge-error">
      ● SYSTEM ERROR
    </div>
    """
)

st.markdown(
    f"""
    <div class="telemetry-header">
      <div>
        <div class="telemetry-title">AUTONOMOUS PERCEPTION</div>
        <div class="telemetry-subtitle">Variable-Resolution LiDAR Grid</div>
        <div class="telemetry-tagline">Real-time spatial perception with adaptive resolution</div>
      </div>
      <div>
        {status_html}
      </div>
    </div>
    <div class="pipeline-breadcrumb">
      LiDAR → Ground Segmentation → Object Clustering → Tracking → Adaptive Grid
    </div>
    """,
    unsafe_allow_html=True,
)

if system_error:
    st.error(f"Perception Pipeline Error: {system_error}")
    st.stop()

assert result is not None

# Compute real measured FPS
fps = 1.0 / result.execution_time if result.execution_time > 0 else 0.0

# Count cluster breakdown
num_static_clusters = sum(1 for lbl in result.cluster_labels if lbl == "static_obstacle")
num_dynamic_clusters = sum(1 for lbl in result.cluster_labels if lbl == "dynamic_object")
ground_points_count = int(np.sum(result.ground_mask))

# ---------------------------------------------------------
# Telemetry Metrics Area (Hierarchy: Primary & Secondary)
# ---------------------------------------------------------
p1, p2, p3, p4 = st.columns(4)

with p1:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">PIPELINE FPS</div>
          <div class="metric-value cyan">{fps:.1f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p2:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">ADAPTIVE CELLS</div>
          <div class="metric-value">{result.adaptive_cells:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p3:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">UNIFORM CELLS (0.05m)</div>
          <div class="metric-value">{result.uniform_cells:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p4:
    st.markdown(
        f"""
        <div class="metric-card highlight">
          <div class="metric-label">MEMORY SAVED</div>
          <div class="metric-value green">{result.memory_saved_pct:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Secondary Diagnostics Strip
st.markdown(
    f"""
    <div class="diag-strip">
      <div class="diag-item">TOTAL POINTS: <span class="val">{len(result.final_points):,}</span></div>
      <div>•</div>
      <div class="diag-item">GROUND POINTS: <span class="val">{ground_points_count:,}</span></div>
      <div>•</div>
      <div class="diag-item">STATIC CLUSTERS: <span class="val">{num_static_clusters}</span></div>
      <div>•</div>
      <div class="diag-item">DYNAMIC CLUSTERS: <span class="val">{num_dynamic_clusters}</span></div>
      <div>•</div>
      <div class="diag-item">EVALUATION RANGE: <span class="val">200m × 200m</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Main Visualization: LIVE LiDAR PERCEPTION
# ---------------------------------------------------------
vis_col_head, vis_col_badge = st.columns([3, 1])
with vis_col_head:
    st.markdown(
        """
        <div style="font-family: ui-monospace, monospace; font-size: 1.05rem; font-weight: 700; color: #f8fafc; letter-spacing: 0.04em;">
          LIVE LiDAR PERCEPTION
        </div>
        <div style="font-size: 0.74rem; color: #94a3b8;">
          Top-down spatial view • Adaptive resolution by range
        </div>
        """,
        unsafe_allow_html=True,
    )
with vis_col_badge:
    st.markdown(
        f"""
        <div style="text-align: right; font-family: ui-monospace, monospace; font-size: 0.75rem; color: #94a3b8; padding-top: 4px;">
          FRAME <span style="color: #38bdf8; font-weight: 700;">#{st.session_state.frame_id:02d}</span> • OFFSET <span style="color: #f59e0b; font-weight: 700;">+{pedestrian_offset:.1f}m</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Custom Legend Bar
st.markdown(
    """
    <div class="legend-bar">
      <div class="legend-item"><span style="color: #10b981; font-size: 0.9rem;">●</span> Ground</div>
      <div class="legend-item"><span style="color: #ef4444; font-size: 0.9rem;">●</span> Static Obstacle</div>
      <div class="legend-item"><span style="color: #f59e0b; font-size: 0.9rem;">●</span> Dynamic Object</div>
      <div class="legend-item"><span style="color: #38bdf8; font-size: 0.9rem;">◆</span> LiDAR Sensor (0,0)</div>
      <div class="legend-item"><span style="color: #f59e0b; font-weight: 700;">╌╌</span> Motion Trail</div>
      <div class="legend-item"><span style="color: rgba(56, 189, 248, 0.7);">◯</span> Adaptive Zone Boundaries (10m, 50m, 100m)</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Plotly Point-Cloud & Adaptive Grid Map
categories: Dict[str, Dict[str, Any]] = {
    "ground": {
        "x": [],
        "y": [],
        "hover": [],
        "show": show_ground,
    },
    "static_obstacle": {
        "x": [],
        "y": [],
        "hover": [],
        "show": show_static,
    },
    "dynamic_object": {
        "x": [],
        "y": [],
        "hover": [],
        "show": show_dynamic,
    },
}

# Rendering separation: Downsample only ground display for ultra-smooth UI
sample_counter = 0
for (cell_size, cell_x, cell_y), data in result.grid.items():
    label = data["label"]
    if label not in categories:
        label = "ground"

    # Fast Ground Render (visual layer only)
    if label == "ground" and ground_downsample:
        sample_counter += 1
        if sample_counter % 2 != 0:
            continue

    real_x = (cell_x + 0.5) * cell_size
    real_y = (cell_y + 0.5) * cell_size
    cat = categories[label]

    cat["x"].append(real_x)
    cat["y"].append(real_y)

    cat["hover"].append(
        f"<b>CLASS:</b> {label.upper()}<br>"
        f"<b>POS:</b> ({real_x:.2f}m, {real_y:.2f}m)<br>"
        f"<b>HEIGHT:</b> {data['height']:.2f}m<br>"
        f"<b>CELL:</b> {cell_size * 100:.0f}cm ({cell_size:.2f}m)<br>"
        f"<b>POINTS:</b> {data['count']}"
    )

fig = go.Figure()

# 1. Ground points (subdued, small)
if categories["ground"]["show"] and categories["ground"]["x"]:
    fig.add_trace(
        go.Scatter(
            x=categories["ground"]["x"],
            y=categories["ground"]["y"],
            mode="markers",
            name="Ground",
            marker=dict(
                size=2.0,
                color="#10b981",
                opacity=0.35,
            ),
            text=categories["ground"]["hover"],
            hoverinfo="text",
            showlegend=False,
        )
    )

# 2. Static Obstacles (bright red, medium size)
if categories["static_obstacle"]["show"] and categories["static_obstacle"]["x"]:
    fig.add_trace(
        go.Scatter(
            x=categories["static_obstacle"]["x"],
            y=categories["static_obstacle"]["y"],
            mode="markers",
            name="Static Obstacle",
            marker=dict(
                size=7.5,
                color="#ef4444",
                opacity=0.95,
                line=dict(width=1.0, color="#ffffff"),
            ),
            text=categories["static_obstacle"]["hover"],
            hoverinfo="text",
            showlegend=False,
        )
    )

# 3. Dynamic Objects (bright amber, large, prominent)
if categories["dynamic_object"]["show"] and categories["dynamic_object"]["x"]:
    fig.add_trace(
        go.Scatter(
            x=categories["dynamic_object"]["x"],
            y=categories["dynamic_object"]["y"],
            mode="markers",
            name="Dynamic Object",
            marker=dict(
                size=11.5,
                color="#f59e0b",
                opacity=1.0,
                line=dict(width=2.0, color="#ffffff"),
            ),
            text=categories["dynamic_object"]["hover"],
            hoverinfo="text",
            showlegend=False,
        )
    )

# 4. Motion Trail & Heading Vector
if show_trail and len(st.session_state.centroid_history) > 1 and show_dynamic:
    t_x = [pt[0] for pt in st.session_state.centroid_history]
    t_y = [pt[1] for pt in st.session_state.centroid_history]
    fig.add_trace(
        go.Scatter(
            x=t_x,
            y=t_y,
            mode="lines+markers",
            name="Motion Trail",
            line=dict(color="#f59e0b", width=2.0, dash="dash"),
            marker=dict(size=4.5, color="#fbbf24"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

# Add "DYNAMIC" Callout Annotation near the dynamic object
if result.dynamic_centroid is not None and show_dynamic:
    dyn_x, dyn_y = result.dynamic_centroid
    fig.add_annotation(
        x=dyn_x,
        y=dyn_y,
        text="<b>DYNAMIC</b>",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowcolor="#f59e0b",
        ax=35,
        ay=-25,
        font=dict(color="#f59e0b", size=10, family="monospace"),
        bgcolor="rgba(15, 23, 42, 0.9)",
        bordercolor="#f59e0b",
        borderwidth=1,
    )

# 5. Sensor Origin Marker & Heading Indicator
fig.add_trace(
    go.Scatter(
        x=[0.0],
        y=[0.0],
        mode="markers",
        name="Sensor (0,0)",
        marker=dict(size=13, color="#38bdf8", symbol="diamond", line=dict(width=1.5, color="#ffffff")),
        text=["<b>LiDAR SENSOR (ORIGIN)</b><br>Coordinates: (0.00, 0.00)<br>Elevation: 0.0m"],
        hoverinfo="text",
        showlegend=False,
    )
)

# Subtle forward heading indicator ray (+X direction)
fig.add_trace(
    go.Scatter(
        x=[0.0, 14.0],
        y=[0.0, 0.0],
        mode="lines",
        name="Sensor Heading",
        line=dict(color="rgba(56, 189, 248, 0.5)", width=1.5, dash="dot"),
        hoverinfo="skip",
        showlegend=False,
    )
)

# Sensor Origin Callout Label (positioned carefully away from points)
fig.add_annotation(
    x=0,
    y=0,
    text="<b>LiDAR SENSOR</b>",
    showarrow=True,
    arrowhead=1,
    arrowsize=0.8,
    arrowcolor="#38bdf8",
    ax=-45,
    ay=-25,
    font=dict(color="#38bdf8", size=10, family="monospace"),
    bgcolor="rgba(15, 23, 42, 0.9)",
    bordercolor="#38bdf8",
    borderwidth=1,
)

# 6. Variable-Resolution Range Rings & Edge Annotations
if show_rings:
    # 10m Ring (Tier 1: 5 cm)
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=-10,
        y0=-10,
        x1=10,
        y1=10,
        line=dict(color="rgba(56, 189, 248, 0.45)", width=1.2, dash="dash"),
    )
    # 50m Ring (Tier 2: 20 cm)
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=-50,
        y0=-50,
        x1=50,
        y1=50,
        line=dict(color="rgba(56, 189, 248, 0.35)", width=1.2, dash="dash"),
    )
    # 100m Ring (Tier 3: 50 cm)
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=-100,
        y0=-100,
        x1=100,
        y1=100,
        line=dict(color="rgba(56, 189, 248, 0.25)", width=1.2, dash="dash"),
    )

    # Edge Annotations near the right side of the rings (+X boundary)
    fig.add_annotation(
        x=10.0,
        y=0.0,
        text="<b>5 CM</b><br><span style='font-size:8px'>0–10m</span>",
        showarrow=False,
        xanchor="left",
        xshift=4,
        font=dict(color="#38bdf8", size=9, family="monospace"),
        bgcolor="rgba(15, 23, 42, 0.85)",
        bordercolor="rgba(56, 189, 248, 0.5)",
        borderwidth=1,
    )
    fig.add_annotation(
        x=50.0,
        y=0.0,
        text="<b>20 CM</b><br><span style='font-size:8px'>10–50m</span>",
        showarrow=False,
        xanchor="left",
        xshift=4,
        font=dict(color="#38bdf8", size=9, family="monospace"),
        bgcolor="rgba(15, 23, 42, 0.85)",
        bordercolor="rgba(56, 189, 248, 0.4)",
        borderwidth=1,
    )
    fig.add_annotation(
        x=100.0,
        y=0.0,
        text="<b>50 CM</b><br><span style='font-size:8px'>50–100m</span>",
        showarrow=False,
        xanchor="left",
        xshift=4,
        font=dict(color="#38bdf8", size=9, family="monospace"),
        bgcolor="rgba(15, 23, 42, 0.85)",
        bordercolor="rgba(56, 189, 248, 0.3)",
        borderwidth=1,
    )

# Equal aspect ratio layout
fig.update_layout(
    xaxis=dict(
        title="X (m)",
        range=[-105, 105],
        zeroline=True,
        zerolinecolor="#1e293b",
        gridcolor="#0f172a",
        title_font=dict(family="monospace", size=11, color="#64748b"),
        tickfont=dict(family="monospace", size=9, color="#64748b"),
    ),
    yaxis=dict(
        title="Y (m)",
        range=[-105, 105],
        scaleanchor="x",
        scaleratio=1,
        zeroline=True,
        zerolinecolor="#1e293b",
        gridcolor="#0f172a",
        title_font=dict(family="monospace", size=11, color="#64748b"),
        tickfont=dict(family="monospace", size=9, color="#64748b"),
    ),
    height=690,
    plot_bgcolor="#060912",
    paper_bgcolor="#060912",
    margin=dict(l=20, r=20, t=20, b=20),
    hoverlabel=dict(
        bgcolor="#0c1222",
        bordercolor="#38bdf8",
        font=dict(family="monospace", size=10, color="#f8fafc"),
    ),
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------
# Information Panel: ADAPTIVE GRID EFFICIENCY
# ---------------------------------------------------------
st.markdown(
    """
    <div style="font-family: ui-monospace, monospace; font-size: 0.95rem; font-weight: 700; color: #f8fafc; letter-spacing: 0.04em; margin-top: 14px; margin-bottom: 8px;">
      ADAPTIVE GRID EFFICIENCY
    </div>
    """,
    unsafe_allow_html=True,
)

eff_col1, eff_col2 = st.columns([1, 1])

with eff_col1:
    st.markdown(
        f"""
        <div style="background: #080d1a; border: 1px solid #162032; border-radius: 6px; padding: 14px 16px; font-family: ui-monospace, monospace; font-size: 0.8rem; line-height: 1.6;">
          <div style="color: #94a3b8; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">Spatial Memory Reduction Summary</div>
          <div>• <b>Uniform 5 cm Grid:</b> <span style="color: #ef4444;">16,000,000 cells</span> ((200m / 0.05m)²)</div>
          <div>• <b>Adaptive Grid:</b> <span style="color: #38bdf8;">{result.adaptive_cells:,} cells</span> (Multi-tier allocation)</div>
          <div>• <b>Cell Reduction:</b> <span style="color: #10b981; font-weight: 700;">{result.memory_saved_pct:.4f}%</span></div>
          <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #1e293b; color: #64748b; font-size: 0.74rem;">
            Tier 1 (0–10m): <b>5cm</b> &nbsp;|&nbsp; Tier 2 (10–50m): <b>20cm</b> &nbsp;|&nbsp; Tier 3 (50–100m): <b>50cm</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with eff_col2:
    # Custom HTML progress bars accurately depicting relative memory footprint
    st.markdown(
        f"""
        <div style="background: #080d1a; border: 1px solid #162032; border-radius: 6px; padding: 14px 16px; font-family: ui-monospace, monospace; font-size: 0.78rem;">
          <div style="color: #94a3b8; font-weight: 700; text-transform: uppercase; margin-bottom: 8px;">Memory Footprint Comparison</div>
          
          <div style="margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
              <span style="color: #e2e8f0;">Uniform 5cm Grid</span>
              <span style="color: #ef4444;">16,000,000 cells (100.0%)</span>
            </div>
            <div style="background: #1e293b; border-radius: 3px; height: 12px; width: 100%;">
              <div style="background: #ef4444; width: 100%; height: 12px; border-radius: 3px;"></div>
            </div>
          </div>

          <div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
              <span style="color: #e2e8f0;">Adaptive Variable Grid</span>
              <span style="color: #10b981;">{result.adaptive_cells:,} cells ({(result.adaptive_cells / result.uniform_cells) * 100:.3f}%)</span>
            </div>
            <div style="background: #1e293b; border-radius: 3px; height: 12px; width: 100%;">
              <div style="background: #10b981; width: {max(0.8, (result.adaptive_cells / result.uniform_cells) * 100):.2f}%; height: 12px; border-radius: 3px;"></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Pipeline Architecture (Collapsible)
# ---------------------------------------------------------
with st.expander("PIPELINE", expanded=False):
    st.markdown(
        """
        ```
        LiDAR Point Cloud (Raw (x,y,z) coordinate array)
                        │
                        ▼
        Ground Segmentation (RANSAC Plane Fitting • terrain_segmentation.py)
                        │
                        ▼
        Object Clustering (Open3D DBSCAN • object_clustering.py)
                        │
                        ▼
        Static / Dynamic Classification (Inter-frame Centroid Displacement)
                        │
                        ▼
        Tracking (Dynamic Object Motion Trail & Heading Vector)
                        │
                        ▼
        Variable-Resolution Grid (Multi-Tier Cell Allocation • variable_resolution_grid.py)
        ```
        """
    )

# ---------------------------------------------------------
# System Status & Technical Context Note
# ---------------------------------------------------------
st.markdown(
    """
    <div style="margin-top: 14px; padding: 10px 14px; background: #080c16; border: 1px solid #162032; border-radius: 4px; font-family: ui-monospace, monospace; font-size: 0.72rem; color: #64748b;">
      <b style="color: #94a3b8;">SYSTEM NOTE:</b> Ground and object segmentation utilizes classical geometry (RANSAC and DBSCAN) as a deterministic stand-in for deep-learning backbones, while the multi-tier variable-resolution spatial grid is the full production logic ready for real LiDAR integration.
    </div>
    """,
    unsafe_allow_html=True,
)
