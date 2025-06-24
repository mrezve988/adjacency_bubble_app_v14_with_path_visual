# ✅ Full app combining working layout with scoring + suggestions
import streamlit as st
import pandas as pd
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os
import itertools
import plotly.graph_objects as go
import networkx as nx

st.set_page_config(page_title="Adjacency-Based Room Planner", layout="wide")
st.title("🏠 Adjacency-Based Room Planner")

room_data = {
    "Living Room": {"length": 15, "width": 12, "privacy": "Public"},
    "Kitchen": {"length": 10, "width": 10, "privacy": "Service"},
    "Dining": {"length": 12, "width": 10, "privacy": "Public"},
    "Bedroom 1": {"length": 12, "width": 12, "privacy": "Private"},
    "Bedroom 2": {"length": 10, "width": 10, "privacy": "Private"},
    "Bedroom 3": {"length": 10, "width": 10, "privacy": "Private"},
    "Toilet 1": {"length": 6, "width": 5, "privacy": "Service"},
    "Bath": {"length": 8, "width": 6, "privacy": "Private"},
    "Store": {"length": 8, "width": 6, "privacy": "Service"}
}
standard_adjacencies = [
    ("Living Room", "Dining"), ("Dining", "Kitchen"), ("Kitchen", "Store"),
    ("Dining", "Bedroom 1"), ("Bedroom 1", "Bath"), ("Dining", "Bedroom 2"),
    ("Dining", "Bedroom 3"), ("Living Room", "Toilet 1")
]
room_list = list(room_data.keys())

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🧭 Standard Room Info + Adjacency")
    adj_map = {room: [] for room in room_list}
    for a, b in standard_adjacencies:
        adj_map[a].append(b)
        adj_map[b].append(a)
    standard_df = pd.DataFrame([
        {"Room": room,
         "Area (L x W = A)": f"{d['length']}' x {d['width']}' = {d['length']*d['width']} sqft",
         "Privacy": d["privacy"],
         "Adjacent To": ", ".join(adj_map[room])} for room, d in room_data.items()
    ])
    st.dataframe(standard_df, use_container_width=True)

with col2:
    st.markdown("### ✍️ User Input")
    with st.expander("📐 Room Sizes and Privacy"):
        user_inputs = []
        for room in room_list:
            c1, c2, c3 = st.columns([1, 1, 2])
            length = c1.number_input(f"{room} Length", 1, 50, room_data[room]["length"], key=f"{room}_l")
            width = c2.number_input(f"{room} Width", 1, 50, room_data[room]["width"], key=f"{room}_w")
            privacy = c3.selectbox(f"{room} Privacy", ["Public", "Private", "Service"],
                                   index=["Public", "Private", "Service"].index(room_data[room]["privacy"]), key=f"{room}_p")
            user_inputs.append({
                "Room": room,
                "Area (L x W = A)": f"{length}' x {width}' = {length * width} sqft",
                "Privacy": privacy
            })
        user_df = pd.DataFrame(user_inputs)

    with st.expander("✅ Define Adjacencies"):
        user_adjacencies = []
        for a, b in itertools.combinations(room_list, 2):
            if st.checkbox(f"{a} adjacent to {b}", key=f"{a}_{b}"):
                user_adjacencies.append((a, b))

privacy_colors = {"Public": "green", "Private": "blue", "Service": "orange"}

def draw_static(df, edges, title):
    G = nx.Graph()
    sizes, colors = [], []
    for _, r in df.iterrows():
        name = r["Room"]
        dims = r["Area (L x W = A)"].split("=")[0].split("x")
        area = int(dims[0].strip().replace("'", "")) * int(dims[1].strip().replace("'", ""))
        sizes.append(area)
        colors.append(privacy_colors.get(r["Privacy"], "gray"))
        G.add_node(name)
    for a, b in edges:
        if a in G.nodes and b in G.nodes:
            G.add_edge(a, b)
    pos = nx.spring_layout(G, seed=42)
    x, y = [], []
    for edge in G.edges():
        x += [pos[edge[0]][0], pos[edge[1]][0], None]
        y += [pos[edge[0]][1], pos[edge[1]][1], None]
    node_x, node_y = zip(*[pos[n] for n in G.nodes()])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(width=1)))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', marker=dict(size=[a/3 for a in sizes], color=colors),
                             text=list(G.nodes()), textposition="bottom center"))
    fig.update_layout(title=title, showlegend=False, height=500, width=500)
    st.plotly_chart(fig)

def draw_interactive(df, edges):
    net = Network(height="500px", width="100%", bgcolor="#fff")
    for _, r in df.iterrows():
        room = r["Room"]
        dims = r["Area (L x W = A)"].split("=")[0].split("x")
        area = int(dims[0].strip().replace("'", "")) * int(dims[1].strip().replace("'", ""))
        net.add_node(room, label=room, size=area / 2, color=privacy_colors.get(r["Privacy"], "gray"))
    for a, b in edges:
        if a in df["Room"].values and b in df["Room"].values:
            net.add_edge(a, b)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp.name)
    components.html(open(tmp.name, 'r').read(), height=500)
    os.unlink(tmp.name)

# Visual comparison
st.markdown("### 📐 Plan Diagrams")
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Standard")
    draw_static(standard_df, standard_adjacencies, "Standard Bubble")
with c2:
    st.markdown("#### User")
    draw_static(user_df, user_adjacencies, "User Bubble")

c3, c4 = st.columns(2)
with c3:
    st.markdown("#### Standard Interactive")
    draw_interactive(standard_df, standard_adjacencies)
with c4:
    st.markdown("#### User Interactive")
    draw_interactive(user_df, user_adjacencies)

# ---------- Scoring ----------
std_set = set(tuple(sorted(p)) for p in standard_adjacencies)
usr_set = set(tuple(sorted(p)) for p in user_adjacencies)
intersection = std_set & usr_set
union = std_set | usr_set
jaccard_score = len(intersection) / len(union) if union else 1

size_devs = []
for room in room_list:
    std_area = room_data[room]['length'] * room_data[room]['width']
    user_dims = user_df[user_df["Room"] == room]["Area (L x W = A)"].values[0].split("=")[1].strip().split()[0]
    user_area = int(user_dims)
    size_devs.append(abs(std_area - user_area) / std_area)
size_score = 1 - (sum(size_devs) / len(size_devs))

privacy_mismatches = sum([
    room_data[room]['privacy'] != user_df[user_df["Room"] == room]["Privacy"].values[0]
    for room in room_list
])
privacy_score = 1 - (privacy_mismatches / len(room_list))
final_score = round((0.4 * jaccard_score + 0.4 * size_score + 0.2 * privacy_score) * 100, 2)


# ---------- Suggestions ----------
improvements = []

missing = std_set - usr_set
for a, b in sorted(missing):
    improvements.append(f"Connect **{a}** to **{b}** (missing adjacency).")

for room in room_list:
    std_priv = room_data[room]['privacy']
    usr_priv = user_df[user_df["Room"] == room]["Privacy"].values[0]
    if std_priv != usr_priv:
        improvements.append(f"Change **{room}** privacy from **{usr_priv}** to **{std_priv}**.")

for room in room_list:
    std_area = room_data[room]['length'] * room_data[room]['width']
    usr_area = int(user_df[user_df["Room"] == room]["Area (L x W = A)"].values[0].split("=")[1].split()[0])
    deviation = abs(std_area - usr_area) / std_area * 100
    if deviation > 20:
        improvements.append(f"Adjust **{room}** size (deviation is {deviation:.1f}%).")


# ---------- Circulation Distance Estimation ----------

import numpy as np
from sklearn.linear_model import LinearRegression

def estimate_circulation(df, edges):
    # Assign arbitrary layout positions (grid layout for simplification)
    positions = {}
    for i, room in enumerate(df["Room"]):
        x = i % 3
        y = i // 3
        positions[room] = (x * 10, y * 10)  # grid with 10ft spacing

    distances = []
    for a, b in edges:
        if a in positions and b in positions:
            xa, ya = positions[a]
            xb, yb = positions[b]
            dist = np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)
            distances.append(dist)

    total_dist = sum(distances)
    return round(total_dist, 2)

std_circ = estimate_circulation(standard_df, standard_adjacencies)
usr_circ = estimate_circulation(user_df, user_adjacencies)

st.markdown("### 🔎 Layout Evaluation Summary")

col1, col2, col3 = st.columns([1, 1.5, 1.5])

# --- Column 1: Scores ---
with col1:
    st.markdown("#### 📊 Scores")
    st.metric("Adjacency Match", f"{jaccard_score*100:.1f}%")
    st.metric("Size Accuracy", f"{size_score*100:.1f}%")
    st.metric("Privacy Match", f"{privacy_score*100:.1f}%")
    st.metric("Final Score", f"{final_score}%")

# --- Column 2: Suggestions ---
with col2:
    st.markdown("#### 💡 Suggestions")
    if improvements:
        for item in improvements:
            st.markdown(f"- {item}")
    else:
        st.success("✅ Your layout is well-matched!")

# --- Column 3: Circulation ---
with col3:
    st.markdown("#### 📏 Circulation")
    st.metric("Standard", f"{std_circ} ft")
    st.metric("User", f"{usr_circ} ft")
    if usr_circ > std_circ:
        st.warning("⚠️ Reduce user circulation.")
    else:
        st.success("✅ Circulation optimized.")

# ---------- Circulation Path Visualization ----------
st.markdown("### 🗺 Circulation Path Visualization")

def draw_force_circulation(df, edges, title):
    import networkx as nx
    import plotly.graph_objects as go
    import numpy as np

    G = nx.Graph()
    color_map = []
    size_map = []

    for _, row in df.iterrows():
        room = row["Room"]
        dims = row["Area (L x W = A)"].split("=")[0].split("x")
        length = int(dims[0].strip().replace("'", ""))
        width = int(dims[1].strip().replace("'", ""))
        area = length * width
        G.add_node(room)
        color_map.append(privacy_colors.get(row["Privacy"], "gray"))
        size_map.append(area)

    for a, b in edges:
        if a in G.nodes and b in G.nodes:
            G.add_edge(a, b)

    pos = nx.spring_layout(G, seed=42)  # logical spring layout
    edge_x, edge_y = [], []
    edge_labels = []
    for a, b in G.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        dist = round(np.sqrt((x0 - x1)**2 + (y0 - y1)**2) * 10, 1)
        edge_labels.append(((x0 + x1) / 2, (y0 + y1) / 2, dist))

    node_x, node_y = zip(*[pos[n] for n in G.nodes()])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='gray')))
    for x, y, label in edge_labels:
        fig.add_trace(go.Scatter(x=[x], y=[y], text=[f"{label} ft"], mode='text', textfont=dict(size=10)))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        marker=dict(size=[a / 3 for a in size_map], color=color_map, line=dict(width=2, color='DarkSlateGrey')),
        text=list(G.nodes()), textposition="bottom center"
    ))
    fig.update_layout(title=title, showlegend=False, height=550, width=550, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig)

col_fd1, col_fd2 = st.columns(2)
with col_fd1:
    st.markdown("#### 🧭 Standard Force-Directed")
    draw_force_circulation(standard_df, standard_adjacencies, "Standard Circulation")

with col_fd2:
    st.markdown("#### 🧑‍🎨 User Force-Directed")
    draw_force_circulation(user_df, user_adjacencies, "User Circulation")


# ---------- Interactive Sketch Pad ----------
from streamlit_drawable_canvas import st_canvas
import plotly.graph_objects as go
import math
import pandas as pd
import io
from PIL import Image

st.markdown("### 🧊 Sketch Your Plan")

col1, col2, col3 = st.columns([1, 2.5, 1.5])

with col1:
    st.markdown("#### 🎨 Drawing Controls")
    drawing_mode = st.radio("Tool", ["rect", "circle", "freedraw"], label_visibility="collapsed")
    stroke_width = st.slider("Stroke", 1, 5, 2)
    room_name = st.text_input("Name", value="Room")
    zoning = st.selectbox("Zoning", ["Public", "Private", "Service"], index=2)

    privacy_colors = {
        "Public": "#00cc44",
        "Private": "#3399ff",
        "Service": "#ff9900"
    }
    room_color = privacy_colors[zoning]

    st.markdown("#### 📏 Room Details")
    total_area = 0.0
    object_details = []
    total_cost = 0.0
cost_per_sqft = 162  # 2024 construction rate from NAHB

with col2:
    canvas_result = st_canvas(
        fill_color=room_color + "66",
        stroke_width=stroke_width,
        stroke_color=room_color,
        background_color="#FFFFFF",
        height=600,
        width=1000,
        drawing_mode=drawing_mode,
        key="sketch-canvas-final"
    )  # ✅ correctly closed here

with col3:
    st.markdown("#### 🧠 Reference Bubble Diagram")
    draw_interactive(user_df, user_adjacencies)


# ---------- Shape Metadata + Display ----------
if canvas_result.json_data and "objects" in canvas_result.json_data:
    current_shapes = canvas_result.json_data["objects"]
    current_count = len(current_shapes)

    # ✅ Initialize or sync shape metadata
    if "shape_meta" not in st.session_state:
        st.session_state.shape_meta = []

    if current_count > len(st.session_state.shape_meta):
        st.session_state.shape_meta.append({
            "name": room_name,
            "zoning": zoning
        })
    elif current_count < len(st.session_state.shape_meta):
        st.session_state.shape_meta = st.session_state.shape_meta[:current_count]

    for i, obj in enumerate(current_shapes):
        shape = obj.get("type")
        meta = st.session_state.shape_meta[i]
        name = meta["name"]
        zoning_type = meta["zoning"]
        color_icon = {"Public": "🟩", "Private": "🟦", "Service": "🟧"}.get(zoning_type, "⬜")

        if shape == "rect":
            width = obj.get("width", 0)
            height = obj.get("height", 0)
            area = width * height / 100
            cost = area * cost_per_sqft
            object_details.append(
    f"{color_icon} {name} ({zoning_type}) – Rect Area: {area:.2f} sqft – Est. Cost: ${cost:,.2f}"
)
            total_area += area
            total_cost += cost
        elif shape == "circle":
            radius = obj.get("radius", 0)
            area = math.pi * radius**2 / 100
            cost = area * cost_per_sqft
            object_details.append(
    f"{color_icon} {name} ({zoning_type}) – Rect Area: {area:.2f} sqft – Est. Cost: ${cost:,.2f}"
)
            total_area += area
            total_cost += cost
        elif shape == "path":
            object_details.append(f"{color_icon} {name} ({zoning_type}) - Freehand (area not calculated)")

    with col1:
        for detail in object_details:
            st.markdown(detail)
        st.markdown(f"#### 📐 Total Plan Area: **{total_area:.2f} sqft**")
        st.markdown(f"#### 💵 Estimated Construction Cost: **${total_cost:,.2f}**")

    # ---------- Plotly Preview with Download (Room Name + Area) ----------
    fig = go.Figure()
    fill_opacity = 0.4

    for i, obj in enumerate(current_shapes):
        shape = obj.get("type")
        meta = st.session_state.shape_meta[i]
        name = meta["name"]
        zoning_type = meta["zoning"]
        color = privacy_colors[zoning_type]
        x, y = obj.get("left", 0), obj.get("top", 0)

        if shape == "rect":
            w = obj.get("width", 0)
            h = obj.get("height", 0)
            area = w * h / 100
            label = f"{name}<br>{area:.2f} sqft"
            fig.add_shape(type="rect",
                          x0=x, y0=y, x1=x + w, y1=y + h,
                          line=dict(color=color), fillcolor=color, opacity=fill_opacity)
            fig.add_trace(go.Scatter(x=[x + w / 2], y=[y + h / 2],
                                     text=[label], mode="text",
                                     textposition="middle center",
                                     textfont=dict(size=14, color="black")))
        elif shape == "circle":
            r = obj.get("radius", 0)
            area = math.pi * r**2 / 100
            label = f"{name}<br>{area:.2f} sqft"
            fig.add_shape(type="circle",
                          x0=x - r, y0=y - r, x1=x + r, y1=y + r,
                          line=dict(color=color), fillcolor=color, opacity=fill_opacity)
            fig.add_trace(go.Scatter(x=[x], y=[y],
                                     text=[label], mode="text",
                                     textposition="middle center",
                                     textfont=dict(size=14, color="black")))

    fig.update_layout(
    title="🧾 Sketch with Room Labels + Area",
    showlegend=False,
    height=600,
    width=1000,
    xaxis=dict(
        visible=False,
        scaleanchor="y",  # 🔐 lock aspect ratio
        scaleratio=1
    ),
    yaxis=dict(
        visible=False,
        autorange='reversed'
    ),
    margin=dict(l=10, r=10, t=30, b=10),
    plot_bgcolor="white"
)

    # Convert Plotly figure to PNG
    img_bytes = fig.to_image(format="png", engine="kaleido")
    buffer = io.BytesIO(img_bytes)

    st.image(Image.open(buffer), caption="🖼️ Sketch Image Preview")

    st.download_button(
        label="📥 Download Sketch as PNG",
        data=buffer,
        file_name="room_sketch.png",
        mime="image/png"
    )

else:
    with col1:
        st.info("Draw shapes to display room details.")
