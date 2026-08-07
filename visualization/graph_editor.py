"""Mouse-based graph editor for the Streamlit application.

Interaction:
    * Click an empty location to add a node.
    * Click two existing nodes to add an undirected edge.

The editor stores nodes as a list of string labels and edges as endpoint
tuples, matching the representation used by app.py.
"""

from __future__ import annotations

import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas


CANVAS_WIDTH = 760
CANVAS_HEIGHT = 450
NODE_RADIUS = 24


def initialize_graph_state() -> None:
    """Create the graph-related session state used by the editor."""
    if "nodes" not in st.session_state:
        st.session_state.nodes = ["0", "1", "2", "3"]

    if "node_positions" not in st.session_state:
        st.session_state.node_positions = {
            "0": (100, 225),
            "1": (280, 225),
            "2": (460, 225),
            "3": (640, 225),
        }

    # Make the editor robust to a session created by the previous app version.
    # Such a session may already contain nodes but no saved positions.
    for index, node_id in enumerate(st.session_state.nodes):
        if node_id not in st.session_state.node_positions:
            x = 100 + (index % 4) * 180
            y = 100 + (index // 4) * 100
            st.session_state.node_positions[node_id] = (x, y)

    if "edges" not in st.session_state:
        st.session_state.edges = [("0", "1"), ("1", "2"), ("2", "3")]

    if "edge_selection" not in st.session_state:
        st.session_state.edge_selection = []

    if "canvas_revision" not in st.session_state:
        st.session_state.canvas_revision = 0


def _next_node_label() -> str:
    """Return the first unused nonnegative integer label."""
    integer_labels = {
        int(label)
        for label in st.session_state.nodes
        if str(label).isdigit()
    }
    candidate = 0
    while candidate in integer_labels:
        candidate += 1
    return str(candidate)


def _node_at(x: float, y: float) -> str | None:
    """Return the node at a canvas location, if one is nearby."""
    for node_id, (node_x, node_y) in st.session_state.node_positions.items():
        if (x - node_x) ** 2 + (y - node_y) ** 2 <= NODE_RADIUS**2:
            return node_id
    return None


def _edge_exists(u: str, v: str) -> bool:
    return (u, v) in st.session_state.edges or (v, u) in st.session_state.edges


def _handle_click(x: float, y: float) -> None:
    """Add a node or use the click to create an edge."""
    clicked_node = _node_at(x, y)

    if clicked_node is None:
        node_id = _next_node_label()
        st.session_state.nodes.append(node_id)
        st.session_state.node_positions[node_id] = (round(x), round(y))
        st.session_state.edge_selection = []
        st.session_state.editor_message = f"Added node {node_id}."
        return

    selected = st.session_state.edge_selection
    if not selected:
        st.session_state.edge_selection = [clicked_node]
        st.session_state.editor_message = (
            f"Selected node {clicked_node}; click another node to add an edge."
        )
        return

    first_node = selected[0]
    st.session_state.edge_selection = []

    if first_node == clicked_node:
        st.session_state.editor_message = "Edge selection cancelled."
    elif _edge_exists(first_node, clicked_node):
        st.session_state.editor_message = "That edge already exists."
    else:
        st.session_state.edges.append((first_node, clicked_node))
        st.session_state.editor_message = (
            f"Added edge {first_node} — {clicked_node}."
        )


def _background_image(edge_probability: float) -> Image.Image:
    """Draw the current graph as the canvas background."""
    image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    for u, v in st.session_state.edges:
        if u not in st.session_state.node_positions or v not in st.session_state.node_positions:
            continue
        draw.line(
            [st.session_state.node_positions[u], st.session_state.node_positions[v]],
            fill="#777777",
            width=2,
        )

    source = st.session_state.get("source")
    sink = st.session_state.get("sink")
    selected = st.session_state.edge_selection[0] if st.session_state.edge_selection else None

    for node_id in st.session_state.nodes:
        x, y = st.session_state.node_positions[node_id]
        if node_id == source:
            color = "#2ca02c"
        elif node_id == sink:
            color = "#d62728"
        else:
            color = "#4682b4"

        if node_id == selected:
            color = "#f0ad00"

        draw.ellipse(
            (x - NODE_RADIUS, y - NODE_RADIUS, x + NODE_RADIUS, y + NODE_RADIUS),
            fill=color,
            outline="#333333",
            width=2,
        )
        bbox = draw.textbbox((0, 0), str(node_id))
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text(
            (x - text_width / 2, y - text_height / 2 - bbox[1]),
            str(node_id),
            fill="white",
        )

    return image


def render_graph_editor(edge_probability: float) -> None:
    """Render the editor and process new mouse clicks."""
    initialize_graph_state()

    st.caption(
        "Click empty space to add a node. Click two existing nodes to add an edge. "
        "Source is green and target is red."
    )

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=1,
        stroke_color="rgba(0, 0, 0, 0)",
        background_image=_background_image(edge_probability),
        update_streamlit=True,
        drawing_mode="point",
        point_display_radius=1,
        width=CANVAS_WIDTH,
        height=CANVAS_HEIGHT,
        key=f"graph_canvas_{st.session_state.canvas_revision}",
    )

    objects = []
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])

    if objects:
        latest = objects[-1]
        _handle_click(latest.get("left", 0), latest.get("top", 0))
        st.session_state.canvas_revision += 1
        st.rerun()

    if st.session_state.get("editor_message"):
        st.info(st.session_state.editor_message)


def remove_node(node_id: str) -> None:
    """Remove a node and all incident edges."""
    st.session_state.nodes.remove(node_id)
    st.session_state.node_positions.pop(node_id, None)
    st.session_state.edges = [
        (u, v)
        for u, v in st.session_state.edges
        if u != node_id and v != node_id
    ]
    st.session_state.edge_selection = []
