from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

from binary_sync.model import Graph
from visualization.computation import compute_all_correlations


st.set_page_config(page_title="Binary Synchronization", layout="wide")
st.title("Binary Synchronization")


# -----------------------------------------------------------------------------
# Graph state
# -----------------------------------------------------------------------------

if "nodes" not in st.session_state:
    st.session_state.nodes = ["0", "1", "2", "3"]

if "edges" not in st.session_state:
    st.session_state.edges = [("0", "1"), ("1", "2"), ("2", "3")]

if "source" not in st.session_state:
    st.session_state.source = st.session_state.nodes[0]

if "sink" not in st.session_state:
    st.session_state.sink = st.session_state.nodes[-1]


def repair_source_and_sink():
    """Keep source and sink valid after nodes are added or deleted."""
    if not st.session_state.nodes:
        st.session_state.source = None
        st.session_state.sink = None
        return

    if st.session_state.source not in st.session_state.nodes:
        st.session_state.source = st.session_state.nodes[0]

    if st.session_state.sink not in st.session_state.nodes:
        st.session_state.sink = st.session_state.nodes[-1]


repair_source_and_sink()


# -----------------------------------------------------------------------------
# Define graph
# -----------------------------------------------------------------------------

st.header("Define Graph")

edge_probability = st.number_input(
    "Global edge flip probability p",
    min_value=0.0,
    max_value=0.499999,
    value=0.1,
    step=0.01,
    format="%.3f",
)

# Colors used in the graph display.
SOURCE_COLOR = "#2ca02c"   # green
SINK_COLOR = "#d62728"     # red
NODE_COLOR = "#4682b4"      # blue

if st.session_state.nodes:
    source_col, sink_col = st.columns(2)

    with source_col:
        source = st.selectbox(
            "Source",
            st.session_state.nodes,
            index=st.session_state.nodes.index(st.session_state.source),
            key="source_selector",
        )
        st.session_state.source = source

    with sink_col:
        sink = st.selectbox(
            "Sink",
            st.session_state.nodes,
            index=st.session_state.nodes.index(st.session_state.sink),
            key="sink_selector",
        )
        st.session_state.sink = sink

    st.caption("Source is green; target is red; ordinary nodes are blue.")

    graph_nodes = []
    for node_id in st.session_state.nodes:
        if node_id == st.session_state.source:
            color = SOURCE_COLOR
        elif node_id == st.session_state.sink:
            color = SINK_COLOR
        else:
            color = NODE_COLOR

        graph_nodes.append(
            Node(
                id=node_id,
                label=node_id,
                size=28,
                color=color,
            )
        )

    graph_edges = [
        Edge(source=u, target=v, label="")
        for u, v in st.session_state.edges
    ]

    graph_config = Config(
        width="100%",
        height=450,
        directed=False,
        physics=False,
        hierarchical=False,
    )

    selected_node = agraph(
        nodes=graph_nodes,
        edges=graph_edges,
        config=graph_config,
    )

    if selected_node is not None:
        st.info(f"Selected node: {selected_node}")
else:
    st.info("Add at least one node to display the graph.")


# -----------------------------------------------------------------------------
# Graph editing controls
# -----------------------------------------------------------------------------

st.subheader("Edit Graph")

adjacency_text = st.text_area(
    "Adjacency list",
    value="""0: 1
    1: 0, 2
    2: 1, 3
    3: 2""",
    height=180,
)

def parse_adjacency_list(text):
    adjacency = {}

    for line in text.strip().splitlines():
        if not line.strip() or ":" not in line:
            continue

        node, neighbors = line.split(":", 1)
        node = node.strip()

        adjacency[node] = [
            neighbor.strip()
            for neighbor in neighbors.split(",")
            if neighbor.strip()
        ]

    return adjacency


adjacency = parse_adjacency_list(adjacency_text)

st.code(adjacency_text, language="text")

# -----------------------------------------------------------------------------
# Compute correlations
# -----------------------------------------------------------------------------

st.header("Compute Correlations")

degree = st.number_input(
    "Low-degree polynomial degree",
    min_value=0,
    max_value=10,
    value=3,
    step=1,
)


def make_graph(adjacency, edge_probability):
    graph = Graph()
    added_edges = set()

    for u, neighbors in adjacency.items():
        for v in neighbors:
            edge = tuple(sorted((u, v)))

            if edge not in added_edges:
                graph.add_edge(u, v, p=edge_probability)
                added_edges.add(edge)

    return graph


if st.button("Compute correlations"):
    if st.session_state.source is None or st.session_state.sink is None:
        st.error("The graph must contain at least two nodes.")
    elif st.session_state.source == st.session_state.sink:
        st.error("Source and sink must be different.")
    elif not st.session_state.edges:
        st.error("The graph must contain at least one edge.")
    else:
        try:
            results = compute_all_correlations(
                make_graph(adjacency, edge_probability),
                st.session_state.source,
                st.session_state.sink,
                degree=degree,
            )

            st.subheader("Results")

            for method, result in results.items():
                value = result["corr2"] if isinstance(result, dict) else result
                st.write(f"**{method}:** {float(value):.4f}")

        except Exception as error:
            st.error(str(error))
