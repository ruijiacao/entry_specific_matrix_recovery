import matplotlib

matplotlib.use("kitcat")

import matplotlib.pyplot as plt
import networkx as nx


def draw_graph(ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    G = nx.Graph()
    G.add_edges_from(
        [
            ("s", "a"),
            ("a", "t"),
            ("s", "b"),
            ("b", "t"),
        ]
    )

    pos = {
        "s": (0, 0),
        "a": (1, 1),
        "b": (1, -1),
        "t": (2, 0),
    }

    nx.draw_networkx(
        G,
        pos,
        ax=ax,
        node_color="lightblue",
        edge_color="gray",
    )

    ax.set_axis_off()
    return ax


print("before:", plt.get_fignums())

ax = draw_graph()

print("after draw:", plt.get_fignums())

plt.show()

print("after show:", plt.get_fignums())
