"""Draw a PauliState the way the browser app does (needs matplotlib).

Z strings run along edges (solid orange), X strings cross edges on the dual lattice (dashed blue). e anyons are
orange circles on vertices, m anyons blue squares on plaquettes. A Z-basis configuration (from sample_picture) is
drawn as purple segments across its |1> edges, which is where its loops show. The faded right and top lines are the
left and bottom lines again: the torus is glued there.
"""

import numpy as np

E_COLOR, M_COLOR, CONFIG, GRID, MUTED = "#eb6834", "#2a78d6", "#4a3aa7", "#c9c8c2", "#8a8984"


def _segments(torus, e):
    """The edge e as a segment between vertices, and its dual segment between plaquette centres."""
    i, j, horizontal = torus.edge_info(e)
    if horizontal:
        return ((i, j), (i + 1, j)), ((i + 0.5, j - 0.5), (i + 0.5, j + 0.5))
    return ((i, j), (i, j + 1)), ((i - 0.5, j + 0.5), (i + 0.5, j + 0.5))


def draw(state, ax=None, title=None, labels=False, legend=True, config=None):
    """Draw the strings and anyons of `state` on its L x L torus.

    labels=True writes every edge number. config (0/1 per edge) draws a Z-basis configuration instead of the strings.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    t, L = state.torus, state.torus.L
    if ax is None:
        ax = plt.subplots(figsize=(4.6, 4.6) if labels else (3.4, 3.4))[1]
    clip = Rectangle((-0.05, -0.05), L + 0.1, L + 0.1, transform=ax.transData)
    shifts = [(dx, dy) for dx in (-L, 0, L) for dy in (-L, 0, L)]  # copies, so strings through a side reappear

    for k in range(L + 1):
        alpha = 0.5 if k == L else 1
        ax.plot([0, L], [k, k], color=GRID, lw=1, ls="--" if k == L else "-", alpha=alpha, zorder=0)
        ax.plot([k, k], [0, L], color=GRID, lw=1, ls="--" if k == L else "-", alpha=alpha, zorder=0)
    vx, vy = np.meshgrid(np.arange(L), np.arange(L))
    ax.scatter(vx.ravel(), vy.ravel(), s=6, color=MUTED, zorder=1)

    for e in range(t.n_edges):
        edge, dual = _segments(t, e)
        if labels:
            (x1, y1), (x2, y2) = edge
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, str(e), fontsize=6, color=MUTED, ha="center", va="center",
                    bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none"}, zorder=5)
        if config is None:
            lines = [(*edge, state.b[e], {"color": E_COLOR, "lw": 3.5}),
                     (*dual, state.a[e], {"color": M_COLOR, "lw": 2.5, "ls": (0, (4, 2.5))})]
        else:
            lines = [(*dual, config[e], {"color": CONFIG, "lw": 2.6})]
        for (x1, y1), (x2, y2), on, style in lines:
            if on:
                for dx, dy in shifts:
                    ln = ax.plot([x1 + dx, x2 + dx], [y1 + dy, y2 + dy], solid_capstyle="round", zorder=3, **style)[0]
                    ln.set_clip_path(clip)

    e_anyons, m_anyons = state.anyons()
    for i, j in e_anyons:
        ax.scatter(i, j, s=80, color=E_COLOR, edgecolors="white", linewidths=1.5, zorder=4)
    for i, j in m_anyons:
        ax.scatter(i + 0.5, j + 0.5, s=80, marker="s", color=M_COLOR, edgecolors="white", linewidths=1.5, zorder=4)

    e_mark = Line2D([], [], marker="o", ls="", color=E_COLOR, markersize=8, label="e anyon")
    m_mark = Line2D([], [], marker="s", ls="", color=M_COLOR, markersize=8, label="m anyon")
    if config is None:
        handles = [Line2D([], [], color=E_COLOR, lw=3.5, label="Z string"), e_mark,
                   Line2D([], [], color=M_COLOR, lw=2.5, ls=(0, (4, 2.5)), label="X string"), m_mark]
    else:  # a picture: only what is on it
        handles = [Line2D([], [], color=CONFIG, lw=2.6, label="|1> edge")]
        handles += [e_mark] * bool(e_anyons) + [m_mark] * bool(m_anyons)
    if legend:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2 if config is None else 3,
                  frameon=False, fontsize=7, handlelength=2)
    ax.set_xlim(-0.4, L + 0.4)
    ax.set_ylim(-0.4, L + 0.4)
    ax.set_aspect("equal")
    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    ax.tick_params(length=0, labelsize=7, labelcolor=MUTED)
    for side in ax.spines.values():
        side.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9)
    return ax
