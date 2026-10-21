"""toric.draw puts one marker per anyon on the picture."""

import matplotlib

matplotlib.use("Agg")

import toric  # noqa: E402


def test_draw_marks_every_anyon():
    s = toric.new_state(4)
    s.apply("Z", s.torus.h(0, 0)).apply("X", s.torus.v(3, 3))  # an e pair and an m pair, the m pair across a side
    ax = toric.draw(s, labels=True)
    assert sum(len(c.get_offsets()) for c in ax.collections) == 16 + 4  # 16 vertices, 4 anyons
