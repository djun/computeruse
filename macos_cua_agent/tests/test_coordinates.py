from cua_agent.utils.coordinates import clamp_point, point_to_px, px_to_point


def test_px_to_point_round_trip():
    x, y = px_to_point(200, 100, scale=2.0)
    assert (x, y) == (100.0, 50.0)
    px = point_to_px(x, y, scale=2.0)
    assert px == (200, 100)


def test_clamp_point():
    x, y = clamp_point(-10, 900, width=800, height=600)
    assert x == 0.0
    assert y == 599.0
