from pathlib import Path

from app_helpers import (
    box_intersects_zone,
    get_absolute_zone,
    point_in_zone,
    resolve_project_path,
)


def test_resolve_project_path_uses_project_root():
    resolved = resolve_project_path("runs", "detect", "train", "weights", "best.pt")
    assert resolved == Path(__file__).resolve().parents[1] / "runs" / "detect" / "train" / "weights" / "best.pt"


def test_get_absolute_zone_converts_relative_coordinates():
    zone = get_absolute_zone((0.25, 0.25, 0.75, 0.75), 100, 200)
    assert zone == (25, 50, 75, 150)


def test_point_in_zone_checks_membership():
    assert point_in_zone(50, 100, (0, 0, 100, 200))
    assert not point_in_zone(101, 100, (0, 0, 100, 200))


def test_box_intersects_zone_detects_overlap():
    assert box_intersects_zone((0, 0, 100, 100), (50, 50, 150, 150))
    assert box_intersects_zone((60, 40, 90, 80), (50, 50, 100, 100))
    assert not box_intersects_zone((0, 0, 10, 10), (20, 20, 30, 30))
