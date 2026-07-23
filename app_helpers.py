"""Uygulamanın arayüzden bağımsız, test edilebilir yardımcı fonksiyonları."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(*parts: str) -> Path:
    """Proje köküne göre güvenilir bir dosya yolu üretir."""
    return PROJECT_ROOT.joinpath(*parts)


def point_in_zone(cx: int, cy: int, zone: tuple[int, int, int, int]) -> bool:
    """Bir noktanın dikdörtgen tehlikeli alan içinde olup olmadığını döndürür."""
    x1, y1, x2, y2 = zone
    return x1 <= cx <= x2 and y1 <= cy <= y2


def box_intersects_zone(
    box: tuple[int, int, int, int], zone: tuple[int, int, int, int]
) -> bool:
    """Bir algılama kutusunun tehlikeli alanla kesişip kesişmediğini döndürür."""
    bx1, by1, bx2, by2 = box
    zx1, zy1, zx2, zy2 = zone
    return not (bx2 < zx1 or bx1 > zx2 or by2 < zy1 or by1 > zy2)


def get_absolute_zone(
    relative_zone: tuple[float, float, float, float] | None,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """0-1 aralığındaki alan koordinatlarını piksel koordinatlarına çevirir."""
    if relative_zone is None:
        return None

    rx1, ry1, rx2, ry2 = relative_zone
    return (
        int(rx1 * width),
        int(ry1 * height),
        int(rx2 * width),
        int(ry2 * height),
    )
