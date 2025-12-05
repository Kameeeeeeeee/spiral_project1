# generate_hex_mesh.py
#
# Генерирует звено щупальца в виде шестиугольной призмы.
# Исходные координаты взяты из документации и нормализованы так,
# что по оси X звено занимает от 0 до 1 (0 - корневая узкая грань).

from __future__ import annotations
from pathlib import Path


def write_hex_mesh(filename: str = "hex_base.obj") -> None:
    # координаты из документации (вид сверху, XY-плоскость)
    verts_2d = [
        (0.156, 0.00),   # v0
        (0.10, 0.325),   # v1
        (-0.10, 0.325),  # v2
        (-0.25, 0.00),   # v3
        (-0.10, -0.325), # v4
        (0.10, -0.325),  # v5
    ]

    # нормируем X в [0,1], чтобы узкая левая грань стала в x=0,
    # а правая - в x=1
    xs = [x for x, _ in verts_2d]
    min_x = min(xs)
    max_x = max(xs)
    length_x = max_x - min_x
    if length_x <= 0:
        length_x = 1.0

    norm_2d = [((x - min_x) / length_x, y) for x, y in verts_2d]

    thickness = 1.0
    half_z = thickness / 2.0

    verts_3d = []

    # верхнее основание
    for x, y in norm_2d:
        verts_3d.append((x, y, half_z))

    # нижнее основание
    for x, y in norm_2d:
        verts_3d.append((x, y, -half_z))

    faces = []

    # верхнее основание (0..5)
    faces.append((0, 1, 2))
    faces.append((0, 2, 3))
    faces.append((0, 3, 4))
    faces.append((0, 4, 5))

    # нижнее основание (6..11), ориентированное так, чтобы нормали смотрели наружу
    faces.append((8, 7, 6))
    faces.append((9, 8, 6))
    faces.append((10, 9, 6))
    faces.append((11, 10, 6))

    # боковые грани
    for i in range(6):
        j = (i + 1) % 6
        top_i = i
        top_j = j
        bot_i = i + 6
        bot_j = j + 6
        faces.append((top_i, bot_i, bot_j))
        faces.append((top_i, bot_j, top_j))

    p = Path(filename)
    with p.open("w", encoding="utf-8") as f:
        f.write("# hex link mesh normalized along x in [0,1]\n")
        for x, y, z in verts_3d:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {a+1} {b+1} {c+1}\n")

    print(f"wrote {filename}")


if __name__ == "__main__":
    write_hex_mesh()
