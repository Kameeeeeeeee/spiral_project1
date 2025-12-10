# generate_hex_mesh.py
#
# Базовая шестиугольная призма для звена щупальца.
# Меш нормируется так, что вдоль оси X длина = 1 (от широкой грани к узкой),
# по Y - ширина около [-0.5, 0.5], по Z - толщина = 1.
#
# В MJCF mesh масштабируется по X, Y и Z под конкретную
# длину звена, ширину и толщину.

from __future__ import annotations
from pathlib import Path


def write_hex_mesh(filename: str = "hex_base.obj") -> None:
    # Координаты в плоскости XY (вид сверху).
    # Узкая грань справа (x ~ 0.156), широкая слева (x ~ -0.25).
    verts_2d = [
        (0.156, 0.00),   # v0 - правая "узкая" вершина
        (0.10,  0.50),   # v1
        (-0.10, 0.50),   # v2
        (-0.25, 0.00),   # v3 - левая "широкая" вершина
        (-0.10, -0.50),  # v4
        (0.10,  -0.50),  # v5
    ]

    # Нормируем X в [0, 1], чтобы 0 - широкая сторона (корень),
    # 1 - узкая сторона (кончик звена).
    xs = [x for x, _ in verts_2d]
    min_x = min(xs)
    max_x = max(xs)
    length_x = max_x - min_x if max_x > min_x else 1.0

    norm_2d = [((x - min_x) / length_x, y) for x, y in verts_2d]

    thickness = 1.0
    half_z = thickness / 2.0

    verts_3d: list[tuple[float, float, float]] = []

    # Верхнее основание (z = +1/2)
    for x, y in norm_2d:
        verts_3d.append((x, y, half_z))

    # Нижнее основание (z = -1/2)
    for x, y in norm_2d:
        verts_3d.append((x, y, -half_z))

    faces: list[tuple[int, int, int]] = []

    # Верхнее основание (0..5) - триангуляция вокруг v0
    faces.extend([
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 4),
        (0, 4, 5),
    ])

    # Нижнее основание (6..11), ориентация так, чтобы нормали смотрели наружу
    faces.extend([
        (8, 7, 6),
        (9, 8, 6),
        (10, 9, 6),
        (11, 10, 6),
    ])

    # Боковые грани
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
        f.write("# hex link mesh, X in [0,1], Y around [-0.5,0.5], Z in [-0.5,0.5]\n")
        for x, y, z in verts_3d:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {a+1} {b+1} {c+1}\n")

    print(f"wrote {filename}")


if __name__ == "__main__":
    write_hex_mesh()
