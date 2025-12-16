from __future__ import annotations

import os
from typing import List, Tuple

Vec3 = Tuple[float, float, float]
Face = Tuple[int, int, int]


def _triangulate_fan(n: int) -> List[Face]:
    tris: List[Face] = []
    for i in range(1, n - 1):
        tris.append((0, i, i + 1))
    return tris


def unit_y_span_over_x_span(verts_2d: List[Tuple[float, float]]) -> float:
    xs = [p[0] for p in verts_2d]
    ys = [p[1] for p in verts_2d]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xspan = max(1e-12, xmax - xmin)
    yspan = max(1e-12, ymax - ymin)
    return yspan / xspan


def build_hex_prism_mesh(
    verts_2d: List[Tuple[float, float]],
    thickness: float = 1.0,
) -> Tuple[List[Vec3], List[Face]]:
    if len(verts_2d) < 3:
        raise ValueError("Need at least 3 vertices")

    xs = [p[0] for p in verts_2d]
    ys = [p[1] for p in verts_2d]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    xspan = max(1e-12, xmax - xmin)
    ymid = 0.5 * (ymin + ymax)

    scale = 1.0 / xspan

    v2 = []
    for x, y in verts_2d:
        xn = (x - xmin) * scale
        yn = (y - ymid) * scale
        v2.append((xn, yn))

    n = len(v2)
    hz = 0.5 * thickness

    verts: List[Vec3] = []
    for x, y in v2:
        verts.append((x, y, -hz))
    for x, y in v2:
        verts.append((x, y, +hz))

    faces: List[Face] = []

    top = _triangulate_fan(n)
    bottom = [(a, c, b) for (a, b, c) in top]

    faces.extend(bottom)
    faces.extend([(a + n, b + n, c + n) for (a, b, c) in top])

    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, j + n))
        faces.append((i, j + n, i + n))

    return verts, faces


def write_obj(path: str, verts: List[Vec3], faces: List[Face]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("o spirob_unit_hex\n")
        for x, y, z in verts:
            f.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for a, b, c in faces:
            f.write(f"f {a+1} {b+1} {c+1}\n")
    return path


def ensure_unit_hex_mesh_obj(path: str) -> str:
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path

    verts_2d = [
        (0.156, 0.00),
        (0.10, 0.50),
        (-0.10, 0.50),
        (-0.25, 0.00),
        (-0.10, -0.50),
        (0.10, -0.50),
    ]
    verts, faces = build_hex_prism_mesh(verts_2d, thickness=1.0)
    return write_obj(path, verts, faces)


if __name__ == "__main__":
    print(ensure_unit_hex_mesh_obj("assets/hex_base.obj"))
