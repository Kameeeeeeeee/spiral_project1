# generate_hex_mesh.py
from __future__ import annotations

import os
from typing import List, Tuple

Vec3 = Tuple[float, float, float]
Face = Tuple[int, int, int]


def _triangulate_fan(n: int) -> List[Face]:
    # polygon vertices 0..n-1 (assume CCW)
    tris: List[Face] = []
    for i in range(1, n - 1):
        tris.append((0, i, i + 1))
    return tris


def build_hex_prism_mesh(
    verts_2d: List[Tuple[float, float]],
    thickness: float = 1.0,
) -> Tuple[List[Vec3], List[Face]]:
    """
    Extrude a 2D polygon into a closed triangular prism.

    Output is a "unit link" mesh with:
      - X normalized into [0, 1]
      - Y centered and scaled so that original Y-range becomes about [-0.5, 0.5]
      - Z in [-thickness/2, +thickness/2]
    """
    if len(verts_2d) < 3:
        raise ValueError("Need at least 3 vertices")

    xs = [p[0] for p in verts_2d]
    ys = [p[1] for p in verts_2d]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    xspan = max(1e-12, xmax - xmin)
    yspan = max(1e-12, ymax - ymin)

    v2 = []
    for x, y in verts_2d:
        xn = (x - xmin) / xspan
        yn = (y - 0.5 * (ymin + ymax)) / yspan
        v2.append((xn, yn))

    n = len(v2)
    hz = 0.5 * thickness

    verts: List[Vec3] = []
    for x, y in v2:
        verts.append((x, y, -hz))
    for x, y in v2:
        verts.append((x, y, +hz))

    faces: List[Face] = []

    # caps
    top_tris = _triangulate_fan(n)
    bottom_tris = [(a, c, b) for (a, b, c) in top_tris]  # flip for outward normals
    faces.extend(bottom_tris)
    faces.extend([(a + n, b + n, c + n) for (a, b, c) in top_tris])

    # sides
    for i in range(n):
        j = (i + 1) % n
        bi, bj = i, j
        ti, tj = i + n, j + n
        faces.append((bi, bj, tj))
        faces.append((bi, tj, ti))

    return verts, faces


def write_obj(path: str, verts: List[Vec3], faces: List[Face], obj_name: str = "unit_hex") -> str:
    """
    Minimal Wavefront OBJ writer (triangles only).
    Faces are 1-indexed in OBJ.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"o {obj_name}\n")
        for x, y, z in verts:
            f.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for a, b, c in faces:
            f.write(f"f {a+1} {b+1} {c+1}\n")
    return path


def ensure_unit_hex_mesh_obj(path: str) -> str:
    """
    Creates the base unit mesh as OBJ if it does not exist.
    """
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path

    verts_2d = [
        (0.156, 0.00),
        (0.10,  0.50),
        (-0.10, 0.50),
        (-0.25, 0.00),
        (-0.10, -0.50),
        (0.10, -0.50),
    ]
    v, f = build_hex_prism_mesh(verts_2d, thickness=1.0)
    return write_obj(path, v, f, obj_name="spirob_unit_hex")


if __name__ == "__main__":
    print(ensure_unit_hex_mesh_obj("assets/unit_hex_link.obj"))
