import matplotlib.pyplot as plt
import numpy as np

verts_2d = [
    (0.156, 0.00),   # v0
    (0.10, 0.325),   # v1
    (-0.10, 0.325),  # v2
    (-0.25, 0.00),   # v3
    (-0.10, -0.325), # v4
    (0.10, -0.325),  # v5
]

# функция для расчета внутреннего угла в вершине i
def vertex_angle(verts, i):
    n = len(verts)
    p_prev = np.array(verts[(i - 1) % n])
    p_curr = np.array(verts[i])
    p_next = np.array(verts[(i + 1) % n])

    v1 = p_prev - p_curr
    v2 = p_next - p_curr

    dot = np.dot(v1, v2)
    norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_prod == 0:
        return 0.0
    cos_angle = np.clip(dot / norm_prod, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

angles = [vertex_angle(verts_2d, i) for i in range(len(verts_2d))]

# замыкаем шестиугольник, добавив первую точку в конец
xs = [v[0] for v in verts_2d] + [verts_2d[0][0]]
ys = [v[1] for v in verts_2d] + [verts_2d[0][1]]

plt.figure()
plt.plot(xs, ys, marker="o")

# подписи вершин и углов
for i, ((x, y), ang) in enumerate(zip(verts_2d, angles)):
    plt.text(x, y, f"v{i}", fontsize=9, ha="right", va="bottom")
    # небольшое смещение, чтобы число угла не налезало на подпись вершины
    plt.text(x + 0.015, y + 0.02, f"{ang:.1f}°", fontsize=8, ha="left", va="bottom")

plt.axhline(0)
plt.axvline(0)
plt.gca().set_aspect("equal", adjustable="box")
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Шестиугольник с подписями углов")
plt.show()
