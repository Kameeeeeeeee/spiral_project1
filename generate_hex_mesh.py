# generate_hex_mesh.py
#
# Один раз запускаем, чтобы создать hex_base.obj.
# Это лежачий шестиугольник, вытянутый по оси X:
# кончики по X, широкая часть по Y.

def write_hex_mesh(filename: str = "hex_base.obj"):
    # Шестиугольник в плоскости XY, кончики по оси X.
    # Вид сверху (примерно):
    #
    #        (0.25, 0.5)      ( -0.25, 0.5)
    #            \               /
    #   (0.5, 0)  \             /  (-0.5, 0)
    #             /             \
    #        (0.25, -0.5)   ( -0.25, -0.5)
    #
    top_xy = [
        (0.5, 0.0),     # v0 - правый кончик
        (0.25, 0.5),    # v1
        (-0.25, 0.5),   # v2
        (-0.5, 0.0),    # v3 - левый кончик
        (-0.25, -0.5),  # v4
        (0.25, -0.5),   # v5
    ]

    verts = []
    # Два слоя по z: верх и низ, толщина = 1
    for z in (0.5, -0.5):
        for x, y in top_xy:
            verts.append((x, y, z))

    faces = []

    # Верхняя грань (z = +0.5): вершины 1..6
    faces.append((1, 2, 3, 4, 5, 6))

    # Нижняя грань (z = -0.5): 7..12 (обход в обратную сторону)
    faces.append((12, 11, 10, 9, 8, 7))

    # Боковые грани - шесть четырехугольников
    side_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    for a, b in side_pairs:
        top_a = a + 1        # 1..6
        top_b = b + 1
        bot_a = a + 1 + 6    # 7..12
        bot_b = b + 1 + 6
        faces.append((top_a, top_b, bot_b, bot_a))

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# hex base mesh (lying hexagon)\n")
        for x, y, z in verts:
            f.write(f"v {x} {y} {z}\n")
        for face in faces:
            idx_str = " ".join(str(i) for i in face)
            f.write(f"f {idx_str}\n")

    print(f"Mesh saved to {filename}")


if __name__ == "__main__":
    write_hex_mesh()
