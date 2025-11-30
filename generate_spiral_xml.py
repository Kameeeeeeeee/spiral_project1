# generate_spiral_xml.py
#
# Использует hex_base.obj и создает 24 звена-hex,
# которые сужаются и укорачиваются к концу щупальца.

def generate_spiral_tentacle_xml(
    num_links: int = 24,
    base_radius: float = 0.02,    # половина ширины у основания (по Y)
    tip_radius: float = 0.006,    # половина ширины на конце
    total_length: float = 0.45,   # общая длина щупальца (по X)
    thickness: float = 0.01,      # толщина по Z
) -> str:
    """
    Щупальце из num_links звеньев.
    Каждое звено - лежачий шестиугольник (mesh hex_base)
    с кончиками по оси X и шириной по оси Y.
    Звенья уменьшаются к концу. Движение только в горизонтали.
    """

    # коэффициенты сужения: от 1 до 0.4
    scales = []
    for i in range(num_links):
        if num_links > 1:
            t = i / (num_links - 1)
        else:
            t = 0.0
        s = 1.0 - 0.6 * t
        scales.append(s)

    scale_sum = sum(scales)
    base_length = total_length / scale_sum
    link_lengths = [base_length * s for s in scales]

    widths = []
    for i in range(num_links):
        if num_links > 1:
            t = i / (num_links - 1)
        else:
            t = 0.0
        radius_i = (1 - t) * base_radius + t * tip_radius
        widths.append(2.0 * radius_i)

    base_z = thickness * 0.5

    xml = []
    xml.append('<mujoco model="spiral_tentacle_hex">')
    xml.append('  <option timestep="0.002" gravity="0 0 0"/>')
    xml.append('')
    xml.append('  <asset>')
    # для каждого звена - своя mesh с нужным scale,
    # но все используют один и тот же файл hex_base.obj
    for i in range(num_links):
        L = link_lengths[i]
        W = widths[i]
        T = thickness
        xml.append(
            f'    <mesh name="hex_{i}" file="hex_base.obj" '
            f'scale="{L:.6f} {W:.6f} {T:.6f}"/>'
        )
    xml.append('  </asset>')
    xml.append('')
    xml.append('  <worldbody>')

    # база
    xml.append(f'    <body name="base" pos="0 0 {base_z:.6f}">')

    for i in range(num_links):
        body_name = f"link_{i}"
        joint_name = f"joint_{i}"
        geom_name = f"geom_{i}"
        site_left_name = f"site_left_{i}"
        site_right_name = f"site_right_{i}"

        L = link_lengths[i]
        W = widths[i]
        half_L = L / 2.0
        half_W = W / 2.0

        # каждое тело смещаем на L относительно предыдущего
        xml.append(f'      <body name="{body_name}" pos="{L:.6f} 0 0">')

        # сустав вокруг Z, опорная точка примерно между звеньями
        xml.append(
            f'        <joint name="{joint_name}" type="hinge" axis="0 0 1" '
            f'pos="{-half_L:.6f} 0 0" range="-120 120" damping="0.05"/>'
        )

        # геом - шестиугольник, уже "лежит" вдоль X, узкая к узкой
        # hex_base имеет длину 1 от кончика до кончика,
        # scale по X = L, по Y = W, по Z = thickness
        xml.append(
            f'        <geom name="{geom_name}" type="mesh" mesh="hex_{i}" '
            f'pos="0 0 0" density="1000" rgba="0.9 0.9 0.9 1"/>'
        )

        # сайты для тросов по боковым кромкам (левый/правый)
        xml.append(
            f'        <site name="{site_left_name}" '
            f'pos="{half_L:.6f} {half_W:.6f} 0" size="{half_W/4:.6f}" '
            f'rgba="1 0 0 1"/>'
        )
        xml.append(
            f'        <site name="{site_right_name}" '
            f'pos="{half_L:.6f} {-half_W:.6f} 0" size="{half_W/4:.6f}" '
            f'rgba="0 0 1 1"/>'
        )

    # закрываем все link-body
    for _ in range(num_links):
        xml.append('      </body>')

    xml.append('    </body> <!-- end base -->')

    # "пол" далеко снизу
    xml.append(
        '    <geom name="floor" type="plane" size="5 5 0.1" '
        'pos="0 0 -1" rgba="0.8 0.8 0.8 1"/>'
    )

    # мяч перед концом щупальца
    ball_x = total_length + 0.05
    ball_radius = base_radius * 0.7
    xml.append(f'    <body name="ball" pos="{ball_x:.6f} 0 {base_z:.6f}">')
    xml.append('      <joint name="ball_free" type="free"/>')
    xml.append(
        f'      <geom name="ball_geom" type="sphere" size="{ball_radius:.6f}" '
        'density="300" friction="1 0.1 0.01" rgba="0 1 0 1"/>'
    )
    xml.append('    </body>')

    xml.append('  </worldbody>')
    xml.append('')

    # тросы
    xml.append('  <tendon>')
    xml.append('    <spatial name="tendon_left" width="0.001">')
    for i in range(num_links):
        xml.append(f'      <site site="site_left_{i}"/>')
    xml.append('    </spatial>')
    xml.append('    <spatial name="tendon_right" width="0.001">')
    for i in range(num_links):
        xml.append(f'      <site site="site_right_{i}"/>')
    xml.append('    </spatial>')
    xml.append('  </tendon>')
    xml.append('')

    # актуаторы
    xml.append('  <actuator>')
    xml.append(
        '    <motor name="motor_left" tendon="tendon_left" '
        'ctrlrange="-1 1" gear="1"/>'
    )
    xml.append(
        '    <motor name="motor_right" tendon="tendon_right" '
        'ctrlrange="-1 1" gear="1"/>'
    )
    xml.append('  </actuator>')

    xml.append('</mujoco>')

    return "\\n".join(xml)


if __name__ == "__main__":
    xml = generate_spiral_tentacle_xml()
    with open("spiral_tentacle_hex.xml", "w", encoding="utf-8") as f:
        f.write(xml)
    print("Saved to spiral_tentacle_hex.xml")
