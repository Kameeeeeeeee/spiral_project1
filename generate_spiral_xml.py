# generate_spiral_xml.py

def generate_spiral_tentacle_xml(
    num_links: int = 10,
    base_radius: float = 0.02,   # радиус у основания
    tip_radius: float = 0.006,   # радиус на кончике
    link_length: float = 0.05,
) -> str:
    """
    Щупальце из num_links звеньев, лежит горизонтально вдоль оси X на полу.
    Радиус звеньев плавно уменьшается от base_radius до tip_radius.
    Два троса идут слева и справа (по оси Y), суставы крутятся вокруг оси Z,
    так что щупальце может изгибаться только влево-вправо.

    Мяч лежит на том же уровне, перед концом щупальца.
    """

    total_length = num_links * link_length
    # высота основания: по самому толстому радиусу
    base_z = base_radius

    xml_parts = []

    xml_parts.append('<mujoco model="spiral_tentacle_2tendons_ball">')
    xml_parts.append('  <option timestep="0.002" gravity="0 0 -9.81"/>')
    xml_parts.append('')
    xml_parts.append('  <default>')
    xml_parts.append('    <joint limited="true" range="-120 120" damping="0.05"/>')
    # geom без размера - зададим size индивидуально для каждого звена
    xml_parts.append('  </default>')
    xml_parts.append('')
    xml_parts.append('  <worldbody>')

    # основание щупальца
    xml_parts.append(f'    <body name="base" pos="0 0 {base_z}">')

    # цепочка звеньев вдоль оси X
    for i in range(num_links):
        body_name = f"link_{i}"
        joint_name = f"joint_{i}"
        geom_name = f"geom_{i}"
        site_left_name = f"site_left_{i}"
        site_right_name = f"site_right_{i}"

        # линейная интерполяция радиуса от основания к кончику
        if num_links > 1:
            t = i / (num_links - 1)
        else:
            t = 0.0
        radius_i = (1 - t) * base_radius + t * tip_radius

        xml_parts.append(
            f'      <body name="{body_name}" pos="{link_length} 0 0">'
        )
        # сустав вокруг оси Z
        xml_parts.append(
            f'        <joint name="{joint_name}" type="hinge" axis="0 0 1" pos="{-link_length/2} 0 0"/>'
        )
        # капсула вдоль X с индивидуальным радиусом
        xml_parts.append(
            f'        <geom name="{geom_name}" type="capsule" '
            f'size="{radius_i} {link_length/2}" '
            f'fromto="{-link_length/2} 0 0 {link_length/2} 0 0" '
            f'density="1000"/>'
        )
        # точки для тросов слева/справа по Y на конце звена
        xml_parts.append(
            f'        <site name="{site_left_name}" '
            f'pos="{link_length/2} {radius_i} 0" size="{radius_i/2}" rgba="1 0 0 1"/>'
        )
        xml_parts.append(
            f'        <site name="{site_right_name}" '
            f'pos="{link_length/2} {-radius_i} 0" size="{radius_i/2}" rgba="0 0 1 1"/>'
        )

    # закрываем вложенные body
    for _ in range(num_links):
        xml_parts.append('      </body>')

    xml_parts.append('    </body> <!-- end base -->')

    # пол
    xml_parts.append(
        '    <geom name="floor" type="plane" size="1 1 0.1" pos="0 0 0" rgba="0.8 0.8 0.8 1"/>'
    )

    # мяч перед концом щупальца, на том же уровне (используем радиус основания)
    ball_x = total_length + 0.05
    ball_radius = base_radius * 0.8  # чуть меньше толщины у основания
    xml_parts.append(
        f'    <body name="ball" pos="{ball_x} 0 {base_z}">'
    )
    xml_parts.append('      <joint name="ball_free" type="free"/>')
    xml_parts.append(
        f'      <geom name="ball_geom" type="sphere" size="{ball_radius}" '
        'density="500" friction="1 0.1 0.01" rgba="0 1 0 1"/>'
    )
    xml_parts.append('    </body>')

    xml_parts.append('  </worldbody>')
    xml_parts.append('')

    # два троса по бокам
    xml_parts.append('  <tendon>')
    xml_parts.append('    <spatial name="tendon_left" width="0.001">')
    for i in range(num_links):
        xml_parts.append(f'      <site site="site_left_{i}"/>')
    xml_parts.append('    </spatial>')

    xml_parts.append('    <spatial name="tendon_right" width="0.001">')
    for i in range(num_links):
        xml_parts.append(f'      <site site="site_right_{i}"/>')
    xml_parts.append('    </spatial>')
    xml_parts.append('  </tendon>')
    xml_parts.append('')

    # актуаторы
    xml_parts.append('  <actuator>')
    xml_parts.append(
        '    <motor name="motor_left" tendon="tendon_left" ctrlrange="-1 1" gear="1"/>'
    )
    xml_parts.append(
        '    <motor name="motor_right" tendon="tendon_right" ctrlrange="-1 1" gear="1"/>'
    )
    xml_parts.append('  </actuator>')

    xml_parts.append('</mujoco>')

    return "\n".join(xml_parts)


if __name__ == "__main__":
    xml = generate_spiral_tentacle_xml(num_links=10)
    with open("spiral_tentacle_2tendons_ball.xml", "w", encoding="utf-8") as f:
        f.write(xml)
    print("Saved to spiral_tentacle_2tendons_ball.xml")
