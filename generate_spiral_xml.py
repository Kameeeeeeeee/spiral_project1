# generate_spiral_xml.py

def generate_spiral_tentacle_xml(
    num_links: int = 10,
    link_length: float = 0.05,
    link_radius: float = 0.01,
) -> str:
    """
    Генерирует MJCF XML строку с щупальцем:
    - num_links звеньев, каждое - капсула и hinge сустав
    - два троса: левый и правый, проходящие по бокам звеньев
    """

    xml_parts = []

    # Заголовок
    xml_parts.append('<mujoco model="spiral_tentacle_2tendons">')
    xml_parts.append('  <option timestep="0.002" gravity="0 0 -9.81"/>')
    xml_parts.append('')
    xml_parts.append('  <default>')
    xml_parts.append('    <joint limited="true" range="-120 120" damping="0.05"/>')
    xml_parts.append(
        f'    <geom type="capsule" size="{link_radius} {link_length/2}" density="1000"/>'
    )
    xml_parts.append('  </default>')
    xml_parts.append('')
    xml_parts.append('  <worldbody>')
    xml_parts.append('    <!-- основание щупальца -->')
    xml_parts.append('    <body name="base" pos="0 0 0">')

    # Цепочка звеньев - вложенные body
    for i in range(num_links):
        body_name = f"link_{i}"
        joint_name = f"joint_{i}"
        geom_name = f"geom_{i}"
        site_left_name = f"site_left_{i}"
        site_right_name = f"site_right_{i}"

        xml_parts.append(
            f'      <body name="{body_name}" pos="0 0 {link_length}">'
        )
        xml_parts.append(
            f'        <joint name="{joint_name}" type="hinge" axis="0 1 0" pos="0 0 {-link_length/2}"/>'
        )
        xml_parts.append(
            f'        <geom name="{geom_name}" fromto="0 0 {-link_length/2} 0 0 {link_length/2}"/>'
        )
        # Левый site (y > 0)
        xml_parts.append(
            f'        <site name="{site_left_name}" pos="0 {link_radius} {link_length/2}" size="{link_radius/2}" rgba="1 0 0 1"/>'
        )
        # Правый site (y < 0)
        xml_parts.append(
            f'        <site name="{site_right_name}" pos="0 {-link_radius} {link_length/2}" size="{link_radius/2}" rgba="0 0 1 1"/>'
        )

    # Закрываем вложенные body в обратном порядке
    for i in range(num_links):
        xml_parts.append('      </body>')

    # Пол
    xml_parts.append('    </body> <!-- end base -->')
    xml_parts.append(
        '    <geom name="floor" type="plane" size="1 1 0.1" pos="0 0 0" rgba="0.8 0.8 0.8 1"/>'
    )
    xml_parts.append('  </worldbody>')
    xml_parts.append('')

    # Два троса: левый и правый
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

    # Актуаторы, тянущие тросы
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
    # Если хочешь сохранить в отдельный файл
    xml = generate_spiral_tentacle_xml(num_links=10)
    with open("spiral_tentacle_2tendons.xml", "w", encoding="utf-8") as f:
        f.write(xml)
    print("Saved to spiral_tentacle_2tendons.xml")
