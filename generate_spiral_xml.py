# generate_spiral_xml.py
#
# Генерирует MJCF-модель щупальца:
# - звенья: mesh hex_base.obj, сужающиеся к концу
# - привод: два троса (tendon_left, tendon_right) и два мотора

from __future__ import annotations
from typing import List


def _linspace(start: float, end: float, n: int) -> List[float]:
    if n <= 1:
        return [start]
    step = (end - start) / (n - 1)
    return [start + i * step for i in range(n)]


def generate_spiral_tentacle_xml(
    num_links: int = 24,
    total_length: float = 0.45,
    base_radius: float = 0.02,
    tip_radius: float = 0.006,
    thickness: float = 0.01,
    link_density: float = 300.0,
    motor_gear: float = 3.0,
    joint_damping: float = 0.4,
    joint_frictionloss: float = 0.02,
) -> str:
    """
    Щупальце из шестиугольных звеньев, приводимая в движение двумя тросами.
    Вдоль левого и правого края проходят spatial-tendon, к которым привязаны моторы.
    """

    # относительные длины звеньев вдоль щупальца
    rel_lengths = _linspace(1.0, 0.6, num_links)
    length_sum = sum(rel_lengths)
    base_len = total_length / length_sum
    link_lengths = [base_len * r for r in rel_lengths]

    # ширина (по Y) плавно сужается
    widths = _linspace(base_radius * 2.0, tip_radius * 2.0, num_links)

    xml: list[str] = []

    xml.append('<mujoco model="spiral_tentacle_hex">')
    xml.append('  <option timestep="0.002" gravity="0 0 -9.81"/>')
    xml.append('  <default>')
    xml.append(
        '    <joint damping="{:.4f}" frictionloss="{:.4f}"/>'.format(
            joint_damping, joint_frictionloss
        )
    )
    xml.append(
        '    <geom condim="3" friction="0.8 0.1 0.01" rgba="0.9 0.9 0.9 1"/>'
    )
    xml.append('  </default>')

    # asset: один базовый mesh и по mesh-asset на звено с нужным scale
    xml.append('  <asset>')
    xml.append('    <mesh name="hex_base" file="hex_base.obj"/>')
    for i in range(num_links):
        L = link_lengths[i]
        W = widths[i]
        xml.append(
            f'    <mesh name="hex_{i}" file="hex_base.obj" '
            f'scale="{L:.6f} {W:.6f} {thickness:.6f}"/>'
        )
    xml.append('  </asset>')

    xml.append('  <worldbody>')

    # пол
    xml.append(
        '    <geom name="floor" type="plane" size="2 2 0.1" '
        'rgba="0.8 0.8 0.8 1"/>'
    )

    # база
    base_z = thickness * 1.5
    xml.append(f'    <body name="base" pos="0 0 {base_z:.6f}">')
    xml.append('      <geom type="sphere" size="0.03" rgba="0.3 0.3 0.3 1"/>')
    # точки крепления тросов на базе
    xml.append(f'      <site name="tendon_left_base"  pos="0 { base_radius:.6f} 0" size="0.002"/>')
    xml.append(f'      <site name="tendon_right_base" pos="0 {-base_radius:.6f} 0" size="0.002"/>')

    # цепочка звеньев (каждое звено - вложенный body)
    indent = "      "
    for i in range(num_links):
        L = link_lengths[i]
        W = widths[i]
        body_name = f"link_{i}"
        joint_name = f"joint_{i}"
        geom_name = f"link_geom_{i}"
        mesh_name = f"hex_{i}"

        # смещение тела относительно родителя:
        # первое звено сидит в базе, остальные - на конце предыдущего
        if i == 0:
            pos_str = "0 0 0"
        else:
            prev_L = link_lengths[i - 1]
            pos_str = f"{prev_L:.6f} 0 0"

        xml.append(f'{indent}<body name="{body_name}" pos="{pos_str}">')
        xml.append(
            f'{indent}  <joint name="{joint_name}" type="hinge" axis="0 0 1" '
            f'pos="0 0 0" range="-120 120"/>'
        )
        xml.append(
            f'{indent}  <geom name="{geom_name}" type="mesh" mesh="{mesh_name}" '
            f'pos="0 0 0" density="{link_density:.1f}"/>'
        )
        # сайты для тросов по левому/правому краю звена (примерно по середине длины)
        half_L = 0.5 * L
        y_off = 0.5 * W
        xml.append(
            f'{indent}  <site name="tendon_left_{i}"  pos="{half_L:.6f} { y_off:.6f} 0" size="0.002"/>'
        )
        xml.append(
            f'{indent}  <site name="tendon_right_{i}" pos="{half_L:.6f} {-y_off:.6f} 0" size="0.002"/>'
        )

        indent += "  "

    # закрываем body звеньев и базу
    for _ in range(num_links):
        indent = indent[:-2]
        xml.append(f'{indent}</body>')
    xml.append('    </body>  <!-- base -->')

    # шар-объект: free joint, позиция тут только начальная,
    # реальный спавн происходит в среде
    total_len = sum(link_lengths)
    obj_r = base_radius * 0.7
    xml.append(
        f'    <body name="obj_sphere_hi" pos="{total_len * 0.7:.6f} 0 {obj_r:.6f}">'
    )
    xml.append('      <joint name="obj_sphere_hi_free" type="free"/>')
    xml.append(
        f'      <geom name="obj_sphere_hi_geom" type="sphere" size="{obj_r:.6f}" '
        'density="300" friction="1.2 0.3 0.02" rgba="0 1 0 1"/>'
    )
    xml.append('    </body>')

    xml.append('  </worldbody>')

    # тросы
    xml.append('  <tendon>')
    # левый
    xml.append('    <spatial name="tendon_left" limited="false" width="0.002">')
    xml.append('      <site site="tendon_left_base"/>')
    for i in range(num_links):
        xml.append(f'      <site site="tendon_left_{i}"/>')
    xml.append('    </spatial>')
    # правый
    xml.append('    <spatial name="tendon_right" limited="false" width="0.002">')
    xml.append('      <site site="tendon_right_base"/>')
    for i in range(num_links):
        xml.append(f'      <site site="tendon_right_{i}"/>')
    xml.append('    </spatial>')
    xml.append('  </tendon>')

    # два мотора, по одному на трос
    xml.append('  <actuator>')
    xml.append(
        f'    <motor name="motor_left"  tendon="tendon_left"  gear="{motor_gear:.3f}"/>'
    )
    xml.append(
        f'    <motor name="motor_right" tendon="tendon_right" gear="{motor_gear:.3f}"/>'
    )
    xml.append('  </actuator>')

    xml.append('</mujoco>')

    return "\n".join(xml)


if __name__ == "__main__":
    xml = generate_spiral_tentacle_xml()
    with open("spiral_tentacle_hex.xml", "w", encoding="utf-8") as f:
        f.write(xml)
    print("Saved spiral_tentacle_hex.xml")
