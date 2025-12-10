# generate_spiral_xml.py
#
# SpiRob-подобная щупальца с двумя тросами.
# Логарифмическая спираль, 24 звена, taper ~15°.
#
# Геометрия звена - ТВОЙ шестиугольный mesh (type="mesh"), он же используется
# для коллизий, никакие капсулы не добавляются.
#
# Для уменьшения пролёта шара:
# - уменьшен timestep;
# - немного усилен solver;
# - добавлены solref/solimp и чуть увеличен margin;
# - у шара коллизионный радиус чуть больше визуального.

from __future__ import annotations
import math
from typing import List


def _compute_spiral_geometry(
    num_links: int,
    total_length: float,
    taper_angle_deg: float,
    delta_theta_deg: float,
    base_width_target: float,
) -> tuple[List[float], List[float], float, float]:
    """
    Вычисляет длины и ширины звеньев из логарифмической спирали.
    """
    delta_theta = math.radians(delta_theta_deg)
    theta_N = num_links * delta_theta

    # параметр логарифмической спирали (≈15° taper)
    b = 0.22

    exp_b_thetaN = math.exp(b * theta_N)
    denom = exp_b_thetaN - 1.0
    if denom <= 0:
        raise ValueError("denominator for length computation is non positive")

    lengths_tip2base: List[float] = []
    for k in range(num_links):
        theta_k = k * delta_theta
        theta_k1 = (k + 1) * delta_theta
        segment = math.exp(b * theta_k1) - math.exp(b * theta_k)
        L_k = total_length * segment / denom
        lengths_tip2base.append(L_k)

    theta_mid_last = (num_links - 0.5) * delta_theta
    w0 = base_width_target / math.exp(b * theta_mid_last)

    widths_tip2base: List[float] = []
    for k in range(num_links):
        theta_mid = (k + 0.5) * delta_theta
        w_k = w0 * math.exp(b * theta_mid)
        widths_tip2base.append(w_k)

    link_lengths = list(reversed(lengths_tip2base))
    link_widths = list(reversed(widths_tip2base))

    base_width = link_widths[0]
    tip_width = link_widths[-1]

    return link_lengths, link_widths, base_width, tip_width


def generate_spiral_tentacle_xml(
    num_links: int = 24,
    total_length: float = 0.20,
    taper_angle_deg: float = 15.0,
    delta_theta_deg: float = 30.0,
    base_width_target: float = 0.03,
    thickness: float = 0.05,
    link_density: float = 300.0,
    motor_gear: float = 5000.0,
    joint_damping: float = 0.25,
    joint_frictionloss: float = 0.01,
) -> str:

    link_lengths, link_widths, base_width, tip_width = _compute_spiral_geometry(
        num_links=num_links,
        total_length=total_length,
        taper_angle_deg=taper_angle_deg,
        delta_theta_deg=delta_theta_deg,
        base_width_target=base_width_target,
    )

    base_radius = base_width / 2.0

    # Профиль жёсткости:
    # - основание всё ещё жёстче, но разница с кончиком НЕ в сотни раз,
    #   чтобы двигались все звенья, а не только хвост.
    STIFF_BASE = 6.0
    STIFF_TIP = 1.0
    # экспоненту делаем меньше, чтобы градиент был более плавным
    SOFTNESS_EXP = 1.5


    joint_stiffness: List[float] = []
    for i in range(num_links):
        t = i / max(num_links - 1, 1)
        s = t ** SOFTNESS_EXP
        k = STIFF_BASE * (STIFF_TIP / STIFF_BASE) ** s
        joint_stiffness.append(k)

    RANGE_BASE = 45.0
    RANGE_TIP = 300.0
    joint_range_min: List[float] = []
    joint_range_max: List[float] = []
    for i in range(num_links):
        t = i / max(num_links - 1, 1)
        r = RANGE_BASE * (1.0 - t) + RANGE_TIP * t
        joint_range_min.append(-r)
        joint_range_max.append(r)

    xml: List[str] = []

    xml.append('<mujoco model="spiral_tentacle_hex_spirob_2c">')
    # timestep уменьшаем, solver чуть усиливаем, но без Newton
    xml.append(
        '  <option timestep="0.00015" gravity="0 0 -9.81" '
        'integrator="Euler" iterations="120"/>'
    )

    xml.append('  <default>')
    xml.append(
        f'    <joint damping="{joint_damping:.4f}" '
        f'frictionloss="{joint_frictionloss:.4f}" springref="0"/>'
    )
    xml.append(
        '    <geom condim="3" friction="0.8 0.1 0.01" '
        'solimp="0.9 0.99 0.001 0.5 2" solref="0.002 1" '
        'rgba="0.9 0.9 0.9 1"/>'
    )
    xml.append('  </default>')

    # ---------- asset ----------
    xml.append('  <asset>')
    xml.append('    <mesh name="hex_base" file="hex_base.obj"/>')
    for i in range(num_links):
        L = link_lengths[i]
        W = link_widths[i]
        xml.append(
            f'    <mesh name="hex_{i}" file="hex_base.obj" '
            f'scale="{L:.6f} {W:.6f} {thickness:.6f}"/>'
        )
    xml.append('  </asset>')

    # ---------- worldbody ----------
    xml.append('  <worldbody>')
    xml.append(
        '    <geom name="floor" type="plane" pos="0 0 0" size="5 5 0.1" '
        'rgba="0.8 0.8 0.8 1"/>'
    )

    total_len = sum(link_lengths)
    R_approx = total_len / (2.0 * math.pi)

    # База стоит на полу
    base_z = thickness / 2.0
    base_x = -R_approx
    base_y = 0.0

    xml.append(
        f'    <body name="base" pos="{base_x:.6f} {base_y:.6f} {base_z:.6f}">'
    )
    xml.append('      <geom type="sphere" size="0.015" rgba="0.3 0.3 0.3 1"/>')
    xml.append(
        f'      <site name="tendon_left_base"  pos="0 { base_radius:.6f} 0" size="0.002"/>'
    )
    xml.append(
        f'      <site name="tendon_right_base" pos="0 {-base_radius:.6f} 0" size="0.002"/>'
    )

    indent = "      "
    for i in range(num_links):
        L = link_lengths[i]
        W = link_widths[i]
        body_name = f"link_%d" % i
        joint_name = f"joint_%d" % i
        geom_name = f"link_geom_%d" % i
        mesh_name = f"hex_%d" % i

        if i == 0:
            pos_str = "0 0 0"
        else:
            prev_L = link_lengths[i - 1]
            pos_str = f"{prev_L:.6f} 0 0"

        xml.append(f'{indent}<body name="{body_name}" pos="{pos_str}">')
        xml.append(
            f'{indent}  <joint name="{joint_name}" type="hinge" axis="0 0 1" '
            f'pos="0 0 0" range="{joint_range_min[i]:.1f} {joint_range_max[i]:.1f}" '
            f'stiffness="{joint_stiffness[i]:.3f}"/>'
        )
        # Шестиугольный mesh - единственная геометрия (и визуал, и коллизии)
        xml.append(
            f'{indent}  <geom name="{geom_name}" type="mesh" mesh="{mesh_name}" '
            f'pos="0 0 0" density="{link_density:.1f}" margin="0.003"/>'
        )

        half_L = 0.5 * L
        # Плечо троса: кончик всё ещё имеет чуть большее плечо,
        # но без безумного роста.
        ARM_BASE = 0.7
        ARM_TIP = 1.0
        t = i / max(num_links - 1, 1)
        arm_factor = ARM_BASE * (1.0 - t) + ARM_TIP * t
        y_off = 0.5 * W * arm_factor


        xml.append(
            f'{indent}  <site name="tendon_left_{i}"  pos="{half_L:.6f} { y_off:.6f} 0" size="0.002"/>'
        )
        xml.append(
            f'{indent}  <site name="tendon_right_{i}" pos="{half_L:.6f} {-y_off:.6f} 0" size="0.002"/>'
        )

        indent += "  "

    for _ in range(num_links):
        indent = indent[:-2]
        xml.append(f'{indent}</body>')
    xml.append('    </body>  <!-- base -->')

    # ---------- объект-шар ----------
    OBJ_RADIUS_VIS = 0.01
    OBJ_RADIUS_COLL = 0.018  # немного больше, чем визуальный

    obj_x_rel = 0.8
    obj_x = base_x + total_len * obj_x_rel
    obj_y = (base_radius + OBJ_RADIUS_COLL) * 1.3
    obj_z = OBJ_RADIUS_COLL

    xml.append(
        f'    <body name="obj_sphere" pos="{obj_x:.6f} {obj_y:.6f} {obj_z:.6f}">'
    )
    xml.append('      <joint name="obj_sphere_free" type="free"/>')

    # коллизионная оболочка
    xml.append(
        f'      <geom name="obj_sphere_col" type="sphere" size="{OBJ_RADIUS_COLL:.6f}" '
        'density="2000" friction="1.0 0.4 0.03" margin="0.004" rgba="0 0 0 0"/>'
    )
    # визуальный шар
    xml.append(
        f'      <geom name="obj_sphere_vis" type="sphere" size="{OBJ_RADIUS_VIS:.6f}" '
        'contype="0" conaffinity="0" rgba="0 1 0 1"/>'
    )

    xml.append('    </body>')

    xml.append('  </worldbody>')

    # ---------- тросы ----------
    xml.append('  <tendon>')
    xml.append('    <spatial name="tendon_left" limited="false" width="0.002">')
    xml.append('      <site site="tendon_left_base"/>')
    for i in range(num_links):
        xml.append(f'      <site site="tendon_left_{i}"/>')
    xml.append('    </spatial>')

    xml.append('    <spatial name="tendon_right" limited="false" width="0.002">')
    xml.append('      <site site="tendon_right_base"/>')
    for i in range(num_links):
        xml.append(f'      <site site="tendon_right_{i}"/>')
    xml.append('    </spatial>')
    xml.append('  </tendon>')

    # ---------- актуаторы ----------
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
