# generate_spiral_xml.py
#
# Генерирует MJCF-модель двухтросовой SpiRob-подобной щупальцы с 24 звеньями.
# Геометрия следует дискретизации логарифмической спирали с taper angle ≈ 15°.
#
# - звенья по длине и ширине заданы экспоненциально (логарифмическая спираль);
# - размеры идут от основания (крупные) к кончику (тонкие);
# - база вынесена на край окружности, чтобы при полном сворачивании
#   кончик не упирался в основание;
# - ЖЁСТКОСТЬ: основание жёсткое, кончик мягкий;
# - ДИАПАЗОН УГЛОВ: у основания маленький, у кончика большой;
# - плечо троса больше у кончика, меньше у основания.

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

    Возвращает:
        link_lengths   - длина каждого звена (от основания к кончику)
        link_widths    - ширина каждого звена (от основания к кончику)
        base_width     - ширина у основания
        tip_width      - ширина у кончика
    """
    # Шаг по углу
    delta_theta = math.radians(delta_theta_deg)
    theta_N = num_links * delta_theta

    # Параметр спирали b:
    # для taper ≈ 15° в статье использовали b ≈ 0.22.
    b = 0.22

    # ---------- длины звеньев (сначала от кончика к основанию) ----------
    exp_b_thetaN = math.exp(b * theta_N)
    denom = exp_b_thetaN - 1.0
    if denom <= 0:
        raise ValueError("denominator for length computation is non-positive")

    lengths_tip2base: List[float] = []
    for k in range(num_links):
        theta_k = k * delta_theta
        theta_k1 = (k + 1) * delta_theta
        segment = math.exp(b * theta_k1) - math.exp(b * theta_k)
        L_k = total_length * segment / denom
        lengths_tip2base.append(L_k)

    # ---------- ширины звеньев (сначала от кончика к основанию) ----------
    # w(theta) = w0 * e^{b theta}.
    # theta = 0 - кончик, theta = theta_N - основание.
    theta_mid_last = (num_links - 0.5) * delta_theta
    w0 = base_width_target / math.exp(b * theta_mid_last)

    widths_tip2base: List[float] = []
    for k in range(num_links):
        theta_mid = (k + 0.5) * delta_theta
        w_k = w0 * math.exp(b * theta_mid)
        widths_tip2base.append(w_k)

    # Для MJCF удобнее порядок от основания к кончику.
    link_lengths = list(reversed(lengths_tip2base))
    link_widths = list(reversed(widths_tip2base))

    base_width = link_widths[0]
    tip_width = link_widths[-1]

    return link_lengths, link_widths, base_width, tip_width


def generate_spiral_tentacle_xml(
    num_links: int = 24,
    total_length: float = 0.45,      # полная длина вдоль оси (м)
    taper_angle_deg: float = 15.0,   # целевой taper (через b ≈ 0.22)
    delta_theta_deg: float = 30.0,   # шаг дискретизации по углу
    base_width_target: float = 0.06, # ширина у основания (м)
    thickness: float = 0.01,         # толщина звена по Z (м)
    link_density: float = 300.0,
    motor_gear: float = 3.0,
    joint_damping: float = 0.4,
    joint_frictionloss: float = 0.02,
) -> str:
    """
    Двухтросовая спиральная щупальцевая рука на основе логарифмической спирали.
    """

    # 1. Геометрия по спирали
    link_lengths, link_widths, base_width, tip_width = _compute_spiral_geometry(
        num_links=num_links,
        total_length=total_length,
        taper_angle_deg=taper_angle_deg,
        delta_theta_deg=delta_theta_deg,
        base_width_target=base_width_target,
    )

    base_radius = base_width / 2.0
    tip_radius = tip_width / 2.0

    # 2. Распределение жёсткости и диапазонов для суставов
    # ТЕПЕРЬ ПРАВИЛЬНО:
    #   основание жёсткое, кончик мягкий.
    stiff_base = 40.0
    stiff_tip = 2.0
    stiff_ratio = (stiff_tip / stiff_base) ** (1.0 / max(num_links - 1, 1))
    joint_stiffness: List[float] = [
        stiff_base * (stiff_ratio ** i) for i in range(num_links)
    ]
    # joint_stiffness[0] ≈ 40 (основание), joint_stiffness[-1] ≈ 2 (кончик)

    # Диапазон углов:
    #   основание - маленький, кончик - большой.
    range_base = 40.0   # у основания
    range_tip = 180.0   # у кончика
    joint_range_min: List[float] = []
    joint_range_max: List[float] = []
    for i in range(num_links):
        t = i / max(num_links - 1, 1)
        # линейная интерполяция от основания к кончику
        r = range_base * (1.0 - t) + range_tip * t
        joint_range_min.append(-r)
        joint_range_max.append(r)

    xml: List[str] = []

    xml.append('<mujoco model="spiral_tentacle_hex_spirob_2c">')
    xml.append('  <option timestep="0.002" gravity="0 0 -9.81"/>')

    xml.append('  <default>')
    xml.append(
        f'    <joint damping="{joint_damping:.4f}" frictionloss="{joint_frictionloss:.4f}" '
        'springref="0"/>'
    )
    xml.append(
        '    <geom condim="3" friction="0.8 0.1 0.01" rgba="0.9 0.9 0.9 1"/>'
    )
    xml.append('  </default>')

    # ---------- asset: базовый mesh и mesh-asset для каждого звена ----------
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

    # ---------------- worldbody ----------------
    xml.append('  <worldbody>')

    xml.append(
        '    <geom name="floor" type="plane" size="2 2 0.1" '
        'rgba="0.8 0.8 0.8 1"/>'
    )

    # суммарная длина щупальца
    total_len = sum(link_lengths)

    # приближённый радиус окружности, в которую сворачивается цепочка
    R_approx = total_len / (2.0 * math.pi)

    # базу ставим на край этой окружности
    base_x = -R_approx
    base_z = thickness * 1.5

    xml.append(
        f'    <body name="base" pos="{base_x:.6f} 0 {base_z:.6f}">'
    )
    xml.append('      <geom type="sphere" size="0.03" rgba="0.3 0.3 0.3 1"/>')

    # точки крепления тросов на базе
    xml.append(
        f'      <site name="tendon_left_base"  pos="0 { base_radius:.6f} 0" size="0.002"/>'
    )
    xml.append(
        f'      <site name="tendon_right_base" pos="0 {-base_radius:.6f} 0" size="0.002"/>'
    )

    # цепочка звеньев
    indent = "      "
    for i in range(num_links):
        L = link_lengths[i]
        W = link_widths[i]
        body_name = f"link_{i}"
        joint_name = f"joint_{i}"
        geom_name = f"link_geom_{i}"
        mesh_name = f"hex_{i}"

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
        xml.append(
            f'{indent}  <geom name="{geom_name}" type="mesh" mesh="{mesh_name}" '
            f'pos="0 0 0" density="{link_density:.1f}"/>'
        )

        half_L = 0.5 * L
        # Плечо троса:
        #   основание: arm_factor ≈ 0.4 (меньше),
        #   кончик:    arm_factor ≈ 0.9 (больше).
        t = i / max(num_links - 1, 1)
        arm_factor = 0.4 * (1.0 - t) + 0.9 * t
        y_off = 0.5 * W * arm_factor

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

    # объект-шар (пример)
    obj_r = tip_radius * 1.2
    xml.append(
        f'    <body name="obj_sphere" pos="0.0 0 {obj_r:.6f}">'
    )
    xml.append('      <joint name="obj_sphere_free" type="free"/>')
    xml.append(
        f'      <geom name="obj_sphere_geom" type="sphere" size="{obj_r:.6f}" '
        'density="300" friction="1.2 0.3 0.02" rgba="0 1 0 1"/>'
    )
    xml.append('    </body>')

    xml.append('  </worldbody>')

    # тросы
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
