# generate_spiral_xml.py
#
# Генератор MJCF для двухтроссовой спиральной щупальцы SpiRob (rigid-link + spatial tendons)
# плюс кубик-объект для grasp-and-transport.
#
# Все актуальные изменения:
# - Кубик: cube_half = 0.02, корректная масса и инерция (I = (1/6) m a^2), жесткий контакт.
# - Спавн кубика: расстояние r равномерно в [0.1L, 0.55L], случайный угол в секторе впереди базы,
#   кубик не спавнится на центральной полосе щупальцы.
# - Контакт: solref сделан быстрее (0.003 1.2), timestep = 0.001, nconmax/njmax увеличены.
# - Щупальца толще по z (tip_thickness = 0.0024) и приподнята над полом.
# - Чтобы вернуть сильное сворачивание при более "жестком" контакте:
#   1) motor_gear увеличен до 2600 (больше максимальное натяжение),
#   2) margin на сегментах уменьшен до 0.0020*scale (контакт не стартует слишком рано),
#   3) damping_mul и frictionloss_tip слегка снижены (меньше диссипации).

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SpiralParams:
    n_segments: int = 24
    delta_deg: float = 30.0

    # Логарифмическая спираль: r(θ) = a·exp(bθ), b = cot(ψ)
    psi_deg: float = 77.6

    # Геометрия в развернутом (прямом) состоянии
    total_length: float = 0.45

    # Ширина звена и "высота" (толщина) по z
    tip_width: float = 0.0075
    tip_thickness: float = 0.0024

    # Поднятие над полом (к 0.5*толщины базы)
    lift_z: float = 0.010

    # Сайты тросов симметрично относительно центральной оси
    tendon_offset_frac: float = 0.55

    # База жестче, кончик мягче
    k_tip: float = 3.0
    damping_mul: float = 1.2  # было 1.6, снизили для более сильного изгиба

    # Потери для устойчивости (снизили, чтобы не "душить" изгиб)
    frictionloss_tip: float = 0.012  # было 0.018
    armature_mul: float = 1.2

    # Масса на кончике, дальше растет по q^(3j)
    m_tip: float = 2e-6

    # Актуаторы по tendon (увеличили, чтобы вернуть сильное сворачивание)
    motor_gear: float = 2600.0

    # Меньший шаг интегрирования снижает "пролет" через контакт
    timestep: float = 0.001

    # Кубик (размер задается половиной ребра)
    cube_half: float = 0.02
    cube_density: float = 800.0
    cube_friction: str = "1.1 0.02 0.0001"


def _fmt(x: float) -> str:
    return f"{x:.10g}"


def _cot(x: float) -> float:
    return 1.0 / math.tan(x)


def _ensure_unit_mesh_obj() -> str:
    # Используем obj, который генерирует generate_hex_mesh.py (требование проекта).
    from generate_hex_mesh import ensure_unit_hex_mesh_obj  # type: ignore

    return ensure_unit_hex_mesh_obj("assets/hex_base.obj")


def _sample_cube_xy(
    *,
    rng: random.Random,
    total_length: float,
    cube_half: float,
    y_clear: float,
) -> tuple[float, float]:
    # Кубик спавнится на расстоянии r от базы: r ∈ [0.1L, 0.55L], равномерно по r.
    r_min = 0.10 * total_length
    r_max = 0.55 * total_length

    # Сектор впереди базы, чтобы спавн был пригоден для grasp.
    ang_min = -0.85 * math.pi / 2.0
    ang_max = +0.85 * math.pi / 2.0

    for _ in range(12000):
        r = rng.uniform(r_min, r_max)
        ang = rng.uniform(ang_min, ang_max)
        x = r * math.cos(ang)
        y = r * math.sin(ang)

        # Щупальца лежит вдоль +X, поэтому берем только x>0.
        if x <= 0.0:
            continue

        # Не спавним на центральной полосе щупальцы.
        if abs(y) < y_clear:
            continue

        # Зазор от базы, чтобы кубик не пересекался со стартовым звеном.
        if x < (cube_half * 2.5):
            continue

        return x, y

    return 0.30 * total_length, 0.12 * total_length


def generate_spiral_tentacle_xml(
    *,
    n_segments: int = SpiralParams.n_segments,
    delta_deg: float = SpiralParams.delta_deg,
    psi_deg: float = SpiralParams.psi_deg,
    total_length: float = SpiralParams.total_length,
    tip_width: float = SpiralParams.tip_width,
    tip_thickness: float = SpiralParams.tip_thickness,
    lift_z: float = SpiralParams.lift_z,
    tendon_offset_frac: float = SpiralParams.tendon_offset_frac,
    k_tip: float = SpiralParams.k_tip,
    damping_mul: float = SpiralParams.damping_mul,
    frictionloss_tip: float = SpiralParams.frictionloss_tip,
    armature_mul: float = SpiralParams.armature_mul,
    m_tip: float = SpiralParams.m_tip,
    motor_gear: float = SpiralParams.motor_gear,
    timestep: float = SpiralParams.timestep,
    cube_half: float = SpiralParams.cube_half,
    cube_density: float = SpiralParams.cube_density,
    cube_friction: str = SpiralParams.cube_friction,
    # seed для воспроизводимого спавна (полезно для отладки/RL)
    cube_seed: int | None = None,
) -> str:
    unit_obj = _ensure_unit_mesh_obj()

    # Δθ дискретизация по углу, q = exp(bΔθ), b = cot(ψ)
    delta = math.radians(delta_deg)
    b = _cot(math.radians(psi_deg))
    q = math.exp(b * delta)

    if n_segments < 2:
        raise ValueError("n_segments must be >= 2")

    # Отношение размеров unit-mesh по XY для корректной ширины.
    try:
        from generate_hex_mesh import unit_y_span_over_x_span  # type: ignore

        verts_2d = [
            (0.156, 0.00),
            (0.10, 0.50),
            (-0.10, 0.50),
            (-0.25, 0.00),
            (-0.10, -0.50),
            (0.10, -0.50),
        ]
        unit_y_over_x = float(unit_y_span_over_x_span(verts_2d))
    except Exception:
        unit_y_over_x = 2.4630541871921187

    # Длины звеньев геом. прогрессией по q, суммарно total_length.
    L_tip = total_length * (q - 1.0) / (q**n_segments - 1.0)

    # Масштабирование меша: x-span=1, y-span=unit_y_over_x, z-span=1.
    sx_tip = L_tip
    sy_tip = tip_width / max(1e-12, unit_y_over_x)
    sz_tip = tip_thickness

    # Толщина базы.
    base_thickness = sz_tip * (q ** (n_segments - 1))

    # Поднятие щупальцы над полом.
    base_z = 0.5 * base_thickness + max(0.0, lift_z)

    # Защита от mjMINVAL.
    mass_min = 2e-6
    inertia_min = 1e-9

    # Контакт делаем быстрым и стабильным.
    solimp = "0.95 0.95 0.01"
    solref = "0.003 1.2"
    friction = "1.0 0.01 0.0001"

    meshes_xml: list[str] = []
    excludes_xml: list[str] = []
    body_lines: list[str] = []

    tendon_sites_left: list[str] = []
    tendon_sites_right: list[str] = []

    link_len = [0.0] * n_segments
    link_w = [0.0] * n_segments
    mass = [0.0] * n_segments
    Ixx = [0.0] * n_segments
    Iyy = [0.0] * n_segments
    Izz = [0.0] * n_segments
    stiffness = [0.0] * n_segments
    damping = [0.0] * n_segments
    frictionloss = [0.0] * n_segments
    armature = [0.0] * n_segments
    scale = [0.0] * n_segments

    for i in range(n_segments):
        # j=0 на кончике, j=N-1 у основания.
        j = (n_segments - 1) - i
        s = q**j
        scale[i] = s

        sx = sx_tip * s
        sy = sy_tip * s
        sz = sz_tip * s

        L = sx
        W = sy * unit_y_over_x
        T = sz

        # Масса масштабируется по q^(3j).
        m = max(mass_min, m_tip * (q ** (3.0 * j)))

        # Инерция как у прямоугольного бруса.
        ixx = max(inertia_min, (1.0 / 12.0) * m * (W * W + T * T))
        iyy = max(inertia_min, (1.0 / 12.0) * m * (L * L + T * T))
        izz = max(inertia_min, (1.0 / 12.0) * m * (L * L + W * W))

        # Жесткость растет к базе.
        k_j = k_tip * (q**j)
        c_j = damping_mul * math.sqrt(max(1e-12, k_j))

        # Потери и armature.
        f_j = frictionloss_tip * (q**j)
        a_j = armature_mul * m * (L * L)

        link_len[i] = L
        link_w[i] = W
        mass[i] = m
        Ixx[i] = ixx
        Iyy[i] = iyy
        Izz[i] = izz
        stiffness[i] = k_j
        damping[i] = c_j
        frictionloss[i] = f_j
        armature[i] = a_j

        mesh_name = f"hex_{i:02d}"
        meshes_xml.append(
            f'<mesh name="{mesh_name}" file="{unit_obj}" scale="{_fmt(sx)} {_fmt(sy)} {_fmt(sz)}"/>'
        )

        # Исключаем контакт соседей.
        if i >= 1:
            excludes_xml.append(f'<exclude body1="link_{i-1:02d}" body2="link_{i:02d}"/>')

        tendon_sites_left.append(f"site_left_{i:02d}")
        tendon_sites_right.append(f"site_right_{i:02d}")

    # Цепочка hinge в плоскости пола (ось z).
    indent = "    "
    for i in range(n_segments):
        name = f"link_{i:02d}"

        if i == 0:
            pos = f"0 0 {_fmt(base_z)}"
            joint_xml = ""
        else:
            pos = f"{_fmt(link_len[i-1])} 0 0"
            joint_xml = (
                f'<joint name="hinge_{i:02d}" type="hinge" axis="0 0 1" '
                f'limited="true" range="-2.6 2.6" '
                f'stiffness="{_fmt(stiffness[i])}" '
                f'damping="{_fmt(damping[i])}" '
                f'frictionloss="{_fmt(frictionloss[i])}" '
                f'armature="{_fmt(armature[i])}"/>'
            )

        inertial_xml = (
            f'<inertial pos="{_fmt(0.5 * link_len[i])} 0 0" '
            f'mass="{_fmt(mass[i])}" '
            f'diaginertia="{_fmt(Ixx[i])} {_fmt(Iyy[i])} {_fmt(Izz[i])}"/>'
        )

        mesh_name = f"hex_{i:02d}"

        # Контакт для мешей: margin нужен, но слишком большой мешает плотной упаковке.
        # Компромисс: 0.0020*scale (было 0.0030*scale).
        margin = 0.0020 * scale[i]

        geom_xml = (
            f'<geom name="geom_{i:02d}" type="mesh" mesh="{mesh_name}" '
            f'friction="{friction}" solimp="{solimp}" solref="{solref}" '
            f'contype="1" conaffinity="1" condim="3" '
            f'margin="{_fmt(margin)}"/>'
        )

        # Сайты тросов в середине звена, симметрично по y.
        y_off = tendon_offset_frac * 0.5 * link_w[i]
        site_x = 0.5 * link_len[i]
        left_site = tendon_sites_left[i]
        right_site = tendon_sites_right[i]
        site_size = max(0.0009, 0.0012 * scale[i])

        sites_xml = (
            f'<site name="{left_site}" pos="{_fmt(site_x)} {_fmt(+y_off)} 0" size="{_fmt(site_size)}"/>'
            f'<site name="{right_site}" pos="{_fmt(site_x)} {_fmt(-y_off)} 0" size="{_fmt(site_size)}"/>'
        )

        body_lines.append(f'{indent * i}<body name="{name}" pos="{pos}">')
        if joint_xml:
            body_lines.append(f"{indent * (i + 1)}{joint_xml}")
        body_lines.append(f"{indent * (i + 1)}{inertial_xml}")
        body_lines.append(f"{indent * (i + 1)}{geom_xml}")
        body_lines.append(f"{indent * (i + 1)}{sites_xml}")

    for i in reversed(range(n_segments)):
        body_lines.append(f"{indent * i}</body>")

    tendon_left_sites = "".join([f'<site site="{s}"/>' for s in tendon_sites_left])
    tendon_right_sites = "".join([f'<site site="{s}"/>' for s in tendon_sites_right])

    # Кубик: масса из плотности и объема.
    cube_side = 2.0 * cube_half
    cube_vol = cube_side * cube_side * cube_side
    cube_mass = max(mass_min, cube_density * cube_vol)

    # Кубик: корректная инерция (ключевой фикс от "пролета").
    cube_I = (1.0 / 6.0) * cube_mass * (cube_side * cube_side)
    cube_I = max(inertia_min, cube_I)

    # Спавн кубика.
    rng = random.Random(cube_seed)

    # Полоса запрета по y, чтобы не спавнить на теле щупальцы.
    base_width = tip_width * (q ** (n_segments - 1))
    y_clear = max(1.8 * cube_half, 1.0 * base_width)

    cx, cy = _sample_cube_xy(rng=rng, total_length=total_length, cube_half=cube_half, y_clear=y_clear)
    cz = cube_half

    xml = f"""<mujoco model="spirob_2tendon_with_cube">
  <compiler angle="radian" coordinate="local" meshdir="."/>
  <option timestep="{_fmt(timestep)}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <size njmax="12000" nconmax="12000"/>

  <default>
    <geom rgba="0.75 0.78 0.82 1"/>
    <site rgba="0.95 0.2 0.2 0.7"/>
  </default>

  <asset>
    {''.join(meshes_xml)}
  </asset>

  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="5 5 0.1"
          friction="{friction}" solimp="{solimp}" solref="{solref}" rgba="0.2 0.2 0.2 1"/>

    <!-- Кубик-цель: случайное расстояние r в [0.1L, 0.55L] -->
    <body name="cube" pos="{_fmt(cx)} {_fmt(cy)} {_fmt(cz)}">
      <freejoint/>
      <inertial mass="{_fmt(cube_mass)}" pos="0 0 0"
               diaginertia="{_fmt(cube_I)} {_fmt(cube_I)} {_fmt(cube_I)}"/>
      <geom name="cube_geom" type="box" size="{_fmt(cube_half)} {_fmt(cube_half)} {_fmt(cube_half)}"
            rgba="0.15 0.65 0.25 1" friction="{cube_friction}"
            solimp="{solimp}" solref="{solref}" contype="1" conaffinity="1" condim="3"
            margin="{_fmt(0.003)}"/>
    </body>

{chr(10).join(body_lines)}
  </worldbody>

  <contact>
    {''.join(excludes_xml)}
  </contact>

  <tendon>
    <spatial name="tendon_left" width="0.002">
      {tendon_left_sites}
    </spatial>
    <spatial name="tendon_right" width="0.002">
      {tendon_right_sites}
    </spatial>
  </tendon>

  <actuator>
    <!-- tension-only -->
    <motor name="motor_left" tendon="tendon_left" ctrlrange="0 1" ctrllimited="true"
           gear="{_fmt(motor_gear)}"/>
    <motor name="motor_right" tendon="tendon_right" ctrlrange="0 1" ctrllimited="true"
           gear="{_fmt(motor_gear)}"/>
  </actuator>
</mujoco>
"""
    return xml
