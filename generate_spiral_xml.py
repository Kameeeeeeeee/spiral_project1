# generate_spiral_xml.py
#
# Цель: физически консистентная MuJoCo модель SpiRob без "виртуальных" моментов.
# Ключевые правки относительно твоей текущей ситуации:
# 1) Только реальная актуация через tendon motor (force-based). Никаких компенсаций через qfrc_applied.
# 2) Реалистичный диапазон сил: motor_gear = 100 Н при ctrl=1 (в статье порядок десятков ньютонов).
# 3) Убираем "вязкость, формирующую движение": damping_mul и frictionloss_tip уменьшены.
# 4) Контакт устойчивый (timestep=0.001, solref быстрее), кубик с корректной инерцией, чтобы не "пролетал".
# 5) Кубик спавнится случайно на расстоянии [0.1L, 0.55L] от базы, на той же плоскости, но не на щупальце.
#
# ВАЖНО: форма "?" зависит от сценария управления (force sequence), а не от огромной силы.
# Это реализовано в debug_viewer через пресеты и плавный ramp сил.

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

    # Длина в развернутом состоянии
    total_length: float = 0.45

    # Геометрия звена
    tip_width: float = 0.0075
    tip_thickness: float = 0.0024  # увеличенная толщина по z для хватания кубика

    # Подъем над полом
    lift_z: float = 0.010

    # Сайты троса (плечо момента)
    tendon_offset_frac: float = 0.65  # увеличили плечо, чтобы легче получать "?" при тех же силах

    # Механика: градиент податливости правильный (база жестче), но без "удушающей" вязкости
    k_tip: float = 1.5
    damping_mul: float = 0.55        # было больше, из-за этого система становилась "в геле"
    frictionloss_tip: float = 0.003  # было больше, это убивало квазистатику
    armature_mul: float = 1.2

    # Масса на кончике
    m_tip: float = 2e-6

    # Реалистичный привод: ctrl=1 -> 100 Н натяжение
    motor_gear: float = 120.0

    # Контакт и устойчивость
    timestep: float = 0.001

    # Кубик
    cube_half: float = 0.02
    cube_density: float = 800.0
    cube_friction: str = "1.1 0.02 0.0001"


def _fmt(x: float) -> str:
    return f"{x:.10g}"


def _cot(x: float) -> float:
    return 1.0 / math.tan(x)


def _ensure_unit_mesh_obj() -> str:
    # Требование: звено - obj из generate_hex_mesh.py
    from generate_hex_mesh import ensure_unit_hex_mesh_obj  # type: ignore

    return ensure_unit_hex_mesh_obj("assets/hex_base.obj")


def _sample_cube_xy(
    *,
    rng: random.Random,
    total_length: float,
    cube_half: float,
    y_clear: float,
) -> tuple[float, float]:
    # Кубик: r ∈ [0.1L, 0.55L], впереди базы, не на центральной полосе щупальцы.
    r_min = 0.10 * total_length
    r_max = 0.55 * total_length

    # Сектор спавна впереди
    ang_min = -0.85 * math.pi / 2.0
    ang_max = +0.85 * math.pi / 2.0

    for _ in range(20000):
        r = rng.uniform(r_min, r_max)
        ang = rng.uniform(ang_min, ang_max)
        x = r * math.cos(ang)
        y = r * math.sin(ang)

        if x <= 0.0:
            continue
        if abs(y) < y_clear:
            continue
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
    cube_seed: int | None = None,
) -> str:
    unit_obj = _ensure_unit_mesh_obj()

    # q = exp(bΔθ), b = cot(ψ)
    delta = math.radians(delta_deg)
    b = _cot(math.radians(psi_deg))
    q = math.exp(b * delta)

    if n_segments < 2:
        raise ValueError("n_segments must be >= 2")

    # unit mesh пропорции по XY
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

    # L_tip так, чтобы сумма геом. прогрессии дала total_length
    L_tip = total_length * (q - 1.0) / (q**n_segments - 1.0)

    sx_tip = L_tip
    sy_tip = tip_width / max(1e-12, unit_y_over_x)
    sz_tip = tip_thickness

    base_thickness = sz_tip * (q ** (n_segments - 1))
    base_z = 0.5 * base_thickness + max(0.0, lift_z)

    # Safety против mjMINVAL
    mass_min = 2e-6
    inertia_min = 1e-9

    # Контакт: быстрый (меньше проникновения), но без чрезмерного "раннего" упора
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
        # i=0 база, i=N-1 кончик
        j = (n_segments - 1) - i  # j=0 tip, j=N-1 base
        s = q**j
        scale[i] = s

        sx = sx_tip * s
        sy = sy_tip * s
        sz = sz_tip * s

        L = sx
        W = sy * unit_y_over_x
        T = sz

        # mass_i = m0*q^(3j)
        m = max(mass_min, m_tip * (q ** (3.0 * j)))

        # инерция как у прямоугольного бруса
        ixx = max(inertia_min, (1.0 / 12.0) * m * (W * W + T * T))
        iyy = max(inertia_min, (1.0 / 12.0) * m * (L * L + T * T))
        izz = max(inertia_min, (1.0 / 12.0) * m * (L * L + W * W))

        # stiffness растет к базе
        k_j = k_tip * (q**j)

        # демпфирование привязываем к sqrt(k), но множитель маленький, чтобы не формировать движение
        c_j = damping_mul * math.sqrt(max(1e-12, k_j))

        # frictionloss тоже растет к базе, но с малым базовым уровнем
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

        if i >= 1:
            excludes_xml.append(f'<exclude body1="link_{i-1:02d}" body2="link_{i:02d}"/>')

        tendon_sites_left.append(f"site_left_{i:02d}")
        tendon_sites_right.append(f"site_right_{i:02d}")

    # Цепочка hinge вдоль +X, изгиб в плоскости пола (ось z)
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

        # margin умеренный: уменьшили, чтобы не мешать плотной упаковке и форме "?"
        margin = 0.0012 * scale[i]

        geom_xml = (
            f'<geom name="geom_{i:02d}" type="mesh" mesh="{mesh_name}" '
            f'friction="{friction}" solimp="{solimp}" solref="{solref}" '
            f'contype="1" conaffinity="1" condim="3" '
            f'margin="{_fmt(margin)}"/>'
        )

        # Сайты троса в середине звена, симметрично по y
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

    # Кубик: масса и корректная инерция
    cube_side = 2.0 * cube_half
    cube_vol = cube_side * cube_side * cube_side
    cube_mass = max(mass_min, cube_density * cube_vol)
    cube_I = (1.0 / 6.0) * cube_mass * (cube_side * cube_side)
    cube_I = max(inertia_min, cube_I)

    rng = random.Random(cube_seed)

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

    <!-- Кубик-цель -->
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
    <!-- tension-only: отрицательного натяжения не существует -->
    <motor name="motor_left" tendon="tendon_left" ctrlrange="0 1" ctrllimited="true"
           gear="{_fmt(motor_gear)}"/>
    <motor name="motor_right" tendon="tendon_right" ctrlrange="0 1" ctrllimited="true"
           gear="{_fmt(motor_gear)}"/>
  </actuator>
</mujoco>
"""
    return xml
