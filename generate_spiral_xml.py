# generate_spiral_xml.py
from __future__ import annotations

import math
from typing import Tuple, List

from generate_hex_mesh import ensure_unit_hex_mesh_obj


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _fmt(seq) -> str:
    if isinstance(seq, (tuple, list)):
        return " ".join(f"{v:.8g}" for v in seq)
    return f"{seq:.8g}"


def generate_spiral_tentacle_xml(
    *,
    n_segments: int = 24,
    delta_theta_deg: float = 30.0,
    taper_angle_deg: float = 15.0,
    tip_width: float = 0.012,
    tip_thickness: float = 0.010,
    tip_length: float = 0.020,
    cable_offset_frac: float = 0.45,
    joint_epsilon_frac: float = 0.08,
    base_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ground_friction: Tuple[float, float, float] = (1.2, 0.02, 0.0005),
    link_friction: Tuple[float, float, float] = (1.0, 0.02, 0.0005),
    solref: Tuple[float, float] = (0.004, 1.0),
    solimp: Tuple[float, float, float] = (0.98, 0.995, 0.001),
    k0: float = 30.0,
    damping_ratio: float = 0.55,
    armature_base: float = 0.0025,
    armature_tip_mult: float = 0.6,
    motor_gear: float = 900.0,
    tendon_stiffness: float = 2500.0,
    tendon_damping: float = 35.0,
    tendon_width: float = 0.002,
    pretension_frac: float = 0.01,
    tendon_rgba_left: Tuple[float, float, float, float] = (0.2, 0.6, 1.0, 1.0),
    tendon_rgba_right: Tuple[float, float, float, float] = (1.0, 0.5, 0.2, 1.0),
    steps_per_second: int = 600,
    mesh_obj_path: str = "assets/unit_hex_link.obj",
) -> str:
    if n_segments < 2:
        raise ValueError("n_segments must be >= 2")
    if delta_theta_deg <= 0:
        raise ValueError("delta_theta_deg must be > 0")
    if tip_width <= 0 or tip_thickness <= 0 or tip_length <= 0:
        raise ValueError("tip_width/tip_thickness/tip_length must be > 0")

    mesh_obj_path = mesh_obj_path.replace("\\", "/")
    ensure_unit_hex_mesh_obj(mesh_obj_path)

    # Log spiral ratio q = exp(b Δθ)
    alpha = math.radians(taper_angle_deg)
    b = _clamp(
        0.10 + 0.25 * ((alpha - math.radians(5.0)) / (math.radians(20.0) - math.radians(5.0))),
        0.10, 0.35
    )
    delta_theta = math.radians(delta_theta_deg)
    q = math.exp(b * delta_theta)
    a = 1.0

    def scale_at(i: int) -> float:
        return q ** (n_segments - 1 - i)  # base biggest

    def dims_at(i: int) -> Tuple[float, float, float]:
        s = scale_at(i)
        return tip_length * s, tip_width * s, tip_thickness * s

    m_tip = 0.010
    mjmin_mass = 1e-6
    mjmin_inertia = 1e-10

    def mass_at(i: int) -> float:
        s = scale_at(i)
        m = m_tip * (s ** 3)
        return max(m, mjmin_mass * 100.0)

    def inertial_at(i: int) -> Tuple[float, Tuple[float, float, float]]:
        m = mass_at(i)
        L, _, T = dims_at(i)
        r = 0.5 * T
        Ixx = 0.5 * m * (r ** 2)
        Iyy = (1.0 / 12.0) * m * (L ** 2) + 0.25 * m * (r ** 2)
        Izz = Iyy
        Ixx = max(Ixx, mjmin_inertia * 100.0)
        Iyy = max(Iyy, mjmin_inertia * 100.0)
        Izz = max(Izz, mjmin_inertia * 100.0)
        return m, (Ixx, Iyy, Izz)

    def stiffness_at(i: int) -> float:
        return k0 * (q ** (-i))

    c0 = 2.0 * damping_ratio

    def damping_at(i: int) -> float:
        return c0 * math.sqrt(max(1e-12, stiffness_at(i)))

    def armature_at(i: int) -> float:
        t = i / max(1, n_segments - 1)
        return armature_base * (1.0 - (1.0 - armature_tip_mult) * t)

    bx, by, bz = base_pos
    base_T = dims_at(0)[2]
    z0 = 0.5 * base_T

    def cable_r(i: int) -> float:
        _, W, _ = dims_at(i)
        return cable_offset_frac * 0.5 * W

    def eps_x(i: int) -> float:
        L, _, _ = dims_at(i)
        return max(1e-4, joint_epsilon_frac * L)

    def site_z(i: int) -> float:
        _, _, T = dims_at(i)
        return 0.35 * T

    left_path: List[str] = []
    right_path: List[str] = []
    link_sites: List[List[str]] = [[] for _ in range(n_segments)]

    # base sites on link_0 near proximal
    r0 = cable_r(0)
    e0 = eps_x(0)
    zS0 = site_z(0)
    link_sites[0].append(f'<site name="site_left_base"  pos="{_fmt((e0, +r0, zS0))}" size="{_fmt((0.0015 * scale_at(0),))}" />')
    link_sites[0].append(f'<site name="site_right_base" pos="{_fmt((e0, -r0, zS0))}" size="{_fmt((0.0015 * scale_at(0),))}" />')
    left_path.append('<site site="site_left_base" />')
    right_path.append('<site site="site_right_base" />')

    # joint kink sites
    for i in range(n_segments - 1):
        Li, _, _ = dims_at(i)
        rip = cable_r(i)
        ric = cable_r(i + 1)
        epi = eps_x(i)
        epc = eps_x(i + 1)
        zpi = site_z(i)
        zci = site_z(i + 1)

        lp = f"site_left_p_{i}"
        rp = f"site_right_p_{i}"
        lc = f"site_left_c_{i}"
        rc = f"site_right_c_{i}"

        link_sites[i].append(f'<site name="{lp}" pos="{_fmt((Li - epi, +rip, zpi))}" size="{_fmt((0.0015 * scale_at(i),))}" />')
        link_sites[i].append(f'<site name="{rp}" pos="{_fmt((Li - epi, -rip, zpi))}" size="{_fmt((0.0015 * scale_at(i),))}" />')
        link_sites[i + 1].append(f'<site name="{lc}" pos="{_fmt((epc, +ric, zci))}" size="{_fmt((0.0015 * scale_at(i+1),))}" />')
        link_sites[i + 1].append(f'<site name="{rc}" pos="{_fmt((epc, -ric, zci))}" size="{_fmt((0.0015 * scale_at(i+1),))}" />')

        left_path.append(f'<site site="{lp}" />')
        left_path.append(f'<site site="{lc}" />')
        right_path.append(f'<site site="{rp}" />')
        right_path.append(f'<site site="{rc}" />')

    # tip anchors
    last = n_segments - 1
    Llast = dims_at(last)[0]
    rlast = cable_r(last)
    elast = eps_x(last)
    zlast = site_z(last)
    link_sites[last].append(f'<site name="site_left_tip"  pos="{_fmt((Llast - elast, +rlast, zlast))}" size="{_fmt((0.0018 * scale_at(last),))}" />')
    link_sites[last].append(f'<site name="site_right_tip" pos="{_fmt((Llast - elast, -rlast, zlast))}" size="{_fmt((0.0018 * scale_at(last),))}" />')
    left_path.append('<site site="site_left_tip" />')
    right_path.append('<site site="site_right_tip" />')

    # rest length estimate in straight pose
    lengths = [dims_at(i)[0] for i in range(n_segments)]
    x_origin = [0.0] * n_segments
    x_origin[0] = bx
    for i in range(1, n_segments):
        x_origin[i] = x_origin[i - 1] + lengths[i - 1]
    y_origin = [by] * n_segments
    z_origin = [bz + z0] * n_segments

    site_pos_world = {}

    def add_site_world(link_i: int, name: str, local: Tuple[float, float, float]) -> None:
        site_pos_world[name] = (x_origin[link_i] + local[0], y_origin[link_i] + local[1], z_origin[link_i] + local[2])

    add_site_world(0, "site_left_base", (e0, +r0, zS0))
    add_site_world(0, "site_right_base", (e0, -r0, zS0))
    for i in range(n_segments - 1):
        Li = dims_at(i)[0]
        rip = cable_r(i)
        ric = cable_r(i + 1)
        epi = eps_x(i)
        epc = eps_x(i + 1)
        zpi = site_z(i)
        zci = site_z(i + 1)
        add_site_world(i,     f"site_left_p_{i}",  (Li - epi, +rip, zpi))
        add_site_world(i,     f"site_right_p_{i}", (Li - epi, -rip, zpi))
        add_site_world(i + 1, f"site_left_c_{i}",  (epc, +ric, zci))
        add_site_world(i + 1, f"site_right_c_{i}", (epc, -ric, zci))
    add_site_world(last, "site_left_tip", (Llast - elast, +rlast, zlast))
    add_site_world(last, "site_right_tip", (Llast - elast, -rlast, zlast))

    def extract_site_names(path_tags: List[str]) -> List[str]:
        out = []
        for tag in path_tags:
            parts = tag.split('"')
            if len(parts) >= 2:
                out.append(parts[1])
        return out

    def path_len(names: List[str]) -> float:
        L = 0.0
        for i in range(len(names) - 1):
            p = site_pos_world[names[i]]
            q2 = site_pos_world[names[i + 1]]
            dx = q2[0] - p[0]
            dy = q2[1] - p[1]
            dz = q2[2] - p[2]
            L += math.sqrt(dx * dx + dy * dy + dz * dz)
        return L

    left_rest = path_len(extract_site_names(left_path))
    right_rest = path_len(extract_site_names(right_path))
    left_spring = max(0.0, left_rest * (1.0 - pretension_frac))
    right_spring = max(0.0, right_rest * (1.0 - pretension_frac))

    # mesh assets: same OBJ, per-link scale (L,W,T)
    mesh_assets = []
    for i in range(n_segments):
        L, W, T = dims_at(i)
        mesh_assets.append(f'<mesh name="mesh_link_{i}" file="{mesh_obj_path}" scale="{_fmt((L, W, T))}" />')

    # exclude adjacent self-collisions
    exclude_xml = []
    for i in range(n_segments - 1):
        exclude_xml.append(f'<exclude body1="link_{i}" body2="link_{i+1}" />')

    rgba_link = (0.75, 0.75, 0.80, 1.0)
    rgba_tip = (0.85, 0.85, 0.95, 1.0)

    def body_xml(i: int) -> str:
        L, _, _ = dims_at(i)
        m, inertia = inertial_at(i)
        k = stiffness_at(i)
        d = damping_at(i)
        arm = armature_at(i)

        if i == 0:
            pos = (bx, by, bz + z0)
        else:
            pos = (dims_at(i - 1)[0], 0.0, 0.0)

        geom_rgba = rgba_tip if i == (n_segments - 1) else rgba_link

        joint = (
            f'<joint name="hinge_{i}" type="hinge" axis="0 0 1" pos="0 0 0" '
            f'stiffness="{_fmt(k)}" damping="{_fmt(d)}" armature="{_fmt(arm)}" />'
        )
        inertial = f'<inertial pos="{_fmt((0.5 * L, 0.0, 0.0))}" mass="{_fmt(m)}" diaginertia="{_fmt(inertia)}" />'
        geom = (
            f'<geom name="geom_{i}" type="mesh" mesh="mesh_link_{i}" '
            f'friction="{_fmt(link_friction)}" solref="{_fmt(solref)}" solimp="{_fmt(solimp)}" rgba="{_fmt(geom_rgba)}" '
            f'contype="1" conaffinity="1" />'
        )
        sites = "\n".join(link_sites[i])

        inner = ""
        if i + 1 < n_segments:
            inner = "\n" + body_xml(i + 1) + "\n"

        return (
            f'<body name="link_{i}" pos="{_fmt(pos)}">\n'
            f'{inertial}\n{joint}\n{geom}\n{sites}{inner}'
            f'</body>'
        )

    chain = body_xml(0)

    xml = f"""<mujoco model="spirob_2cable_spiral">
  <compiler angle="radian" coordinate="local" inertiafromgeom="false" />
  <option timestep="{1.0/steps_per_second:.8g}" gravity="0 0 -9.81" iterations="80" solver="Newton" />
  <size njmax="8000" nconmax="4000" />

  <default>
    <joint limited="false" />
    <geom density="0" />
    <site rgba="0 0 0 0" />
    <motor ctrlrange="-1 1" ctrllimited="true" />
  </default>

  <asset>
    <texture name="texgrid" type="2d" builtin="checker" width="512" height="512" rgb1="0.2 0.2 0.2" rgb2="0.25 0.25 0.25" />
    <material name="matgrid" texture="texgrid" texrepeat="8 8" reflectance="0.0" rgba="1 1 1 1" />
    {"".join(mesh_assets)}
  </asset>

  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="5 5 0.1" material="matgrid"
          friction="{_fmt(ground_friction)}" solref="{_fmt(solref)}" solimp="{_fmt(solimp)}" contype="1" conaffinity="1" />
    {chain}
  </worldbody>

  <contact>
    {"".join(exclude_xml)}
  </contact>

  <tendon>
    <spatial name="tendon_left" width="{tendon_width:.8g}" stiffness="{tendon_stiffness:.8g}" damping="{tendon_damping:.8g}"
             springlength="{left_spring:.8g}" rgba="{_fmt(tendon_rgba_left)}">
      {"".join(left_path)}
    </spatial>
    <spatial name="tendon_right" width="{tendon_width:.8g}" stiffness="{tendon_stiffness:.8g}" damping="{tendon_damping:.8g}"
             springlength="{right_spring:.8g}" rgba="{_fmt(tendon_rgba_right)}">
      {"".join(right_path)}
    </spatial>
  </tendon>

  <actuator>
    <motor name="motor_left" tendon="tendon_left" gear="{motor_gear:.8g}" />
    <motor name="motor_right" tendon="tendon_right" gear="{motor_gear:.8g}" />
  </actuator>

  <!-- Spiral metadata: a={a:.6g}, b={b:.6g}, Δθ={delta_theta:.6g} rad, q={q:.6g}, α={taper_angle_deg:.6g} deg -->
</mujoco>
"""
    return xml
