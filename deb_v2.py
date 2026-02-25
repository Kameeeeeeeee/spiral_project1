from __future__ import annotations

import math
import time
import os
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import trimesh

from pynput import keyboard


N_SEGMENTS = 19
TARGET_LENGTH_M = 0.20
GEAR = 100.0
ALPHA_DEFAULT = 0.992
ALPHA_STEP = 0.002
ALPHA_MIN = 0.9
ALPHA_MAX = 0.9995
TIP_BOOST_GAIN = 0.75

T_STEP_DEFAULT = 0.5
T_MAX = 100.0
CABLE_MU = 0.18
CABLE_BIAS = 0.006
USE_CAPSTAN_DEMO_ONLY = True

TIMESTEP = 0.0005

FLOOR_Z = -0.05
FLOOR_FRICTION = "0.85 0.002 0.0002"
SOLIMP = "0.998 0.998 0.0002"
SOLREF = "0.003 1.0"
MARGIN = 0.00025
GAP = 0.0000
DENSITY = 10000.0
GEOM_EULER = "0 0 0"
CLEARANCE_X = 0.0006
SPRING_STIFFNESS = 0.06
SITE_X_FRAC = 0.15
SITE_X_MIN = 0.0006
PIVOT_X = 0.0010
DEMO_L_CURL = 95.0
DEMO_R_CURL = 0.0
DEMO_L_END = 0.0
DEMO_R_END = 95.0
DEMO_T_RAMP = 0.0
DEMO_T_SETTLE = 0.3
DEMO_T_CROSS = 32.0
DEMO_T_HOLD = 1.0
DT_SLEW_RAMP = 800.0
DT_SLEW_CROSS = 3.0

MU_STATIC = 0.22
MU_KINETIC = 0.04
DTRATE_LO = 30.0
DTRATE_HI = 250.0
BALL_RADIUS = 0.012
BALL_DENSITY = 1000.0
BALL_Z_CLEARANCE = 0.002
BALL_CLEARANCE_Y = 0.002
BALL_FRICTION = "0.08 0.0010 0.0006"
BALL_SPAWN_RADIUS = 0.0
BALL_MIN_Y_CLEAR = 0.0
BALL_BASE_X = 0.0
BALL_BASE_Y = 0.0
BALL_BASE_SEED = 0
LEFT_Y_SIGN = -1.0
CAM_TOP_NAME = "top"
CAM_TOP_POS = (0.0, 0.0, 0.35)
CAM_TOP_XYAXES = "1 0 0 0 -1 0"
CAM_TOP_FOVY = 60.0
marker_length_m = 0.004
MARKER_LENGTH_PER_ID = {
    0:  0.008,  # world marker on seg_00
    1:  0.008,
    2:  0.008,
    3:  0.008,
    4:  0.008,
    5:  0.008,
    6:  0.008,
    7:  0.006,
    8:  0.006,
    9:  0.006,
    10: 0.006,
    11: 0.006,
    12: 0.006,
    13: 0.006,
    14: 0.004,
    15: 0.004,
    16: 0.004,
    17: 0.004,
    18: 0.004,
}
MARKER_THICKNESS_M = 0.0006
MARKER_SURFACE_CLEARANCE_M = 0.0002
MARKER_EULER = "0 0 0"
OFFSCREEN_WIDTH = 2048
OFFSCREEN_HEIGHT = 2048


def _windows_short_path(path: Path) -> str:
    if os.name != "nt":
        return str(path)
    try:
        import ctypes
        from ctypes import wintypes

        get_short = ctypes.windll.kernel32.GetShortPathNameW
        get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        get_short.restype = wintypes.DWORD
        buf = ctypes.create_unicode_buffer(32768)
        result = get_short(str(path), buf, len(buf))
        if result == 0:
            return str(path)
        return buf.value
    except Exception:
        return str(path)


@dataclass
class DomainRandCfg:
    enabled: bool = True
    seed: int = 123
    log_on_reset: bool = True

    seg_fric_slide: tuple[float, float] = (0.7, 1.4)
    seg_fric_tors: tuple[float, float] = (0.7, 1.6)
    seg_fric_roll: tuple[float, float] = (0.5, 2.0)

    ball_fric_slide: tuple[float, float] = (0.9, 1.1)
    ball_fric_tors: tuple[float, float] = (0.8, 1.2)
    ball_fric_roll: tuple[float, float] = (0.7, 1.3)

    dof_damping: tuple[float, float] = (0.7, 1.5)
    dof_frictionloss: tuple[float, float] = (0.7, 1.8)
    dof_spring: tuple[float, float] = (0.8, 1.3)

    mass_scale: tuple[float, float] = (0.8, 1.25)
    per_link_mass_jitter: float = 0.06

    mu_static: tuple[float, float] = (0.16, 0.30)
    mu_kinetic: tuple[float, float] = (0.02, 0.08)
    cable_bias: tuple[float, float] = (0.0, 0.02)


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(lo + (hi - lo) * rng.random())


def _logu(rng: np.random.Generator, lo: float, hi: float) -> float:
    lo = max(lo, 1e-9)
    hi = max(hi, lo * 1.001)
    return float(math.exp(_u(rng, math.log(lo), math.log(hi))))


def _fmt(x: float) -> str:
    return f"{x:.10g}"


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def _smoothstep(u: float) -> float:
    u = _clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def _sample_ball_xy(
    spawn_radius: float,
    min_y_clear: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    for _ in range(100):
        u = float(rng.random())
        v = float(rng.random())
        r = spawn_radius * math.sqrt(u)
        theta = -0.5 * math.pi + math.pi * v
        ball_x = r * math.cos(theta)
        ball_y = r * math.sin(theta)
        if abs(ball_y) >= min_y_clear:
            return ball_x, ball_y
    max_x = math.sqrt(max(0.0, spawn_radius * spawn_radius - min_y_clear * min_y_clear))
    ball_x = float(rng.random()) * max_x
    ball_y = min_y_clear if float(rng.random()) < 0.5 else -min_y_clear
    return ball_x, ball_y


@dataclass
class ForceController:
    T_left: float = 0.0
    T_right: float = 0.0
    T_step: float = T_STEP_DEFAULT
    Tmax: float = T_MAX
    alpha: float = ALPHA_DEFAULT
    running: bool = True
    T_left_target: float = 0.0
    T_right_target: float = 0.0
    demo_active: bool = False
    demo_stage: int = 0
    demo_t0: float = 0.0
    demo_L0: float = 0.0
    demo_R0: float = 0.0
    demo_uR: float = 0.0
    demo_uL: float = 0.0
    T_left_seg: list[float] = field(default_factory=list)
    T_right_seg: list[float] = field(default_factory=list)
    T_left_prev: float = 0.0
    T_right_prev: float = 0.0
    mu_static: float = MU_STATIC
    mu_kinetic: float = MU_KINETIC
    cable_bias: float = CABLE_BIAS


def build_mjcf(marker_family: str = "aruco") -> str:
    if trimesh is None:
        raise SystemExit("ERROR: trimesh is not installed. Install with: pip install trimesh")

    global BALL_SPAWN_RADIUS, BALL_MIN_Y_CLEAR, BALL_BASE_X, BALL_BASE_Y

    base_dir = Path(__file__).resolve().parent
    stl_dir = base_dir / "assets" / "spiral_tent_stls_vir"
    family = str(marker_family).strip().lower()
    if family not in {"aruco", "apriltag"}:
        raise SystemExit(f"ERROR: Unsupported marker_family '{marker_family}'. Use 'aruco' or 'apriltag'.")

    marker_dir = base_dir / "assets" / ("Aruco4x4" if family == "aruco" else "AprilTag36h11")
    if family == "aruco":
        marker_file_pattern = "aruco_4x4_id_{:02d}.png"
        marker_name_prefix = "aruco"
    else:
        marker_file_pattern = "tag36h11_id_{:02d}.png"
        marker_name_prefix = "apriltag"

    if not stl_dir.is_dir():
        raise SystemExit(f"ERROR: STL directory not found: {stl_dir}")
    if not marker_dir.is_dir():
        raise SystemExit(f"ERROR: Marker directory not found: {marker_dir}")

    stl_paths = [stl_dir / f"seg_{i:02d}.stl" for i in range(N_SEGMENTS)]
    for path in stl_paths:
        if not path.exists():
            raise SystemExit(f"ERROR: Missing STL file: {path}")
    marker_img_paths = [marker_dir / marker_file_pattern.format(i) for i in range(N_SEGMENTS)]
    for path in marker_img_paths:
        if not path.exists():
            raise SystemExit(f"ERROR: Missing marker image ({family}): {path}")

    seg_info = []
    measured_length = 0.0
    for path in stl_paths:
        mesh = trimesh.load_mesh(str(path), force="mesh")
        if mesh.is_empty:
            raise SystemExit(f"ERROR: Empty mesh: {path}")
        bounds = mesh.bounds
        xmin, ymin, zmin = bounds[0]
        xmax, ymax, zmax = bounds[1]
        xspan = float(xmax - xmin)
        vx = mesh.vertices[:, 0]
        x_back_q = float(np.quantile(vx, 0.005))
        x_front_q = float(np.quantile(vx, 0.995))
        xspan_eff = x_front_q - x_back_q
        yspan = float(ymax - ymin)
        zspan = float(zmax - zmin)
        seg_info.append(
            {
                "xmin": float(xmin),
                "ymin": float(ymin),
                "zmin": float(zmin),
                "xmax": float(xmax),
                "ymax": float(ymax),
                "zmax": float(zmax),
                "xspan": xspan,
                "x_back_q": x_back_q,
                "x_front_q": x_front_q,
                "xspan_eff": xspan_eff,
                "yspan": yspan,
                "zspan": zspan,
            }
        )
        measured_length += xspan_eff

    if measured_length <= 0.0:
        raise SystemExit("ERROR: Bad measured length from STL bounds")

    scale_factor = TARGET_LENGTH_M / measured_length
    scaled_length = measured_length * scale_factor
    seg0_zspan = seg_info[0]["zspan"] * scale_factor
    base_z_offset = FLOOR_Z + 0.5 * seg0_zspan

    base_width = seg_info[0]["yspan"] * scale_factor
    tip_width = seg_info[-1]["yspan"] * scale_factor

    print("=== STL SCALE SUMMARY ===")
    print(f"measured_length   = {measured_length:.6f} m")
    print(f"scale_factor      = {scale_factor:.8f}")
    print(f"scaled_length     = {scaled_length:.6f} m")
    print(f"base_width_scaled = {base_width:.6f} m")
    print(f"tip_width_scaled  = {tip_width:.6f} m")

    meshdir = Path(_windows_short_path(stl_dir)).as_posix()

    rng = np.random.default_rng(BALL_BASE_SEED)
    spawn_radius = scaled_length * 0.8
    min_y_clear = 0.5 * max(base_width, tip_width) + BALL_RADIUS + BALL_CLEARANCE_Y
    ball_x, ball_y = _sample_ball_xy(spawn_radius, min_y_clear, rng)
    ball_z = FLOOR_Z + BALL_RADIUS + BALL_Z_CLEARANCE
    BALL_SPAWN_RADIUS = spawn_radius
    BALL_MIN_Y_CLEAR = min_y_clear
    BALL_BASE_X = ball_x
    BALL_BASE_Y = ball_y

    mesh_lines = []
    for i in range(N_SEGMENTS):
        mesh_lines.append(
            f'<mesh name="mesh_{i:02d}" file="seg_{i:02d}.stl" '
            f'scale="{_fmt(scale_factor)} {_fmt(scale_factor)} {_fmt(scale_factor)}"/>'
        )
    marker_asset_lines = []
    for i in range(N_SEGMENTS):
        marker_file = Path(_windows_short_path(marker_img_paths[i])).as_posix()
        marker_asset_lines.append(
            f'<texture name="{marker_name_prefix}_tex_{i:02d}" type="2d" file="{marker_file}"/>'
        )
        marker_asset_lines.append(
            f'<material name="{marker_name_prefix}_mat_{i:02d}" texture="{marker_name_prefix}_tex_{i:02d}" '
            f'rgba="1 1 1 1" specular="0.02" shininess="0.01" reflectance="0"/>'
        )

    body_lines = []
    indent = "    "
    for i in range(N_SEGMENTS):
        info = seg_info[i]
        yspan = info["yspan"] * scale_factor
        x_back_s = info["x_back_q"] * scale_factor
        y_center_s = 0.5 * (info["ymin"] + info["ymax"]) * scale_factor
        z_center_s = 0.5 * (info["zmin"] + info["zmax"]) * scale_factor

        geom_pos = (-(x_back_s + PIVOT_X), -y_center_s, -z_center_s)
        marker_center_x = 0.5 * (info["xmin"] + info["xmax"]) * scale_factor + geom_pos[0]
        marker_center_y = 0.5 * (info["ymin"] + info["ymax"]) * scale_factor + geom_pos[1]
        marker_top_z = info["zmax"] * scale_factor + geom_pos[2]
        marker_center_z = (
            marker_top_z + 0.5 * MARKER_THICKNESS_M + MARKER_SURFACE_CLEARANCE_M
        )

        xlen_s = info["xspan_eff"] * scale_factor
        x_hinge_span = max(1e-6, xlen_s - 2.0 * PIVOT_X)
        site_x = max(SITE_X_MIN, SITE_X_FRAC * x_hinge_span)
        y_site = 0.5 * yspan - 0.002
        y_site = max(y_site, 0.0022)
        if i in (0, 1):
            site_x = max(site_x, 0.45 * x_hinge_span)
            y_site = max(y_site, 0.0045)

        site_pos_l = f"{_fmt(site_x)} {_fmt(LEFT_Y_SIGN * y_site)} 0"
        site_pos_r = f"{_fmt(site_x)} {_fmt(-LEFT_Y_SIGN * y_site)} 0"

        if i == 0:
            body_pos = f"0 0 {_fmt(base_z_offset)}"
        else:
            prev_xlen = seg_info[i - 1]["xspan_eff"] * scale_factor
            pitch = prev_xlen + CLEARANCE_X
            body_pos = f"{_fmt(pitch)} 0 0"

        body_lines.append(f"{indent * i}<body name=\"seg_{i:02d}\" pos=\"{body_pos}\">")
        if i > 0:
            s = i / (N_SEGMENTS - 1)
            fric = 0.002 + 0.006 * (s ** 2)
            damp = 0.006 + 0.014 * (s ** 2)
            stiff = SPRING_STIFFNESS * (1.6 - 0.9 * s)
            stiff = max(0.001, stiff)
            if i == 1:
                stiff *= 0.45
            body_lines.append(
                f"{indent * (i + 1)}<joint name=\"joint_{i:02d}\" type=\"hinge\" "
                f"axis=\"0 0 1\" limited=\"true\" range=\"-0.52 0.52\" "
                f"solreflimit=\"0.0005 1\" solimplimit=\"0.95 0.99 0.001\" "
                f"stiffness=\"{_fmt(stiff)}\" springref=\"0\" "
                f"damping=\"{_fmt(damp)}\" frictionloss=\"{_fmt(fric)}\"/>"
            )

        #if i <= 2:
        #    fric_slide = 0.08
        #    fric_tors = 0.0003
        #    fric_roll = 0.0
        #else:
        fric_slide = 1.6
        fric_tors = 0.015
        fric_roll = 0.00005
        #
        friction_i = f"{_fmt(fric_slide)} {_fmt(fric_tors)} {_fmt(fric_roll)}"
        condim_i = 3 if i <= 2 else 6

        body_lines.append(
            f"{indent * (i + 1)}<geom name=\"geom_{i:02d}\" type=\"mesh\" "
            f"mesh=\"mesh_{i:02d}\" pos=\"{_fmt(geom_pos[0])} {_fmt(geom_pos[1])} {_fmt(geom_pos[2])}\" "
            f"euler=\"{GEOM_EULER}\" "
            f"density=\"{_fmt(DENSITY)}\" rgba=\"0.85 0.86 0.9 1\" "
            f"condim=\"{condim_i}\" contype=\"1\" conaffinity=\"1\" "
            f"friction=\"{friction_i}\" solimp=\"{SOLIMP}\" solref=\"{SOLREF}\" "
            f"margin=\"{_fmt(MARGIN)}\" gap=\"{_fmt(GAP)}\"/>"
        )
        marker_length_i = float(MARKER_LENGTH_PER_ID.get(i, marker_length_m))
        body_lines.append(
            f"{indent * (i + 1)}<geom name=\"{marker_name_prefix}_marker_{i:02d}\" type=\"box\" "
            f"size=\"{_fmt(0.5 * marker_length_i)} {_fmt(0.5 * marker_length_i)} {_fmt(0.5 * MARKER_THICKNESS_M)}\" "
            f"pos=\"{_fmt(marker_center_x)} {_fmt(marker_center_y)} {_fmt(marker_center_z)}\" "
            f"euler=\"{MARKER_EULER}\" material=\"{marker_name_prefix}_mat_{i:02d}\" "
            f"contype=\"0\" conaffinity=\"0\" mass=\"0\"/>"
        )
        body_lines.append(f"{indent * (i + 1)}<site name=\"site_L_{i:02d}\" pos=\"{site_pos_l}\"/>")
        body_lines.append(f"{indent * (i + 1)}<site name=\"site_R_{i:02d}\" pos=\"{site_pos_r}\"/>")

    for i in reversed(range(N_SEGMENTS)):
        body_lines.append(f"{indent * i}</body>")

    tendon_lines = []
    for i in range(N_SEGMENTS - 1):
        tendon_lines.append(f"    <spatial name=\"tendon_L_{i:02d}_{i+1:02d}\" width=\"0.001\">\n"
                            f"      <site site=\"site_L_{i:02d}\"/>\n"
                            f"      <site site=\"site_L_{i+1:02d}\"/>\n"
                            f"    </spatial>")
    for i in range(N_SEGMENTS - 1):
        tendon_lines.append(f"    <spatial name=\"tendon_R_{i:02d}_{i+1:02d}\" width=\"0.001\">\n"
                            f"      <site site=\"site_R_{i:02d}\"/>\n"
                            f"      <site site=\"site_R_{i+1:02d}\"/>\n"
                            f"    </spatial>")

    actuator_lines = []
    for i in range(N_SEGMENTS - 1):
        actuator_lines.append(
            f"    <motor name=\"motor_L_{i:02d}_{i+1:02d}\" tendon=\"tendon_L_{i:02d}_{i+1:02d}\" "
            f"ctrlrange=\"0 1\" ctrllimited=\"true\" gear=\"{_fmt(GEAR)}\"/>"
        )
    for i in range(N_SEGMENTS - 1):
        actuator_lines.append(
            f"    <motor name=\"motor_R_{i:02d}_{i+1:02d}\" tendon=\"tendon_R_{i:02d}_{i+1:02d}\" "
            f"ctrlrange=\"0 1\" ctrllimited=\"true\" gear=\"{_fmt(GEAR)}\"/>"
        )

    exclude_lines = []
    for i in range(N_SEGMENTS - 1):
        exclude_lines.append(f"  <exclude body1=\"seg_{i:02d}\" body2=\"seg_{i+1:02d}\"/>")
    exclude_block = "\n".join(exclude_lines)

    xml = f"""<mujoco model=\"spiral_tentacle_stl\">
  <compiler angle=\"radian\" coordinate=\"local\" inertiafromgeom=\"true\" meshdir=\"{meshdir}\"/>
  <option timestep=\"{_fmt(TIMESTEP)}\" gravity=\"0 0 -9.81\" integrator=\"implicitfast\" iterations=\"800\" ls_iterations=\"300\"/>
  <size njmax=\"6000\" nconmax=\"6000\"/>
  <visual>
    <global offwidth=\"{OFFSCREEN_WIDTH}\" offheight=\"{OFFSCREEN_HEIGHT}\"/>
  </visual>
  <contact>
{exclude_block}
  </contact>

  <default>
    <site size=\"0.001\" rgba=\"0.95 0.2 0.2 0.8\"/>
  </default>

  <asset>
    {''.join(mesh_lines)}
    {''.join(marker_asset_lines)}
  </asset>

  <worldbody>
    <light name=\"key\" pos=\"0.2 -0.3 0.4\" dir=\"-0.3 0.4 -0.8\" diffuse=\"0.7 0.7 0.7\"/>
    <camera name=\"{CAM_TOP_NAME}\" pos=\"{_fmt(CAM_TOP_POS[0])} {_fmt(CAM_TOP_POS[1])} {_fmt(CAM_TOP_POS[2])}\" xyaxes=\"{CAM_TOP_XYAXES}\" fovy=\"{_fmt(CAM_TOP_FOVY)}\"/>
    <geom name=\"floor\" type=\"plane\" pos=\"0 0 {_fmt(FLOOR_Z)}\" size=\"1 1 0.1\"
          friction=\"{FLOOR_FRICTION}\" solimp=\"{SOLIMP}\" solref=\"{SOLREF}\"
          contype=\"1\" conaffinity=\"1\" condim=\"6\" rgba=\"1 0.6 0.8 1\"/>
    <body name=\"ball\" pos=\"{_fmt(ball_x)} {_fmt(ball_y)} {_fmt(ball_z)}\">
      <joint name=\"ball_slide_x\" type=\"slide\" axis=\"1 0 0\" damping=\"0.25\" frictionloss=\"0.015\"/>
      <joint name=\"ball_slide_y\" type=\"slide\" axis=\"0 1 0\" damping=\"0.25\" frictionloss=\"0.015\"/>
      <joint name=\"ball_slide_z\" type=\"slide\" axis=\"0 0 1\" limited=\"true\" range=\"-0.003 0.020\"
             damping=\"0.60\" frictionloss=\"0.025\"/>
      <joint name=\"ball_rot\" type=\"ball\" damping=\"0.003\"/>
      <geom name=\"ball_geom\" type=\"sphere\" size=\"{_fmt(BALL_RADIUS)}\" density=\"{_fmt(BALL_DENSITY)}\"
            friction=\"{BALL_FRICTION}\" solimp=\"{SOLIMP}\" solref=\"{SOLREF}\"
            margin=\"0.0012\" gap=\"0.0005\"
            contype=\"1\" conaffinity=\"1\" condim=\"6\" rgba=\"0.9 0.2 0.2 1\"/>
    </body>

{chr(10).join(body_lines)}
  </worldbody>

  <tendon>
{chr(10).join(tendon_lines)}
  </tendon>

  <actuator>
{chr(10).join(actuator_lines)}
  </actuator>
</mujoco>
"""
    return xml


def _find_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    try:
        return mujoco.mj_name2id(model, obj_type, name)
    except Exception:
        return -1


def _tendon_force(data: mujoco.MjData, tendon_id: int, actuator_id: int) -> float:
    ten_force = getattr(data, "ten_force", None)
    if ten_force is not None:
        return float(ten_force[tendon_id])
    actuator_force = getattr(data, "actuator_force", None)
    if actuator_force is not None and actuator_id >= 0:
        return float(actuator_force[actuator_id])
    return float("nan")


def _cable_tensions(T0: float, q: list[float]) -> list[float]:
    T = float(T0)
    out = []
    for qi in q:
        out.append(T)
        att = math.exp(-(CABLE_BIAS + CABLE_MU * abs(qi)))
        T *= att
    return out


def _capstan_hysteresis(
    T0: float,
    q: list[float],
    T_prev: list[float],
    mu_eff: float,
    bias: float,
) -> list[float]:
    """
    Stateful capstan propagation with a static-friction band.
    Segment 0 is directly driven; capstan starts from segment 1.
    """
    n = N_SEGMENTS - 1
    if not T_prev or len(T_prev) != n:
        T_prev = [0.0] * n

    out = [0.0] * n
    out[0] = float(T0)
    T_in = out[0]

    for i in range(1, n):
        theta = abs(float(q[i - 1]))
        cap = math.exp(bias + mu_eff * theta)

        lo = T_in / cap
        hi = T_in * cap

        T_out = _clip(T_prev[i], lo, hi)

        out[i] = T_out
        T_in = T_out

    return out


def _alpha_tensions(T0: float, alpha: float) -> list[float]:
    return [float(T0) * math.pow(alpha, i) for i in range(N_SEGMENTS - 1)]


def compute_segment_tensions(
    ctrl: ForceController,
    q: list[float],
    dt_real: float,
) -> tuple[list[float], list[float]]:
    """
    Deterministic mapping from (TcmdL, TcmdR) to per-segment tensions.
    This is the only place that creates Tseg arrays.
    """
    dt = max(dt_real, 1e-6)
    dL_rate = abs(ctrl.T_left - ctrl.T_left_prev) / dt
    dR_rate = abs(ctrl.T_right - ctrl.T_right_prev) / dt
    uL = _clip((dL_rate - DTRATE_LO) / (DTRATE_HI - DTRATE_LO), 0.0, 1.0)
    uR = _clip((dR_rate - DTRATE_LO) / (DTRATE_HI - DTRATE_LO), 0.0, 1.0)
    muL = _lerp(ctrl.mu_static, ctrl.mu_kinetic, uL)
    muR = _lerp(ctrl.mu_static, ctrl.mu_kinetic, uR)

    t_left_cap = _capstan_hysteresis(
        ctrl.T_left,
        q,
        ctrl.T_left_seg,
        muL,
        ctrl.cable_bias,
    )
    t_right_cap = _capstan_hysteresis(
        ctrl.T_right,
        q,
        ctrl.T_right_seg,
        muR,
        ctrl.cable_bias,
    )

    alphaL = _lerp(ALPHA_DEFAULT, ALPHA_MAX, uL)
    alphaR = _lerp(ALPHA_DEFAULT, ALPHA_MAX, uR)
    t_left_alpha = _alpha_tensions(ctrl.T_left, alphaL)
    t_right_alpha = _alpha_tensions(ctrl.T_right, alphaR)
    boostL = TIP_BOOST_GAIN * _smoothstep(uL)
    boostR = TIP_BOOST_GAIN * _smoothstep(uR)
    ctrl.T_left_seg = [_lerp(c, a, boostL) for c, a in zip(t_left_cap, t_left_alpha)]
    ctrl.T_right_seg = [_lerp(c, a, boostR) for c, a in zip(t_right_cap, t_right_alpha)]

    return ctrl.T_left_seg, ctrl.T_right_seg


def apply_segment_tensions_to_motors(
    data: mujoco.MjData,
    motor_left_ids: list[int],
    motor_right_ids: list[int],
    t_left: list[float],
    t_right: list[float],
) -> None:
    for i, act_id in enumerate(motor_left_ids):
        data.ctrl[act_id] = _clip(t_left[i] / GEAR, 0.0, 1.0)
    for i, act_id in enumerate(motor_right_ids):
        data.ctrl[act_id] = _clip(t_right[i] / GEAR, 0.0, 1.0)


def _update_demo(ctrl: ForceController, tnow: float) -> None:
    if not ctrl.demo_active:
        return

    t = tnow - ctrl.demo_t0

    if ctrl.demo_stage == 0:
        if DEMO_T_RAMP <= 0.0:
            ctrl.T_left_target = DEMO_L_CURL
            ctrl.T_right_target = DEMO_R_CURL
            ctrl.demo_uR = 0.0
            ctrl.demo_uL = 0.0
            ctrl.demo_stage = 1
            ctrl.demo_t0 = tnow
            return
        u = _smoothstep(t / DEMO_T_RAMP)
        ctrl.T_left_target = _lerp(ctrl.demo_L0, DEMO_L_CURL, u)
        ctrl.T_right_target = _lerp(ctrl.demo_R0, DEMO_R_CURL, u)
        ctrl.demo_uR = 0.0
        ctrl.demo_uL = 0.0
        if t >= DEMO_T_RAMP:
            ctrl.demo_stage = 1
            ctrl.demo_t0 = tnow
        return

    if ctrl.demo_stage == 1:
        ctrl.T_left_target = DEMO_L_CURL
        ctrl.T_right_target = DEMO_R_CURL
        if t >= DEMO_T_SETTLE:
            ctrl.demo_stage = 2
            ctrl.demo_t0 = tnow
        return

    if ctrl.demo_stage == 2:
        u = _smoothstep(t / DEMO_T_CROSS)
        ctrl.demo_uR = u
        ctrl.demo_uL = u
        T_total = DEMO_L_CURL + DEMO_R_CURL
        ctrl.T_right_target = _clip(_lerp(DEMO_R_CURL, DEMO_R_END, u), 0.0, T_total)
        ctrl.T_left_target = T_total - ctrl.T_right_target
        if t >= DEMO_T_CROSS:
            ctrl.demo_stage = 3
            ctrl.demo_t0 = tnow
        return

    if ctrl.demo_stage == 3:
        T_total = DEMO_L_CURL + DEMO_R_CURL
        ctrl.T_right_target = _clip(DEMO_R_END, 0.0, T_total)
        ctrl.T_left_target = T_total - ctrl.T_right_target
        if t >= DEMO_T_HOLD:
            ctrl.demo_active = False
        return


def main() -> None:
    xml = build_mjcf()

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    jid = _find_id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_01")
    if jid >= 0:
        qadr = int(model.jnt_qposadr[jid])
        jrange = model.jnt_range[jid]
        print(f"joint_01 range: {jrange[0]:.6g} {jrange[1]:.6g}")
        print(f"joint_01 qpos: {data.qpos[qadr]:.6g}")
    else:
        print("joint_01 not found")

    motor_left_ids = []
    motor_right_ids = []
    tendon_left_ids = []
    tendon_right_ids = []
    joint_qposadrs = []

    for i in range(N_SEGMENTS - 1):
        name_l = f"motor_L_{i:02d}_{i+1:02d}"
        name_r = f"motor_R_{i:02d}_{i+1:02d}"
        idx_l = _find_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name_l)
        idx_r = _find_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name_r)
        if idx_l < 0 or idx_r < 0:
            raise RuntimeError("Missing actuator id for tendons")
        motor_left_ids.append(idx_l)
        motor_right_ids.append(idx_r)

        tname_l = f"tendon_L_{i:02d}_{i+1:02d}"
        tname_r = f"tendon_R_{i:02d}_{i+1:02d}"
        tid_l = _find_id(model, mujoco.mjtObj.mjOBJ_TENDON, tname_l)
        tid_r = _find_id(model, mujoco.mjtObj.mjOBJ_TENDON, tname_r)
        if tid_l < 0 or tid_r < 0:
            raise RuntimeError("Missing tendon id")
        tendon_left_ids.append(tid_l)
        tendon_right_ids.append(tid_r)

    for j in range(1, N_SEGMENTS):
        jname = f"joint_{j:02d}"
        jid = _find_id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise RuntimeError("Missing joint id")
        joint_qposadrs.append(int(model.jnt_qposadr[jid]))

    ball_x_jid = _find_id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_slide_x")
    ball_y_jid = _find_id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_slide_y")
    ball_z_jid = _find_id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_slide_z")
    if ball_x_jid < 0 or ball_y_jid < 0 or ball_z_jid < 0:
        raise RuntimeError("Missing ball slide joint id")
    ball_x_qadr = int(model.jnt_qposadr[ball_x_jid])
    ball_y_qadr = int(model.jnt_qposadr[ball_y_jid])
    ball_z_qadr = int(model.jnt_qposadr[ball_z_jid])
    ball_bid = _find_id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

    geom_seg_ids = []
    for i in range(N_SEGMENTS):
        gid = _find_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"geom_{i:02d}")
        if gid < 0:
            raise RuntimeError("Missing geom id")
        geom_seg_ids.append(gid)

    ball_gid = _find_id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")

    dof_ids = []
    joint_ids = []
    for j in range(1, N_SEGMENTS):
        jid = _find_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{j:02d}")
        if jid < 0:
            raise RuntimeError("Missing joint id")
        dof = int(model.jnt_dofadr[jid])
        dof_ids.append(dof)
        joint_ids.append(jid)

    base_geom_friction = model.geom_friction.copy()
    base_dof_damping = model.dof_damping.copy()
    base_dof_frictionloss = model.dof_frictionloss.copy()
    base_body_mass = model.body_mass.copy()
    base_body_inertia = model.body_inertia.copy()
    has_dof_spring = hasattr(model, "dof_spring")
    has_jnt_stiffness = hasattr(model, "jnt_stiffness")
    base_dof_spring = model.dof_spring.copy() if has_dof_spring else None
    base_jnt_stiffness = model.jnt_stiffness.copy() if has_jnt_stiffness else None

    dr_cfg = DomainRandCfg(enabled=True, seed=123)
    dr_rng = np.random.default_rng(dr_cfg.seed)
    spawn_rng = np.random.default_rng(dr_cfg.seed + 1)

    def apply_domain_randomization(
        model: mujoco.MjModel,
        ctrl: ForceController,
        rng: np.random.Generator,
        cfg: DomainRandCfg,
    ) -> None:
        if not cfg.enabled:
            return

        for gid in geom_seg_ids:
            f = base_geom_friction[gid].copy()
            f[0] *= _logu(rng, *cfg.seg_fric_slide)
            f[1] *= _logu(rng, *cfg.seg_fric_tors)
            f[2] *= _logu(rng, *cfg.seg_fric_roll)
            model.geom_friction[gid] = f

        if ball_gid >= 0:
            f = base_geom_friction[ball_gid].copy()
            f[0] *= _logu(rng, *cfg.ball_fric_slide)
            f[1] *= _logu(rng, *cfg.ball_fric_tors)
            f[2] *= _logu(rng, *cfg.ball_fric_roll)
            f[0] = float(np.clip(f[0], 0.05, 0.20))
            f[1] = float(np.clip(f[1], 0.0004, 0.0025))
            f[2] = float(np.clip(f[2], 0.0002, 0.0012))
            model.geom_friction[ball_gid] = f

        for dof in dof_ids:
            model.dof_damping[dof] = base_dof_damping[dof] * _logu(
                rng,
                *cfg.dof_damping,
            )
            model.dof_frictionloss[dof] = base_dof_frictionloss[dof] * _logu(
                rng,
                *cfg.dof_frictionloss,
            )

        if has_jnt_stiffness or has_dof_spring:
            for jid, dof in zip(joint_ids, dof_ids):
                scale = _logu(rng, *cfg.dof_spring)
                if has_jnt_stiffness:
                    model.jnt_stiffness[jid] = base_jnt_stiffness[jid] * scale
                if has_dof_spring:
                    model.dof_spring[dof] = base_dof_spring[dof] * scale

        mscale = _logu(rng, *cfg.mass_scale)
        for b in range(model.nbody):
            if b == 0 or b == ball_bid:
                continue
            jitter = 1.0 + _u(
                rng,
                -cfg.per_link_mass_jitter,
                cfg.per_link_mass_jitter,
            )
            s = mscale * jitter
            model.body_mass[b] = base_body_mass[b] * s
            model.body_inertia[b] = base_body_inertia[b] * s

        ctrl.mu_static = _u(rng, *cfg.mu_static)
        ctrl.mu_kinetic = _u(rng, *cfg.mu_kinetic)
        ctrl.cable_bias = _u(rng, *cfg.cable_bias)

        if cfg.log_on_reset:
            seg0_fric = model.geom_friction[geom_seg_ids[0]]
            segN_fric = model.geom_friction[geom_seg_ids[-1]]
            dof0 = dof_ids[0]
            dofN = dof_ids[-1]
            print(
                "DR reset | "
                f"mu_static={ctrl.mu_static:.4g} "
                f"mu_kinetic={ctrl.mu_kinetic:.4g} "
                f"cable_bias={ctrl.cable_bias:.4g} "
                f"mscale={mscale:.4g} | "
                f"seg0_fric=({seg0_fric[0]:.4g},{seg0_fric[1]:.4g},{seg0_fric[2]:.4g}) "
                f"segN_fric=({segN_fric[0]:.4g},{segN_fric[1]:.4g},{segN_fric[2]:.4g}) | "
                f"dof_fricloss0={model.dof_frictionloss[dof0]:.4g} "
                f"dof_friclossN={model.dof_frictionloss[dofN]:.4g}"
            )

    ctrl = ForceController()
    ctrl.T_left_seg = [0.0] * (N_SEGMENTS - 1)
    ctrl.T_right_seg = [0.0] * (N_SEGMENTS - 1)
    ctrl.T_left_prev = ctrl.T_left
    ctrl.T_right_prev = ctrl.T_right

    reset_requested = False

    def _reset_sim() -> None:
        mujoco.mj_resetData(model, data)
        apply_domain_randomization(model, ctrl, dr_rng, dr_cfg)
        ball_x, ball_y = _sample_ball_xy(BALL_SPAWN_RADIUS, BALL_MIN_Y_CLEAR, spawn_rng)
        data.qpos[ball_x_qadr] = ball_x - BALL_BASE_X
        data.qpos[ball_y_qadr] = ball_y - BALL_BASE_Y
        data.qpos[ball_z_qadr] = 0.0
        mujoco.mj_forward(model, data)
        ctrl.demo_active = False
        ctrl.demo_stage = 0
        ctrl.demo_t0 = time.time()
        ctrl.demo_L0 = 0.0
        ctrl.demo_R0 = 0.0
        ctrl.demo_uR = 0.0
        ctrl.demo_uL = 0.0
        ctrl.T_left = 0.0
        ctrl.T_right = 0.0
        ctrl.T_left_target = 0.0
        ctrl.T_right_target = 0.0
        ctrl.T_left_seg = [0.0] * (N_SEGMENTS - 1)
        ctrl.T_right_seg = [0.0] * (N_SEGMENTS - 1)
        ctrl.T_left_prev = 0.0
        ctrl.T_right_prev = 0.0
        data.ctrl[:] = 0.0

    def on_press(key) -> None:
        nonlocal reset_requested
        try:
            k = key.char
        except AttributeError:
            k = None

        if key == keyboard.Key.space:
            ctrl.demo_active = False
            ctrl.T_left = 0.0
            ctrl.T_right = 0.0
            ctrl.T_left_target = 0.0
            ctrl.T_right_target = 0.0
            return

        if k is None:
            if key == keyboard.Key.esc:
                ctrl.running = False
            return

        if k in ("q", "й"):
            ctrl.running = False
        elif k in ("r", "к"):
            reset_requested = True
        elif k in ("a", "ф"):
            ctrl.demo_active = False
            ctrl.T_left = _clip(ctrl.T_left + ctrl.T_step, 0.0, ctrl.Tmax)
            ctrl.T_left_target = ctrl.T_left
        elif k in ("z", "я"):
            ctrl.demo_active = False
            ctrl.T_left = _clip(ctrl.T_left - ctrl.T_step, 0.0, ctrl.Tmax)
            ctrl.T_left_target = ctrl.T_left
        elif k in ("k", "л"):
            ctrl.demo_active = False
            ctrl.T_right = _clip(ctrl.T_right + ctrl.T_step, 0.0, ctrl.Tmax)
            ctrl.T_right_target = ctrl.T_right
        elif k in ("m", "ь"):
            ctrl.demo_active = False
            ctrl.T_right = _clip(ctrl.T_right - ctrl.T_step, 0.0, ctrl.Tmax)
            ctrl.T_right_target = ctrl.T_right
        elif k in ("o", "щ"):
            ctrl.T_step = max(0.5, ctrl.T_step * 0.8)
        elif k in ("p", "з"):
            ctrl.T_step = min(20.0, ctrl.T_step * 1.25)
        elif k == "1":
            ctrl.demo_active = False
            ctrl.T_left, ctrl.T_right = 100.0, 0.0
            ctrl.T_left_target, ctrl.T_right_target = ctrl.T_left, ctrl.T_right
        elif k == "2":
            ctrl.demo_active = False
            ctrl.T_left, ctrl.T_right = 0.0, 100.0
            ctrl.T_left_target, ctrl.T_right_target = ctrl.T_left, ctrl.T_right
        elif k == "3":
            ctrl.demo_active = False
            ctrl.T_left, ctrl.T_right = 30.0, 60.0
            ctrl.T_left_target, ctrl.T_right_target = ctrl.T_left, ctrl.T_right
        elif k == "4":
            ctrl.demo_active = False
            ctrl.T_left, ctrl.T_right = 30.0, 10.0
            ctrl.T_left_target, ctrl.T_right_target = ctrl.T_left, ctrl.T_right
        elif k == "5":
            ctrl.demo_active = True
            ctrl.demo_stage = 0
            ctrl.demo_t0 = time.time()
            ctrl.demo_L0 = float(ctrl.T_left)
            ctrl.demo_R0 = float(ctrl.T_right)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    _reset_sim()

    t_last = time.time()
    print_dt = 0.1
    t_prev = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0

        while viewer.is_running() and ctrl.running:
            step_start = time.time()
            tnow = time.time()
            dt_real = _clip(tnow - t_prev, 1e-6, 0.05)
            t_prev = tnow

            if reset_requested:
                _reset_sim()
                reset_requested = False
                continue

            if ctrl.demo_active:
                prev_stage = ctrl.demo_stage
                _update_demo(ctrl, tnow)
                if prev_stage == 0 and ctrl.demo_stage != 0:
                    ctrl.T_left = _clip(ctrl.T_left_target, 0.0, ctrl.Tmax)
                    ctrl.T_right = _clip(ctrl.T_right_target, 0.0, ctrl.Tmax)
                else:
                    slew = DT_SLEW_RAMP if ctrl.demo_stage in (0, 1) else DT_SLEW_CROSS
                    max_dT = slew * dt_real
                    dL = _clip(ctrl.T_left_target - ctrl.T_left, -max_dT, max_dT)
                    dR = _clip(ctrl.T_right_target - ctrl.T_right, -max_dT, max_dT)
                    ctrl.T_left = _clip(ctrl.T_left + dL, 0.0, ctrl.Tmax)
                    ctrl.T_right = _clip(ctrl.T_right + dR, 0.0, ctrl.Tmax)
            else:
                ctrl.T_left_target = ctrl.T_left
                ctrl.T_right_target = ctrl.T_right

            q = [float(data.qpos[adr]) for adr in joint_qposadrs]
            t_left, t_right = compute_segment_tensions(ctrl, q, dt_real)
            q01 = float(data.qpos[joint_qposadrs[0]])
            q05 = float(data.qpos[joint_qposadrs[4]])
            q18 = float(data.qpos[joint_qposadrs[17]])
            tseg0_l = t_left[0] if t_left else 0.0
            tseg0_r = t_right[0] if t_right else 0.0
            tsegN_l = t_left[-1] if t_left else 0.0
            tsegN_r = t_right[-1] if t_right else 0.0
            ctrl0_l = _clip(tseg0_l / GEAR, 0.0, 1.0)
            ctrl0_r = _clip(tseg0_r / GEAR, 0.0, 1.0)
            ctrlN_l = _clip(tsegN_l / GEAR, 0.0, 1.0)
            ctrlN_r = _clip(tsegN_r / GEAR, 0.0, 1.0)

            apply_segment_tensions_to_motors(
                data,
                motor_left_ids,
                motor_right_ids,
                t_left,
                t_right,
            )

            ctrl.T_left_prev = ctrl.T_left
            ctrl.T_right_prev = ctrl.T_right

            mujoco.mj_step(model, data)

            t = time.time()
            if (t - t_last) >= print_dt:
                t_last = t

                ten_l0 = _tendon_force(data, tendon_left_ids[0], motor_left_ids[0])
                ten_ln = _tendon_force(data, tendon_left_ids[-1], motor_left_ids[-1])
                ten_r0 = _tendon_force(data, tendon_right_ids[0], motor_right_ids[0])
                ten_rn = _tendon_force(data, tendon_right_ids[-1], motor_right_ids[-1])

                print(
                    f"Tcmd(L,R)=({ctrl.T_left:6.1f},{ctrl.T_right:6.1f}) N | "
                    f"Tseg0(L,R)=({tseg0_l:6.1f},{tseg0_r:6.1f}) | "
                    f"TsegN(L,R)=({tsegN_l:6.1f},{tsegN_r:6.1f}) | "
                    f"ctrl0(L,R)=({ctrl0_l:.3f},{ctrl0_r:.3f}) | "
                    f"ctrlN(L,R)=({ctrlN_l:.3f},{ctrlN_r:.3f}) | "
                    f"tenF(L0,LN,R0,RN)=({ten_l0:6.1f},{ten_ln:6.1f},{ten_r0:6.1f},{ten_rn:6.1f}) | "
                    f"q01={q01:+.4f} q05={q05:+.4f} q18={q18:+.4f} | "
                    f"T_step={ctrl.T_step:.2f} | stage={ctrl.demo_stage} "
                    f"uR={ctrl.demo_uR:.2f} uL={ctrl.demo_uL:.2f}"
                )

            viewer.sync()

            elapsed = time.time() - step_start
            sleep = model.opt.timestep - elapsed
            if sleep > 0:
                time.sleep(sleep)

    ctrl.running = False
    listener.stop()
    listener.join(timeout=1.0)


if __name__ == "__main__":
    main()
