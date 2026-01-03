from __future__ import annotations

import math
import time
from dataclasses import dataclass
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

T_STEP_DEFAULT = 5
T_MAX = 100.0
CABLE_MU = 0.18
CABLE_BIAS = 0.006
USE_CAPSTAN_DEMO_ONLY = True

TIMESTEP = 0.0005

FRICTION = "1.6 0.015 0.00005"
SOLIMP = "0.998 0.998 0.0002"
SOLREF = "0.003 1.0"
MARGIN = 0.00025
GAP = 0.0000
DENSITY = 1400.0
GEOM_EULER = "0 0 0"
CLEARANCE_X = 0.0006
SPRING_STIFFNESS = 0.06
SITE_X_FRAC = 0.15
SITE_X_MIN = 0.0006
PIVOT_X = 0.0010
DEMO_L_CURL = 90.0
DEMO_R_CURL = 0.0
DEMO_L_END = 40.0
DEMO_R_END = 90.0
DEMO_T_RAMP = 1.2
DEMO_T_SETTLE = 0.6
DEMO_T_CROSS = 4.0
DEMO_T_HOLD = 1.0
DT_SLEW = 220.0


def _fmt(x: float) -> str:
    return f"{x:.10g}"


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def _smoothstep(u: float) -> float:
    u = _clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


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


def build_mjcf() -> str:
    if trimesh is None:
        raise SystemExit("ERROR: trimesh is not installed. Install with: pip install trimesh")

    base_dir = Path(__file__).resolve().parent
    stl_dir = base_dir / "assets" / "spiral_tent_stls_vir"
    if not stl_dir.is_dir():
        raise SystemExit(f"ERROR: STL directory not found: {stl_dir}")

    stl_paths = [stl_dir / f"seg_{i:02d}.stl" for i in range(N_SEGMENTS)]
    for path in stl_paths:
        if not path.exists():
            raise SystemExit(f"ERROR: Missing STL file: {path}")

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

    base_width = seg_info[0]["yspan"] * scale_factor
    tip_width = seg_info[-1]["yspan"] * scale_factor

    print("=== STL SCALE SUMMARY ===")
    print(f"measured_length   = {measured_length:.6f} m")
    print(f"scale_factor      = {scale_factor:.8f}")
    print(f"scaled_length     = {scaled_length:.6f} m")
    print(f"base_width_scaled = {base_width:.6f} m")
    print(f"tip_width_scaled  = {tip_width:.6f} m")

    meshdir = stl_dir.as_posix()

    mesh_lines = []
    for i in range(N_SEGMENTS):
        mesh_lines.append(
            f'<mesh name="mesh_{i:02d}" file="seg_{i:02d}.stl" '
            f'scale="{_fmt(scale_factor)} {_fmt(scale_factor)} {_fmt(scale_factor)}"/>'
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

        xlen_s = info["xspan_eff"] * scale_factor
        x_hinge_span = max(1e-6, xlen_s - 2.0 * PIVOT_X)
        site_x = max(SITE_X_MIN, SITE_X_FRAC * x_hinge_span)
        y_site = 0.5 * yspan - 0.002
        y_site = max(y_site, 0.0022)

        site_pos_l = f"{_fmt(site_x)} {_fmt(y_site)} 0"
        site_pos_r = f"{_fmt(site_x)} {_fmt(-y_site)} 0"

        if i == 0:
            body_pos = "0 0 0"
        else:
            prev_xlen = seg_info[i - 1]["xspan_eff"] * scale_factor
            pitch = prev_xlen + CLEARANCE_X
            body_pos = f"{_fmt(pitch)} 0 0"

        body_lines.append(f"{indent * i}<body name=\"seg_{i:02d}\" pos=\"{body_pos}\">")
        if i > 0:
            s = i / (N_SEGMENTS - 1)
            fric = 0.002 + 0.006 * (s ** 2)
            damp = 0.006 + 0.014 * (s ** 2)
            body_lines.append(
                f"{indent * (i + 1)}<joint name=\"joint_{i:02d}\" type=\"hinge\" "
                f"axis=\"0 0 1\" limited=\"true\" range=\"-0.52 0.52\" "
                f"solreflimit=\"0.0005 1\" solimplimit=\"0.95 0.99 0.001\" "
                f"stiffness=\"{_fmt(SPRING_STIFFNESS)}\" springref=\"0\" "
                f"damping=\"{_fmt(damp)}\" frictionloss=\"{_fmt(fric)}\"/>"
            )

        body_lines.append(
            f"{indent * (i + 1)}<geom name=\"geom_{i:02d}\" type=\"mesh\" "
            f"mesh=\"mesh_{i:02d}\" pos=\"{_fmt(geom_pos[0])} {_fmt(geom_pos[1])} {_fmt(geom_pos[2])}\" "
            f"euler=\"{GEOM_EULER}\" "
            f"density=\"{_fmt(DENSITY)}\" rgba=\"0.85 0.86 0.9 1\" "
            f"condim=\"6\" contype=\"1\" conaffinity=\"1\" "
            f"friction=\"{FRICTION}\" solimp=\"{SOLIMP}\" solref=\"{SOLREF}\" "
            f"margin=\"{_fmt(MARGIN)}\" gap=\"{_fmt(GAP)}\"/>"
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

    xml = f"""<mujoco model=\"spiral_tentacle_stl\">
  <compiler angle=\"radian\" coordinate=\"local\" inertiafromgeom=\"true\" meshdir=\"{meshdir}\"/>
  <option timestep=\"{_fmt(TIMESTEP)}\" gravity=\"0 0 -9.81\" integrator=\"implicitfast\" iterations=\"800\" ls_iterations=\"300\"/>
  <size njmax=\"20000\" nconmax=\"20000\"/>

  <default>
    <site size=\"0.001\" rgba=\"0.95 0.2 0.2 0.8\"/>
  </default>

  <asset>
    {''.join(mesh_lines)}
  </asset>

  <worldbody>
    <light name=\"key\" pos=\"0.2 -0.3 0.4\" dir=\"-0.3 0.4 -0.8\" diffuse=\"0.7 0.7 0.7\"/>
    <geom name=\"floor\" type=\"plane\" pos=\"0 0 -0.05\" size=\"1 1 0.1\"
          friction=\"{FRICTION}\" solimp=\"{SOLIMP}\" solref=\"{SOLREF}\"
          contype=\"1\" conaffinity=\"1\" condim=\"3\" rgba=\"0.2 0.2 0.2 1\"/>

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


def _alpha_tensions(T0: float, alpha: float) -> list[float]:
    return [float(T0) * math.pow(alpha, i) for i in range(N_SEGMENTS - 1)]


def _update_demo(ctrl: ForceController, tnow: float) -> None:
    if not ctrl.demo_active:
        return

    t = tnow - ctrl.demo_t0

    if ctrl.demo_stage == 0:
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
        uR = _smoothstep(t / DEMO_T_CROSS)
        tL = max(0.0, t - 0.60 * DEMO_T_CROSS)
        uL = _smoothstep(tL / max(1e-6, 0.40 * DEMO_T_CROSS))
        ctrl.demo_uR = uR
        ctrl.demo_uL = uL
        ctrl.T_right_target = _lerp(DEMO_R_CURL, DEMO_R_END, uR)
        ctrl.T_left_target = _lerp(DEMO_L_CURL, DEMO_L_END, uL)
        if t >= DEMO_T_CROSS:
            ctrl.demo_stage = 3
            ctrl.demo_t0 = tnow
        return

    if ctrl.demo_stage == 3:
        ctrl.T_left_target = DEMO_L_END
        ctrl.T_right_target = DEMO_R_END
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

    ctrl = ForceController()

    def on_press(key) -> None:
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
        elif k in ("[", "х"):
            ctrl.T_step = max(0.5, ctrl.T_step * 0.8)
        elif k in ("]", "ъ"):
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
        elif k in (",", "б"):
            ctrl.alpha = _clip(ctrl.alpha - ALPHA_STEP, ALPHA_MIN, ALPHA_MAX)
        elif k in (".", "ю"):
            ctrl.alpha = _clip(ctrl.alpha + ALPHA_STEP, ALPHA_MIN, ALPHA_MAX)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    t_last = time.time()
    print_dt = 0.1

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0

        while viewer.is_running() and ctrl.running:
            step_start = time.time()

            if ctrl.demo_active:
                tnow = time.time()
                _update_demo(ctrl, tnow)
                dt = model.opt.timestep
                max_dT = DT_SLEW * dt
                dL = _clip(ctrl.T_left_target - ctrl.T_left, -max_dT, max_dT)
                dR = _clip(ctrl.T_right_target - ctrl.T_right, -max_dT, max_dT)
                ctrl.T_left = _clip(ctrl.T_left + dL, 0.0, ctrl.Tmax)
                ctrl.T_right = _clip(ctrl.T_right + dR, 0.0, ctrl.Tmax)
            else:
                ctrl.T_left_target = ctrl.T_left
                ctrl.T_right_target = ctrl.T_right

            use_capstan = ctrl.demo_active or not USE_CAPSTAN_DEMO_ONLY
            if use_capstan:
                q = [float(data.qpos[adr]) for adr in joint_qposadrs]
                t_left = _cable_tensions(ctrl.T_left, q)
                t_right = _cable_tensions(ctrl.T_right, q)
            else:
                t_left = _alpha_tensions(ctrl.T_left, ctrl.alpha)
                t_right = _alpha_tensions(ctrl.T_right, ctrl.alpha)
            tseg0_l = t_left[0] if t_left else 0.0
            tseg0_r = t_right[0] if t_right else 0.0
            tsegN_l = t_left[-1] if t_left else 0.0
            tsegN_r = t_right[-1] if t_right else 0.0
            ctrl0_l = _clip(tseg0_l / GEAR, 0.0, 1.0)
            ctrl0_r = _clip(tseg0_r / GEAR, 0.0, 1.0)
            ctrlN_l = _clip(tsegN_l / GEAR, 0.0, 1.0)
            ctrlN_r = _clip(tsegN_r / GEAR, 0.0, 1.0)

            for i, act_id in enumerate(motor_left_ids):
                tseg = t_left[i]
                data.ctrl[act_id] = _clip(tseg / GEAR, 0.0, 1.0)
            for i, act_id in enumerate(motor_right_ids):
                tseg = t_right[i]
                data.ctrl[act_id] = _clip(tseg / GEAR, 0.0, 1.0)

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
