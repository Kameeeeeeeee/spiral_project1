# debug_viewer.py
#
# Viewer + ручное управление катушками (spool) для двух тросов.
#
# Все актуальные изменения:
# - Печать статусов моторов/тросов/катушек (10 Гц).
# - Lmax_frac увеличен до 0.65, чтобы дать больше укорочения троса и сильнее сворачивание.
# - Остальная физика: распределенные потери по тросу (stick-slip), физичная внешняя коррекция
#   (клип |tau_corr|, запрет генерации энергии tau*v>0, шум и домен-рандомизация только в этом блоке).
#
# Управление:
# a/z: left spool speed +/-
# k/m: right spool speed +/-
# space: stop
# [ ]: change vel_step
# - =: change tau_v
# r: reset
# q: quit

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import mujoco
import mujoco.viewer
from pynput import keyboard  # pip install pynput

from generate_spiral_xml import generate_spiral_tentacle_xml


@dataclass
class SpoolController:
    vcmd_left: float = 0.0
    vcmd_right: float = 0.0

    Lspool_left: float = 0.0
    Lspool_right: float = 0.0

    v_max: float = 0.085

    # Увеличили, чтобы дать больше максимального укорочения троса.
    Lmax_frac: float = 0.65
    Lmax_left: float = 0.0
    Lmax_right: float = 0.0

    l_rest_left: float = 0.0
    l_rest_right: float = 0.0

    # PD по длине -> натяжение (tension-only).
    kp: float = 6000.0
    kd: float = 150.0

    tau_v: float = 0.05
    v_left: float = 0.0
    v_right: float = 0.0

    # Параметры распределенного трения (stick-slip).
    mu_s: float = 0.18
    mu_k: float = 0.14
    N0: float = 0.15
    k_theta: float = 0.9
    k_c: float = 0.30
    c_v: float = 0.22

    # Релаксация натяжения.
    tau_T: float = 0.25

    T_left: list[float] | None = None
    T_right: list[float] | None = None

    vel_step: float = 0.12
    running: bool = True


@dataclass
class CorrectionParams:
    tau_min: float = 0.001
    tau_cap: float = 0.45
    tau_bias: float = 0.0
    T_bias: float = 0.05

    noise_std: float = 0.002

    rand_enable: bool = True
    rand_period_s: float = 3.0

    tau_gain: list[float] | None = None
    c_corr: list[float] | None = None


def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _clip01(x: float) -> float:
    return _clip(x, 0.0, 1.0)


def _sgn(x: float) -> float:
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


def _read_tendon_length(data: mujoco.MjData, tendon_id: int) -> float:
    return float(data.ten_length[tendon_id])


def _read_tendon_velocity(data: mujoco.MjData, tendon_id: int) -> float:
    return float(data.ten_velocity[tendon_id])


def propagate_tension(
    ctrl: SpoolController,
    T_base: float,
    v_spool: float,
    thetas: list[float],
    C: list[float],
    T_prev: list[float],
    dt: float,
) -> list[float]:
    # Дискретная модель T(s) с stick-slip и релаксацией, без генерации "ложной" энергии.
    T = T_prev[:]
    T[0] = max(0.0, T_base)

    a_relax = 1.0 - math.exp(-dt / max(1e-6, ctrl.tau_T))
    sgnv = _sgn(v_spool)
    vabs = abs(v_spool)

    for i in range(len(T) - 1):
        Ni = ctrl.N0 + ctrl.k_theta * abs(thetas[i]) + ctrl.k_c * C[i]
        Fs = ctrl.mu_s * Ni
        Fk = ctrl.mu_k * Ni

        Ti = max(0.0, T[i])
        Tn = max(0.0, T[i + 1])
        d = Ti - Tn

        if vabs < 1e-6:
            if abs(d) <= Fs:
                T[i + 1] = max(0.0, Tn + a_relax * (Ti - Tn))
            else:
                target = Ti - math.copysign(Fs, d)
                T[i + 1] = max(0.0, Tn + a_relax * (target - Tn))
            continue

        if abs(d) <= Fs:
            T[i + 1] = max(0.0, Tn + a_relax * (Ti - Tn))
        else:
            drop = Fk + ctrl.c_v * vabs
            T_slip = Ti - sgnv * drop
            if T_slip < 0.0:
                T_slip = 0.0

            if sgnv < 0.0 and T_slip > Tn:
                T[i + 1] = max(0.0, Tn + a_relax * (T_slip - Tn))
            else:
                T[i + 1] = T_slip

    return T


def main() -> None:
    xml = generate_spiral_tentacle_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    id_motor_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_left")
    id_motor_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_right")
    if id_motor_left < 0 or id_motor_right < 0:
        raise RuntimeError("Actuators motor_left and motor_right must exist in the model.")

    id_tendon_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "tendon_left")
    id_tendon_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "tendon_right")
    if id_tendon_left < 0 or id_tendon_right < 0:
        raise RuntimeError("Tendons tendon_left and tendon_right must exist in the model.")

    joint_ids: list[int] = []
    dof_adrs: list[int] = []
    for i in range(1, 10_000):
        name = f"hinge_{i:02d}"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            break
        joint_ids.append(jid)
        dof_adrs.append(int(model.jnt_dofadr[jid]))

    n_joints = len(joint_ids)
    if n_joints <= 0:
        raise RuntimeError("No hinge joints were found (expected hinge_01..).")

    moment_arm: list[float] = []
    geom_ids: list[int] = []
    for i in range(n_joints):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"site_left_{i:02d}")
        if sid < 0:
            raise RuntimeError("Missing tendon sites site_left_XX.")
        moment_arm.append(abs(float(model.site_pos[sid][1])))

        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"geom_{i:02d}")
        if gid < 0:
            raise RuntimeError("Missing segment geom geom_XX.")
        geom_ids.append(gid)

    geom_to_seg = {gid: i for i, gid in enumerate(geom_ids)}
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    ctrl = SpoolController()
    corr = CorrectionParams()

    rng = np.random.default_rng(12345)
    corr.tau_gain = [0.22] * n_joints
    corr.c_corr = [0.06] * n_joints

    last_rand_t = 0.0

    def reroll_correction_params() -> None:
        if not corr.rand_enable:
            return
        for i in range(n_joints):
            corr.tau_gain[i] = float(rng.uniform(0.16, 0.30))
            corr.c_corr[i] = float(rng.uniform(0.03, 0.10))
        corr.noise_std = float(rng.uniform(0.001, 0.006))

    def reset_spool_reference() -> None:
        mujoco.mj_forward(model, data)
        ctrl.Lspool_left = 0.0
        ctrl.Lspool_right = 0.0
        ctrl.vcmd_left = 0.0
        ctrl.vcmd_right = 0.0
        ctrl.v_left = 0.0
        ctrl.v_right = 0.0
        ctrl.l_rest_left = _read_tendon_length(data, id_tendon_left)
        ctrl.l_rest_right = _read_tendon_length(data, id_tendon_right)
        ctrl.Lmax_left = ctrl.Lmax_frac * ctrl.l_rest_left
        ctrl.Lmax_right = ctrl.Lmax_frac * ctrl.l_rest_right
        ctrl.T_left = [0.0] * n_joints
        ctrl.T_right = [0.0] * n_joints

    reset_spool_reference()
    reroll_correction_params()

    def on_press(key) -> None:
        try:
            ch = key.char
        except AttributeError:
            ch = None

        if ch == "a":
            ctrl.vcmd_left = _clip(ctrl.vcmd_left + ctrl.vel_step, -1.0, 1.0)
        elif ch == "z":
            ctrl.vcmd_left = _clip(ctrl.vcmd_left - ctrl.vel_step, -1.0, 1.0)
        elif ch == "k":
            ctrl.vcmd_right = _clip(ctrl.vcmd_right + ctrl.vel_step, -1.0, 1.0)
        elif ch == "m":
            ctrl.vcmd_right = _clip(ctrl.vcmd_right - ctrl.vel_step, -1.0, 1.0)
        elif ch == " ":
            ctrl.vcmd_left = 0.0
            ctrl.vcmd_right = 0.0
        elif ch == "[":
            ctrl.vel_step = max(0.02, ctrl.vel_step * 0.8)
        elif ch == "]":
            ctrl.vel_step = min(0.5, ctrl.vel_step * 1.25)
        elif ch == "-":
            ctrl.tau_v = min(0.25, ctrl.tau_v * 1.25)
        elif ch == "=":
            ctrl.tau_v = max(0.01, ctrl.tau_v * 0.8)
        elif ch == "r":
            mujoco.mj_resetData(model, data)
            reset_spool_reference()
            reroll_correction_params()
        elif ch == "q":
            ctrl.running = False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    dt = float(model.opt.timestep)
    steps_per_frame = 6

    gear_left = float(model.actuator_gear[id_motor_left][0])
    gear_right = float(model.actuator_gear[id_motor_right][0])
    Tmax_left = gear_left * 1.0
    Tmax_right = gear_right * 1.0

    def compute_contact_flags() -> list[float]:
        flags = [0.0] * n_joints
        for k in range(int(data.ncon)):
            c = data.contact[k]
            g1 = int(c.geom1)
            g2 = int(c.geom2)

            if g1 == floor_gid and g2 in geom_to_seg:
                flags[geom_to_seg[g2]] = 1.0
            elif g2 == floor_gid and g1 in geom_to_seg:
                flags[geom_to_seg[g1]] = 1.0
            else:
                if g1 in geom_to_seg:
                    flags[geom_to_seg[g1]] = 1.0
                if g2 in geom_to_seg:
                    flags[geom_to_seg[g2]] = 1.0
        return flags

    # Печать статуса.
    print_hz = 10.0
    next_print_t = 0.0
    last_Tbase_left = 0.0
    last_Tbase_right = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and ctrl.running:
            sim_t = float(data.time)
            if corr.rand_enable and (sim_t - last_rand_t) >= corr.rand_period_s:
                reroll_correction_params()
                last_rand_t = sim_t

            for _ in range(steps_per_frame):
                alpha_v = 1.0 - math.exp(-dt / max(1e-6, ctrl.tau_v))
                vdes_left = ctrl.vcmd_left * ctrl.v_max
                vdes_right = ctrl.vcmd_right * ctrl.v_max
                ctrl.v_left += alpha_v * (vdes_left - ctrl.v_left)
                ctrl.v_right += alpha_v * (vdes_right - ctrl.v_right)

                ctrl.Lspool_left = _clip(ctrl.Lspool_left + ctrl.v_left * dt, 0.0, ctrl.Lmax_left)
                ctrl.Lspool_right = _clip(ctrl.Lspool_right + ctrl.v_right * dt, 0.0, ctrl.Lmax_right)

                ldes_left = ctrl.l_rest_left - ctrl.Lspool_left
                ldes_right = ctrl.l_rest_right - ctrl.Lspool_right
                ldes_dot_left = -ctrl.v_left
                ldes_dot_right = -ctrl.v_right

                l_now_left = _read_tendon_length(data, id_tendon_left)
                l_now_right = _read_tendon_length(data, id_tendon_right)
                ldot_left = _read_tendon_velocity(data, id_tendon_left)
                ldot_right = _read_tendon_velocity(data, id_tendon_right)

                e_left = l_now_left - ldes_left
                e_right = l_now_right - ldes_right
                de_left = ldot_left - ldes_dot_left
                de_right = ldot_right - ldes_dot_right

                Tbase_left = _clip(ctrl.kp * e_left + ctrl.kd * de_left, 0.0, Tmax_left)
                Tbase_right = _clip(ctrl.kp * e_right + ctrl.kd * de_right, 0.0, Tmax_right)
                last_Tbase_left = Tbase_left
                last_Tbase_right = Tbase_right

                data.ctrl[id_motor_left] = _clip01(Tbase_left / max(1e-9, gear_left))
                data.ctrl[id_motor_right] = _clip01(Tbase_right / max(1e-9, gear_right))

                mujoco.mj_forward(model, data)
                C = compute_contact_flags()
                thetas = [float(data.qpos[dof_adrs[i]]) for i in range(n_joints)]

                if ctrl.T_left is None or ctrl.T_right is None:
                    ctrl.T_left = [0.0] * n_joints
                    ctrl.T_right = [0.0] * n_joints

                T_left = propagate_tension(ctrl, Tbase_left, ctrl.v_left, thetas, C, ctrl.T_left, dt)
                T_right = propagate_tension(ctrl, Tbase_right, ctrl.v_right, thetas, C, ctrl.T_right, dt)
                ctrl.T_left = T_left
                ctrl.T_right = T_right

                data.qfrc_applied[:] = 0.0
                qvel = data.qvel
                Tsum_base = Tbase_left + Tbase_right

                for i in range(n_joints):
                    adr = dof_adrs[i]
                    d = moment_arm[i]

                    tau_raw = ((T_left[i] - Tbase_left) - (T_right[i] - Tbase_right)) * d
                    v = float(qvel[adr])
                    tau = tau_raw - float(corr.c_corr[i]) * v

                    if corr.noise_std > 0.0:
                        tau += float(corr.noise_std) * float(rng.standard_normal())

                    tau_max = float(corr.tau_bias) + float(corr.tau_gain[i]) * (d * (Tsum_base + float(corr.T_bias)))
                    tau_max = _clip(tau_max, float(corr.tau_min), float(corr.tau_cap))
                    tau = _clip(tau, -tau_max, tau_max)

                    # Запрет генерации энергии.
                    if abs(v) > 1e-9 and (tau * v) > 0.0:
                        tau = 0.0

                    data.qfrc_applied[adr] += tau

                mujoco.mj_step(model, data)

            viewer.sync()

            # Печать статуса моторов/тросов/катушек.
            sim_t = float(data.time)
            if sim_t >= next_print_t:
                ctrlL = float(data.ctrl[id_motor_left])
                ctrlR = float(data.ctrl[id_motor_right])

                l_now_left = _read_tendon_length(data, id_tendon_left)
                l_now_right = _read_tendon_length(data, id_tendon_right)
                ldot_left = _read_tendon_velocity(data, id_tendon_left)
                ldot_right = _read_tendon_velocity(data, id_tendon_right)

                print(
                    "t="
                    + f"{sim_t:7.3f}"
                    + " | vcmd(L,R)="
                    + f"({ctrl.vcmd_left:+.2f},{ctrl.vcmd_right:+.2f})"
                    + " | vspool(L,R)="
                    + f"({ctrl.v_left:+.3f},{ctrl.v_right:+.3f})"
                    + " | Lspool(L,R)="
                    + f"({ctrl.Lspool_left:.4f},{ctrl.Lspool_right:.4f})"
                    + " | ctrl(L,R)="
                    + f"({ctrlL:.3f},{ctrlR:.3f})"
                    + " | Tbase(L,R)="
                    + f"({last_Tbase_left:7.1f},{last_Tbase_right:7.1f})"
                    + " | ten_len(L,R)="
                    + f"({l_now_left:.4f},{l_now_right:.4f})"
                    + " | ten_vel(L,R)="
                    + f"({ldot_left:+.4f},{ldot_right:+.4f})"
                )
                next_print_t = sim_t + (1.0 / print_hz)

            time.sleep(max(0.0, steps_per_frame * dt - 0.0005))

    ctrl.running = False
    listener.stop()


if __name__ == "__main__":
    main()
