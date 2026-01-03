# debug_viewer.py
#
# Viewer + ручное управление ДВУМЯ ТРОСАМИ (force-control) для двухтроссовой спиральной щупальцы SpiRob.
#
# Цель: поведение как в статье - управление натяжением троса (Н), без внешней "коррекции моментов"
# и без qfrc_applied. Только tendon motors в MJCF.
#
# Управление:
# a/z: увеличить/уменьшить натяжение левого троса
# k/m: увеличить/уменьшить натяжение правого троса
# space: поставить оба натяжения в 0
# [ ]: изменить шаг изменения натяжения (T_step)
# 1: Packing (T_L=T_R=25N)
# 2: Reaching (T_L=25N, T_R=60N)
# 3: Wrapping prep (T_L=10N, T_R=60N)
# 4: Relax left (T_L=0N,  T_R=60N)
# q: выход
#
# Примечание:
# - Реалистичный диапазон сил из статьи: до ~100N (мы даем запас до 120N).
# - Если щупальца не докручивается в packed, это почти всегда контакт (margin/трение) или слишком жесткая ось,
#   а не "нехватка силы".

from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
from pynput import keyboard  # pip install pynput

from generate_spiral_xml import generate_spiral_tentacle_xml


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _find_actuator_id(model: mujoco.MjModel, name: str) -> int:
    try:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    except Exception:
        return -1


def _get_gear_scalar(model: mujoco.MjModel, actuator_id: int) -> float:
    # model.actuator_gear is (nu, 6); for tendon motors gear[0] is the scalar gain
    g = float(model.actuator_gear[actuator_id][0])
    return g if g != 0.0 else 1.0


def _tendon_id(model: mujoco.MjModel, name: str) -> int:
    try:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
    except Exception:
        return -1


@dataclass
class ForceController:
    T_left: float = 0.0   # N
    T_right: float = 0.0  # N
    T_step: float = 2.0   # N per key press

    Tmax: float = 210.0   # N

    running: bool = True


def main() -> None:
    xml = generate_spiral_tentacle_xml()

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    id_motor_left = _find_actuator_id(model, "motor_left")
    id_motor_right = _find_actuator_id(model, "motor_right")
    if id_motor_left < 0 or id_motor_right < 0:
        raise RuntimeError("Не найдены actuators motor_left/motor_right. Проверь generate_spiral_xml.py.")

    gear_left = _get_gear_scalar(model, id_motor_left)
    gear_right = _get_gear_scalar(model, id_motor_right)

    # Для статуса (не обязательно)
    tid_left = _tendon_id(model, "tendon_left")
    tid_right = _tendon_id(model, "tendon_right")

    ctrl = ForceController()

    def on_press(key) -> None:
        try:
            k = key.char
        except AttributeError:
            k = None

        if key == keyboard.Key.space:
            ctrl.T_left = 0.0
            ctrl.T_right = 0.0
            return

        if k is None:
            if key == keyboard.Key.esc:
                ctrl.running = False
            return

        if k == "q":
            ctrl.running = False
        elif k == "a":
            ctrl.T_left = _clip(ctrl.T_left + ctrl.T_step, 0.0, ctrl.Tmax)
        elif k == "z":
            ctrl.T_left = _clip(ctrl.T_left - ctrl.T_step, 0.0, ctrl.Tmax)
        elif k == "k":
            ctrl.T_right = _clip(ctrl.T_right + ctrl.T_step, 0.0, ctrl.Tmax)
        elif k == "m":
            ctrl.T_right = _clip(ctrl.T_right - ctrl.T_step, 0.0, ctrl.Tmax)
        elif k == "[":
            ctrl.T_step = max(0.5, ctrl.T_step * 0.8)
        elif k == "]":
            ctrl.T_step = min(20.0, ctrl.T_step * 1.25)
        elif k == "1":
            ctrl.T_left, ctrl.T_right = 210.0, 0.0
        elif k == "2":
            ctrl.T_left, ctrl.T_right = 40.0, 100.0
        elif k == "3":
            ctrl.T_left, ctrl.T_right = 10.0, 60.0
        elif k == "4":
            ctrl.T_left, ctrl.T_right = 0.0, 60.0

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Частота печати статуса
    t_last = time.time()
    print_hz = 10.0
    print_dt = 1.0 / print_hz

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0

        while viewer.is_running() and ctrl.running:
            step_start = time.time()

            # Прямая установка управления в Н через ctrl = T / gear
            data.ctrl[id_motor_left] = _clip(ctrl.T_left / gear_left, 0.0, 1.0)
            data.ctrl[id_motor_right] = _clip(ctrl.T_right / gear_right, 0.0, 1.0)

            mujoco.mj_step(model, data)

            # Статус
            t = time.time()
            if (t - t_last) >= print_dt:
                t_last = t

                # actuator_force для tendon motor - фактическая сила по тендону (в Н)
                F_left = float(data.actuator_force[id_motor_left])
                F_right = float(data.actuator_force[id_motor_right])

                if tid_left >= 0:
                    L_left = float(data.ten_length[tid_left])
                    v_left = float(data.ten_velocity[tid_left])
                else:
                    L_left, v_left = float("nan"), float("nan")

                if tid_right >= 0:
                    L_right = float(data.ten_length[tid_right])
                    v_right = float(data.ten_velocity[tid_right])
                else:
                    L_right, v_right = float("nan"), float("nan")

                print(
                    f"Tcmd(L,R)=({ctrl.T_left:6.1f},{ctrl.T_right:6.1f}) N | "
                    f"ctrl(L,R)=({data.ctrl[id_motor_left]:.3f},{data.ctrl[id_motor_right]:.3f}) | "
                    f"Tact(L,R)=({F_left:6.1f},{F_right:6.1f}) N | "
                    f"tenL=({L_left:.4f},{L_right:.4f}) m | "
                    f"tenV=({v_left:.4f},{v_right:.4f}) m/s | "
                    f"T_step={ctrl.T_step:.2f} N"
                )

            viewer.sync()

            # Реалтайм
            elapsed = time.time() - step_start
            sleep = model.opt.timestep - elapsed
            if sleep > 0:
                time.sleep(sleep)

    ctrl.running = False
    listener.stop()
    listener.join(timeout=1.0)


if __name__ == "__main__":
    main()
