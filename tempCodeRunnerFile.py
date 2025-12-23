# debug_viewer.py
#
# Force-only debug viewer (как в статье): только tendon motor и управление натяжением.
# Ключевые правки:
# 1) Удалено ВСЕ, что добавляло моменты через data.qfrc_applied.
# 2) Удалены propagate_tension и любые "виртуальные" потери. Сейчас один источник изгиба: натяжение троса.
# 3) Добавлен квазистатический режим: плавный ramp сил (Tcmd -> Ttarget) для воспроизведения формы "?".
# 4) Добавлены пресеты сценариев Packing/Reaching/Wrapping и печать статусов 10 Гц.
#
# Управление:
# a/z: увеличить/уменьшить цель Ttarget_left (Н)
# k/m: увеличить/уменьшить цель Ttarget_right (Н)
# [ ]: изменить шаг dT (Н)
# , .: изменить скорость ramp (Н/с)
#
# Пресеты:
# 1: Packing (25,25)
# 2: Reaching-right (25,60)
# 3: Wrapping-right (0,60)
# 4: Reaching-left (60,25)
# 5: Wrapping-left (60,0)
# 0: Release (0,0)
#
# r: reset (и новый спавн кубика, если ты передашь другой seed в XML вручную)
# q: quit

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
from pynput import keyboard

from generate_spiral_xml import generate_spiral_tentacle_xml


@dataclass
class ForceUI:
    # Текущие силы (плавно догоняют таргеты)
    T_left: float = 0.0
    T_right: float = 0.0

    # Целевые силы, которые меняются клавишами
    Tt_left: float = 0.0
    Tt_right: float = 0.0

    # Шаг по клавишам
    dT: float = 2.0

    # Реалистичный лимит (статья: десятки ньютонов)
    Tmax: float = 80.0

    # Скорость квазистатического ramp (Н/с)
    ramp_rate: float = 40.0

    running: bool = True


def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _ramp(current: float, target: float, rate: float, dt: float) -> float:
    # Плавная подача силы вместо "ступеньки", чтобы получить квазистатику как в статье
    if rate <= 0.0:
        return target
    max_step = rate * dt
    if target > current:
        return min(target, current + max_step)
    else:
        return max(target, current - max_step)


def main() -> None:
    xml = generate_spiral_tentacle_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    id_motor_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_left")
    id_motor_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_right")
    if id_motor_left < 0 or id_motor_right < 0:
        raise RuntimeError("Actuators motor_left and motor_right must exist.")

    id_tendon_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "tendon_left")
    id_tendon_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "tendon_right")
    if id_tendon_left < 0 or id_tendon_right < 0:
        raise RuntimeError("Tendons tendon_left and tendon_right must exist.")

    gear_left = float(model.actuator_gear[id_motor_left][0])
    gear_right = float(model.actuator_gear[id_motor_right][0])
    if gear_left <= 0.0 or gear_right <= 0.0:
        raise RuntimeError("Actuator gear must be positive.")

    ui = ForceUI()

    def reset_all() -> None:
        mujoco.mj_resetData(model, data)
        ui.T_left = 0.0
        ui.T_right = 0.0
        ui.Tt_left = 0.0
        ui.Tt_right = 0.0
        data.ctrl[id_motor_left] = 0.0
        data.ctrl[id_motor_right] = 0.0
        mujoco.mj_forward(model, data)

    def apply_ctrl() -> None:
        # T = gear * ctrl => ctrl = T/gear, tension-only (T>=0)
        Tl = _clip(ui.T_left, 0.0, ui.Tmax)
        Tr = _clip(ui.T_right, 0.0, ui.Tmax)
        data.ctrl[id_motor_left] = _clip(Tl / gear_left, 0.0, 1.0)
        data.ctrl[id_motor_right] = _clip(Tr / gear_right, 0.0, 1.0)

    def preset(p: int) -> None:
        # Пресеты повторяют последовательность из статьи: packing -> reaching -> wrapping
        if p == 1:
            ui.Tt_left, ui.Tt_right = 25.0, 25.0
        elif p == 2:
            ui.Tt_left, ui.Tt_right = 25.0, 60.0
        elif p == 3:
            ui.Tt_left, ui.Tt_right = 0.0, 60.0
        elif p == 4:
            ui.Tt_left, ui.Tt_right = 60.0, 25.0
        elif p == 5:
            ui.Tt_left, ui.Tt_right = 60.0, 0.0
        elif p == 0:
            ui.Tt_left, ui.Tt_right = 0.0, 0.0

    def on_press(key) -> None:
        try:
            ch = key.char
        except AttributeError:
            ch = None

        if ch == "a":
            ui.Tt_left = _clip(ui.Tt_left + ui.dT, 0.0, ui.Tmax)
        elif ch == "z":
            ui.Tt_left = _clip(ui.Tt_left - ui.dT, 0.0, ui.Tmax)
        elif ch == "k":
            ui.Tt_right = _clip(ui.Tt_right + ui.dT, 0.0, ui.Tmax)
        elif ch == "m":
            ui.Tt_right = _clip(ui.Tt_right - ui.dT, 0.0, ui.Tmax)

        elif ch == "[":
            ui.dT = max(0.5, ui.dT * 0.8)
        elif ch == "]":
            ui.dT = min(20.0, ui.dT * 1.25)

        elif ch == ",":
            ui.ramp_rate = max(5.0, ui.ramp_rate * 0.8)
        elif ch == ".":
            ui.ramp_rate = min(400.0, ui.ramp_rate * 1.25)

        elif ch in ("0", "1", "2", "3", "4", "5"):
            preset(int(ch))

        elif ch == "r":
            reset_all()
        elif ch == "q":
            ui.running = False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    reset_all()

    dt = float(model.opt.timestep)
    steps_per_frame = 8
    print_hz = 10.0
    next_print_t = 0.0

    # Для удобства даем подсказку по сценарию "?"
    print("Presets: 1 packing, 2 reaching-right, 3 wrapping-right, 4 reaching-left, 5 wrapping-left, 0 release")
    print("To get '?' try: press 1 (wait) -> 2 (wait) -> 3 (after contact or when tip is near object)")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_t = float(data.time)

        while viewer.is_running() and ui.running:
            t = float(data.time)
            dt_sim = max(1e-9, t - last_t)
            last_t = t

            # Квазистатический ramp к целевым силам
            ui.T_left = _ramp(ui.T_left, ui.Tt_left, ui.ramp_rate, dt_sim)
            ui.T_right = _ramp(ui.T_right, ui.Tt_right, ui.ramp_rate, dt_sim)

            apply_ctrl()

            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            viewer.sync()

            # Печать статусов моторов/тросов/сил
            t = float(data.time)
            if t >= next_print_t:
                ctrlL = float(data.ctrl[id_motor_left])
                ctrlR = float(data.ctrl[id_motor_right])

                lenL = float(data.ten_length[id_tendon_left])
                lenR = float(data.ten_length[id_tendon_right])

                velL = float(data.ten_velocity[id_tendon_left])
                velR = float(data.ten_velocity[id_tendon_right])

                print(
                    "t="
                    + f"{t:7.3f}"
                    + " | T(L,R)="
                    + f"({ui.T_left:5.1f},{ui.T_right:5.1f})"
                    + " | Tt(L,R)="
                    + f"({ui.Tt_left:5.1f},{ui.Tt_right:5.1f})"
                    + " | ctrl(L,R)="
                    + f"({ctrlL:5.3f},{ctrlR:5.3f})"
                    + " | ten_len(L,R)="
                    + f"({lenL:.4f},{lenR:.4f})"
                    + " | ten_vel(L,R)="
                    + f"({velL:+.4f},{velR:+.4f})"
                    + " | dT="
                    + f"{ui.dT:.2f}"
                    + " | ramp="
                    + f"{ui.ramp_rate:.1f}"
                    + " | gear="
                    + f"({gear_left:.1f},{gear_right:.1f})"
                )
                next_print_t = t + (1.0 / print_hz)

            # Небольшая пауза, чтобы не грузить CPU
            time.sleep(max(0.0, steps_per_frame * dt - 0.0005))

    ui.running = False
    listener.stop()


if __name__ == "__main__":
    main()
