# debug_viewer.py
#
# Просмотр щупальцы и ручное управление двумя тросами.
# Поддержка русской раскладки: Ф/Я/Л/Ь.
#
# Клавиши:
#   Левый трос:  A(Ф) увеличить, Z(Я) уменьшить
#   Правый трос: K(Л) увеличить, M(Ь) уменьшить
#   Space: сбросить оба мотора в 0
#   [ / ] : уменьшить / увеличить STEP (шаг изменения команды)
#   Esc: выход

import time
import mujoco
import mujoco.viewer
from pynput import keyboard

from generate_spiral_xml import generate_spiral_tentacle_xml


left_cmd = 0.0
right_cmd = 0.0

STEP = 0.10
CMD_MIN = -1.0
CMD_MAX = 1.0

# Сколько шагов физики делать на один кадр viewer
SUBSTEPS = 8


def clip_cmd(v: float) -> float:
    if v < CMD_MIN:
        return CMD_MIN
    if v > CMD_MAX:
        return CMD_MAX
    return v


def print_state(prefix: str = "") -> None:
    print(f"{prefix}left_cmd={left_cmd:.3f} right_cmd={right_cmd:.3f} STEP={STEP:.3f}")


def on_press(key):
    global left_cmd, right_cmd, STEP

    # Спец-клавиши
    if key == keyboard.Key.space:
        left_cmd = 0.0
        right_cmd = 0.0
        print_state("reset: ")
        return

    if key == keyboard.Key.esc:
        # pynput listener остановится сам после закрытия окна viewer
        print("ESC pressed: close viewer window to exit.")
        return

    # Символ
    try:
        ch = key.char
    except AttributeError:
        ch = None

    if not ch:
        return

    ch = ch.lower()

    # EN + RU раскладки
    # A Z K M  ->  Ф Я Л Ь
    if ch in ("a", "ф"):
        left_cmd = clip_cmd(left_cmd + STEP)
        print_state()
    elif ch in ("z", "я"):
        left_cmd = clip_cmd(left_cmd - STEP)
        print_state()
    elif ch in ("k", "л"):
        right_cmd = clip_cmd(right_cmd + STEP)
        print_state()
    elif ch in ("m", "ь"):
        right_cmd = clip_cmd(right_cmd - STEP)
        print_state()

    # STEP change
    elif ch == "[":
        STEP = max(0.005, STEP * 0.8)
        print_state("step: ")
    elif ch == "]":
        STEP = min(0.8, STEP * 1.25)
        print_state("step: ")


def main():
    global left_cmd, right_cmd

    xml = generate_spiral_tentacle_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)


    act_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_left")
    act_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_right")

    if act_left_id < 0 or act_right_id < 0:
        raise RuntimeError("Actuators motor_left/motor_right not found in model.")

    print("MuJoCo viewer: ручное управление щупальцей (EN + RU раскладка).")
    print("  Левый трос:  A(Ф) +, Z(Я) -")
    print("  Правый трос: K(Л) +, M(Ь) -")
    print("  SPACE: сбросить оба мотора в 0")
    print("  [ / ]: изменить STEP")
    print("  ESC: подсказка (закрой окно viewer для выхода)")
    print_state("init: ")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.ctrl[act_left_id] = left_cmd
            data.ctrl[act_right_id] = right_cmd

            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(0.01)

    listener.stop()


if __name__ == "__main__":
    main()
