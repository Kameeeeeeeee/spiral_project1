# debug_viewer.py
#
# Просмотр формы щупальцы и ручное управление двумя тросами.

import time

import mujoco
import mujoco.viewer
from pynput import keyboard  # pip install pynput

from generate_spiral_xml import generate_spiral_tentacle_xml


left_cmd = 0.0
right_cmd = 0.0

# Чуть меньший шаг, чтобы руками не делать адские рывки.
STEP = 0.1
CMD_MIN = -1.0
CMD_MAX = 1.0


def clip_cmd(v: float) -> float:
    return max(CMD_MIN, min(CMD_MAX, v))


def on_press(key):
    global left_cmd, right_cmd

    try:
        ch = key.char
    except AttributeError:
        ch = None

    if ch == "a":
        left_cmd = clip_cmd(left_cmd + STEP)
        print(f"left_cmd = {left_cmd:.2f}, right_cmd = {right_cmd:.2f}")
    elif ch == "z":
        left_cmd = clip_cmd(left_cmd - STEP)
        print(f"left_cmd = {left_cmd:.2f}, right_cmd = {right_cmd:.2f}")
    elif ch == "k":
        right_cmd = clip_cmd(right_cmd + STEP)
        print(f"left_cmd = {left_cmd:.2f}, right_cmd = {right_cmd:.2f}")
    elif ch == "m":
        right_cmd = clip_cmd(right_cmd - STEP)
        print(f"left_cmd = {left_cmd:.2f}, right_cmd = {right_cmd:.2f}")
    elif key == keyboard.Key.space:
        left_cmd = 0.0
        right_cmd = 0.0
        print("reset motors to 0.0  0.0")


def main():
    global left_cmd, right_cmd

    xml = generate_spiral_tentacle_xml()

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    act_left_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_left"
    )
    act_right_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_right"
    )

    print("MuJoCo viewer: ручное управление щупальцей.")
    print("  A / Z - увеличить / уменьшить левый трос (motor_left)")
    print("  K / M - увеличить / уменьшить правый трос (motor_right)")
    print("  SPACE - сбросить оба мотора в 0")
    print("Закрой окно viewer, чтобы выйти.")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.ctrl[act_left_id] = left_cmd
            data.ctrl[act_right_id] = right_cmd

            # несколько шагов физики на один кадр viewer
            for _ in range(3):
                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(0.003)

    listener.stop()


if __name__ == "__main__":
    main()
