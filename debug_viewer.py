# debug_viewer.py
import time
import numpy as np
import mujoco
import mujoco.viewer

from pynput import keyboard  # pip install pynput

from spiral_env import SpiralTentacle2TEnv


# глобальные команды для моторов
left_cmd = 0.0
right_cmd = 0.0

STEP = 0.05        # шаг изменения сигнала по клавише
CMD_MIN = -1.0     # границы действий среды
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

    # создаем среду с текущими параметрами щупальцы
    env = SpiralTentacle2TEnv(
        render_mode=None,
        num_links=24,
        total_length=0.45,
        base_radius=0.02,
        tip_radius=0.006,
        max_episode_steps=10_000,
    )

    model = env.model
    data = env.data

    obs, _ = env.reset()

    print("Открываю MuJoCo viewer для ручного управления щупальцей.")
    print("Клавиши управления:")
    print("  A / Z - увеличить / уменьшить левый трос (motor_left)")
    print("  K / M - увеличить / уменьшить правый трос (motor_right)")
    print("  SPACE - сбросить оба мотора в 0")
    print("Закрыть окно - просто закрой окно viewer.")

    # запускаем глобальный слушатель клавиатуры
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # берем текущие значения моторчиков из глобальных переменных
            action = np.array([left_cmd, right_cmd], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                # начинаем новый эпизод, управлять можно дальше
                obs, _ = env.reset()

            viewer.sync()
            time.sleep(0.01)

    listener.stop()


if __name__ == "__main__":
    main()
