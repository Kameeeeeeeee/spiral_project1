# debug_viewer.py

import time
import numpy as np
import mujoco
import mujoco.viewer

from generate_spiral_xml import generate_spiral_tentacle_xml


def main():
    # Генерируем модель на 10 звеньев
    xml = generate_spiral_tentacle_xml(num_links=10)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Параметры сигнала
    amp = 0.5
    freq = 0.3  # Гц

    with mujoco.viewer.launch_passive(model, data) as v:
        t0 = time.time()
        while v.is_running():
            t = time.time() - t0
            # два мотора: левый и правый
            left = amp * np.sin(2 * np.pi * freq * t)
            right = amp * np.sin(2 * np.pi * freq * t + np.pi)

            data.ctrl[0] = left
            data.ctrl[1] = right

            mujoco.mj_step(model, data)
            v.sync()


if __name__ == "__main__":
    main()
