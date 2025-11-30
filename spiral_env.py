# spiral_env.py

import numpy as np
import gymnasium as gym
import mujoco

from generate_spiral_xml import generate_spiral_tentacle_xml


class SpiralTentacle2TEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,
        num_links: int = 10,
        link_length: float = 0.05,
        link_radius: float = 0.01,
        max_episode_steps: int = 300,
    ):
        super().__init__()

        self.num_links = num_links
        self.link_length = link_length
        self.link_radius = link_radius
        self.max_episode_steps = max_episode_steps

        # Генерируем MJCF строку и делаем модель
        xml = generate_spiral_tentacle_xml(
            num_links=num_links,
            link_length=link_length,
            link_radius=link_radius,
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Индексы
        self.act_n = self.model.nu
        self.qpos_n = self.model.nq
        self.qvel_n = self.model.nv

        # Действия - усилия в двух тросах
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.act_n,), dtype=np.float32
        )

        # Состояние: qpos, qvel, tip_pos, target_pos, rel (итого qpos_n + qvel_n + 3 + 3 + 3)
        obs_dim = self.qpos_n + self.qvel_n + 3 + 3 + 3
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.step_counter = 0

        # Цель
        self.target_pos = np.array([0.2, 0.0, 0.1], dtype=float)

        # Идентификатор кончика
        tip_body_name = f"link_{self.num_links - 1}"
        self.tip_body_id = self.model.body(tip_body_name).id

    def _get_tip_pos(self):
        return self.data.xpos[self.tip_body_id].copy()

    def _get_obs(self):
        qpos = self.data.qpos.ravel()
        qvel = self.data.qvel.ravel()
        tip = self._get_tip_pos()
        rel = self.target_pos - tip
        # можно добавить в наблюдение саму цель
        return np.concatenate([qpos, qvel, tip, self.target_pos, rel]).astype(
            np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_counter = 0

        # Небольшой рандом по начальному состоянию
        self.data.qpos[:] += 0.01 * self.np_random.normal(size=self.qpos_n)

        # Рандомная цель в секторе перед щупальцем
        # пусть x в [0.15, 0.25], y в [-0.05, 0.05], z в [0.05, 0.15]
        self.target_pos = np.array(
            [
                0.2 + 0.05 * self.np_random.uniform(-1, 1),
                0.0 + 0.05 * self.np_random.uniform(-1, 1),
                0.1 + 0.05 * self.np_random.uniform(-1, 1),
            ],
            dtype=float,
        )

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.step_counter += 1

        # Клипуем действие
        action = np.clip(action, -1.0, 1.0)

        # Масштабируем к реальным силам (можно подбирать)
        max_torque = 1.0
        self.data.ctrl[:] = action * max_torque

        # Несколько внутренних шагов симуляции
        substeps = 5
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)

        tip = self._get_tip_pos()
        dist = np.linalg.norm(tip - self.target_pos)

        # Награда: минус расстояние плюс маленький бонус за близость
        reward = -dist
        if dist < 0.02:
            reward += 1.0

        # Штраф за энергию
        reward -= 0.01 * float(np.sum(np.square(action)))

        terminated = dist < 0.02
        truncated = self.step_counter >= self.max_episode_steps

        obs = self._get_obs()
        info = {"distance": dist}

        return obs, reward, terminated, truncated, info

    # для интеграции с viewer мы не реализуем отрисовку здесь
    def render(self):
        pass

    def close(self):
        pass


