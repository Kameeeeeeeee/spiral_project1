# spiral_env.py

import numpy as np
import gymnasium as gym
import mujoco

from generate_spiral_xml import generate_spiral_tentacle_xml


class SpiralTentacle2TEnv(gym.Env):
    """
    Щупальце с двумя тросами (левый/правый) + мяч для хватания.

    Задача:
    - подтянуть мяч к основанию щупальца (base) и удержать неподвижным.

    Reward:
    - shaping: -dist(ball_xy, base_xy)
    - +0.5 за контакт мяча (с любым объектом, чаще всего с щупальцем)
    - +5, если мяч близко к базе и почти неподвижен N шагов подряд
    - -0.01 * ||action||^2 за энергию
    """

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

        # Генерируем MJCF строку и создаем модель
        xml = generate_spiral_tentacle_xml(
            num_links=num_links,
            link_length=link_length,
            link_radius=link_radius,
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Размерности
        self.act_n = self.model.nu
        self.qpos_n = self.model.nq
        self.qvel_n = self.model.nv

        # Действия - усилия в двух тросах: [left, right]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.act_n,), dtype=np.float32
        )

        # Состояние: qpos, qvel, tip_pos, ball_pos, rel_tip_to_ball
        obs_dim = self.qpos_n + self.qvel_n + 3 + 3 + 3
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Идентификаторы тел и геомов
        tip_body_name = f"link_{self.num_links - 1}"
        self.tip_body_id = self.model.body(tip_body_name).id
        self.base_body_id = self.model.body("base").id
        self.ball_body_id = self.model.body("ball").id
        self.ball_geom_id = self.model.geom("ball_geom").id
        self.floor_geom_id = self.model.geom("floor").id

        # Индексы qpos/qvel для свободного сустава мяча
        ball_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"
        )
        self.ball_qpos_adr = int(self.model.jnt_qposadr[ball_joint_id])
        self.ball_dof_adr = int(self.model.jnt_dofadr[ball_joint_id])

        self.render_mode = render_mode
        self.viewer = None

        self.step_counter = 0
        self.hold_steps = 0
        self.required_hold_steps = 20

        # Порог по расстоянию и скорости для "захвата"
        self.grasp_dist_threshold = 0.03
        self.grasp_speed_threshold = 0.05

        # количество внутренних шагов физики на один шаг RL
        self.substeps = 5

    # --------- Вспомогательные методы ---------

    def _tip_pos(self):
        return self.data.xpos[self.tip_body_id].copy()

    def _ball_pos(self):
        return self.data.xpos[self.ball_body_id].copy()

    def _base_pos(self):
        return self.data.xpos[self.base_body_id].copy()

    def _ball_in_contact_with_tentacle(self):
        """
        Проверяем, есть ли контакт мяча с щупальцем (а не только с полом).
        """
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self.ball_geom_id and c.geom2 != self.floor_geom_id:
                return True
            if c.geom2 == self.ball_geom_id and c.geom1 != self.floor_geom_id:
                return True
        return False

    def _get_obs(self):
        qpos = self.data.qpos.ravel()
        qvel = self.data.qvel.ravel()
        tip = self._tip_pos()
        ball = self._ball_pos()
        rel = ball - tip
        return np.concatenate([qpos, qvel, tip, ball, rel]).astype(np.float32)

    def _compute_reward(self, action):
        base_pos = self._base_pos()
        ball_pos = self._ball_pos()

        # расстояние в плоскости XY от мяча до базы
        dist_xy = np.linalg.norm(ball_pos[:2] - base_pos[:2])

        # 1) shaping - хотим, чтобы мяч был у базы
        reward = -dist_xy

        # 2) контакт мяча с щупальцем
        if self._ball_in_contact_with_tentacle():
            reward += 0.5

        # 3) проверяем "успешный захват": мяч рядом и почти не двигается
        # линейная скорость тела (data.cvel: (nbody, 6))
        ball_linvel = self.data.cvel[self.ball_body_id, 3:]
        speed = float(np.linalg.norm(ball_linvel))

        if dist_xy < self.grasp_dist_threshold and speed < self.grasp_speed_threshold:
            self.hold_steps += 1
        else:
            self.hold_steps = 0

        success = self.hold_steps >= self.required_hold_steps
        if success:
            reward += 5.0

        # штраф за энергию управления
        reward -= 0.01 * float(np.sum(np.square(action)))

        return float(reward), success, dist_xy, speed

    # --------- API Gymnasium ---------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_counter = 0
        self.hold_steps = 0

        # Небольшой шум по положению звеньев
        self.data.qpos[:] += 0.01 * self.np_random.normal(size=self.qpos_n)

        # исходная позиция мяча из модели
        base_ball_pos = self.data.qpos[
            self.ball_qpos_adr : self.ball_qpos_adr + 3
        ].copy()

        # случайное смещение мяча в XY (чтобы не всегда был идеально по центру)
        jitter = np.array(
            [
                0.02 * self.np_random.uniform(-1, 1),
                0.04 * self.np_random.uniform(-1, 1),
                0.0,
            ]
        )
        new_ball_pos = base_ball_pos + jitter
        self.data.qpos[self.ball_qpos_adr : self.ball_qpos_adr + 3] = new_ball_pos

        # обнуляем скорости мяча
        self.data.qvel[self.ball_dof_adr : self.ball_dof_adr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.step_counter += 1

        # клипуем действие
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = action

        # несколько внутренних шагов симуляции
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, success, dist_xy, speed = self._compute_reward(action)

        terminated = success
        truncated = self.step_counter >= self.max_episode_steps

        info = {
            "success": success,
            "dist_ball_base_xy": dist_xy,
            "ball_speed": speed,
            "hold_steps": self.hold_steps,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # --------- Viewer интеграция ---------

    def render(self):
        """
        Встроенный MuJoCo viewer.

        Работает, если при создании env указать render_mode="human".
        Для дебага RL лучше использовать n_envs=1.
        """
        if self.render_mode != "human":
            return

        if self.viewer is None:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer.sync()

    def close(self):
        self.viewer = None
