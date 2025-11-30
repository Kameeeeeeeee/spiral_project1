# spiral_env.py

import numpy as np
import gymnasium as gym
import mujoco

from generate_spiral_xml import generate_spiral_tentacle_xml


class SpiralTentacle2TEnv(gym.Env):
    """
    Щупальце из 24 лежащих шестиугольных звеньев + мяч.

    - движение только в горизонтальной плоскости (суставы вокруг оси Z, gravity = 0)
    - действие: 2 мотора (левый/правый трос)
    - цель: подползти кончиком к мячу и удержать его
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,
        num_links: int = 24,
        total_length: float = 0.45,
        base_radius: float = 0.02,
        tip_radius: float = 0.006,
        max_episode_steps: int = 300,
    ):
        super().__init__()

        self.num_links = num_links
        self.total_length = total_length
        self.base_radius = base_radius
        self.tip_radius = tip_radius
        self.max_episode_steps = max_episode_steps

        # генерируем XML для шестиугольного щупальца
        xml = generate_spiral_tentacle_xml(
            num_links=num_links,
            base_radius=base_radius,
            tip_radius=tip_radius,
            total_length=total_length,
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.act_n = self.model.nu
        self.qpos_n = self.model.nq
        self.qvel_n = self.model.nv

        # 2 мотора - левый и правый трос
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.act_n,), dtype=np.float32
        )

        # наблюдение: qpos, qvel, tip_pos, ball_pos, rel_tip_to_ball
        obs_dim = self.qpos_n + self.qvel_n + 3 + 3 + 3
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # индексы тел и геомов
        tip_body_name = f"link_{self.num_links - 1}"
        self.tip_body_id = self.model.body(tip_body_name).id
        self.base_body_id = self.model.body("base").id
        self.ball_body_id = self.model.body("ball").id
        self.ball_geom_id = self.model.geom("ball_geom").id
        self.floor_geom_id = self.model.geom("floor").id

        # qpos/qvel для свободного сустава мяча
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

        # пороги для захвата
        self.grasp_dist_threshold = 0.05
        self.grasp_speed_threshold = 0.05

        # сколько внутренних шагов физики на один шаг RL
        self.substeps = 5

        # для shaped reward по ∆расстояния кончик-мяч
        self.prev_dist_tip_ball = None

    # ---------- вспомогательные функции ----------

    def _tip_pos(self):
        return self.data.xpos[self.tip_body_id].copy()

    def _ball_pos(self):
        return self.data.xpos[self.ball_body_id].copy()

    def _base_pos(self):
        return self.data.xpos[self.base_body_id].copy()

    def _ball_in_contact_with_tentacle(self):
        """
        Есть ли контакт мяча с чем-то, кроме пола.
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
        """
        Shaping:
          - поощряем уменьшение расстояния кончик-мяч (разность с прошлым шагом)
          - штрафуем за большое расстояние (чтобы не висеть далеко)
          - бонус за контакт
          - большой бонус за устойчивый захват
        """
        tip_pos = self._tip_pos()
        ball_pos = self._ball_pos()

        dist_tip_ball = float(np.linalg.norm(tip_pos - ball_pos))

        # если по какой-то причине prev_dist ещё не определён - просто ставим его
        if self.prev_dist_tip_ball is None:
            self.prev_dist_tip_ball = dist_tip_ball

        # improvement: насколько мы приблизились к мячу с прошлого шага
        delta = self.prev_dist_tip_ball - dist_tip_ball

        # 1) поощрение за приближение к мячу
        #    если двигаемся в неверную сторону (не туда по оси Y), reward < 0
        reward = 5.0 * delta

        # 2) штраф за то, что вообще далеко от мяча
        reward -= 0.1 * dist_tip_ball

        # 3) бонус за контакт (не с полом)
        if self._ball_in_contact_with_tentacle():
            reward += 1.0

        # 4) успешный захват: мяч рядом с кончиком и почти не двигается
        ball_linvel = self.data.cvel[self.ball_body_id, 3:]
        speed = float(np.linalg.norm(ball_linvel))

        if dist_tip_ball < self.grasp_dist_threshold and speed < self.grasp_speed_threshold:
            self.hold_steps += 1
        else:
            self.hold_steps = 0

        success = self.hold_steps >= self.required_hold_steps
        if success:
            reward += 10.0

        # 5) небольшой штраф за энергию
        reward -= 0.001 * float(np.sum(np.square(action)))

        # обновляем prev_dist для следующего шага
        self.prev_dist_tip_ball = dist_tip_ball

        return reward, success, dist_tip_ball, speed

    # ---------- API Gymnasium ----------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_counter = 0
        self.hold_steps = 0
        self.prev_dist_tip_ball = None

        # небольшой шум по qpos, чтобы не заучивал одну позу
        self.data.qpos[:] += 0.01 * self.np_random.normal(size=self.qpos_n)

        mujoco.mj_forward(self.model, self.data)
        base_pos = self._base_pos()
        base_xy = base_pos[:2]
        base_z = base_pos[2]

        # мяч спавним в секторе перед щупальцем, обязательно в зоне досягаемости
        # берем радиус поближе к кончику, чтобы движение было содержательным
        min_r = 0.6 * self.total_length
        max_r = 0.9 * self.total_length
        r = self.np_random.uniform(min_r, max_r)

        # угол шире, чтобы мяч часто был то слева, то справа
        theta = self.np_random.uniform(-np.pi / 2, np.pi / 2)

        x = base_xy[0] + r * np.cos(theta)
        y = base_xy[1] + r * np.sin(theta)
        z = base_z

        ball_pos = np.array([x, y, z], dtype=float)
        self.data.qpos[self.ball_qpos_adr : self.ball_qpos_adr + 3] = ball_pos
        self.data.qvel[self.ball_dof_adr : self.ball_dof_adr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)

        # инициализируем prev_dist для shaped reward
        tip_pos = self._tip_pos()
        self.prev_dist_tip_ball = float(np.linalg.norm(tip_pos - ball_pos))

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.step_counter += 1

        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = action

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, success, dist_tip_ball, speed = self._compute_reward(action)

        terminated = success
        truncated = self.step_counter >= self.max_episode_steps

        info = {
            "success": success,
            "dist_tip_ball": dist_tip_ball,
            "ball_speed": speed,
            "hold_steps": self.hold_steps,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ---------- viewer ----------

    def render(self):
        if self.render_mode != "human":
            return

        if self.viewer is None:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer.sync()

    def close(self):
        self.viewer = None
