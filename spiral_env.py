# spiral_env.py

import numpy as np
import gymnasium as gym
import mujoco

from generate_spiral_xml import generate_spiral_tentacle_xml


class SpiralTentacle2TEnv(gym.Env):
    """
    Щупальце из шестиугольных звеньев, приводимая в движение двумя тросами.
    Действия агента: [u_left, u_right] - управляющие сигналы моторам тросов.

    Цель:
      - дотянуться до шара,
      - обхватить его несколькими звеньями,
      - подтянуть к базе и удержать.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,
        num_links: int = 24,
        total_length: float = 0.45,
        base_radius: float = 0.02,
        tip_radius: float = 0.006,
        max_episode_steps: int = 400,
    ):
        super().__init__()

        self.num_links = num_links
        self.total_length = total_length
        self.base_radius = base_radius
        self.tip_radius = tip_radius
        self.max_episode_steps = max_episode_steps

        xml = generate_spiral_tentacle_xml(
            num_links=num_links,
            total_length=total_length,
            base_radius=base_radius,
            tip_radius=tip_radius,
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # два актора - два троса
        self.act_n = self.model.nu
        assert self.act_n == 2, f"expected 2 actuators, got {self.act_n}"

        self.qpos_n = self.model.nq
        self.qvel_n = self.model.nv

        # масштаб действий
        self.ctrl_scale = 1.5

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.act_n,),
            dtype=np.float32,
        )

        # observables: qpos, qvel, tip, ball, rel_tip_ball, rel_ball_base
        obs_dim = self.qpos_n + self.qvel_n + 3 + 3 + 3 + 3
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # идентификаторы тел и геомов
        self.tip_body_id = self.model.body(f"link_{self.num_links - 1}").id
        self.base_body_id = self.model.body("base").id
        self.ball_body_id = self.model.body("obj_sphere_hi").id
        self.ball_geom_id = self.model.geom("obj_sphere_hi_geom").id
        self.floor_geom_id = self.model.geom("floor").id
        self.ball_radius = float(self.model.geom_size[self.ball_geom_id][0])

        # все звенья - для проверки расстояния при спавне мяча
        self.link_body_ids = [
            self.model.body(f"link_{i}").id for i in range(self.num_links)
        ]

        # свободное сочленение мяча
        ball_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_sphere_hi_free"
        )
        self.ball_qpos_adr = int(self.model.jnt_qposadr[ball_joint_id])
        self.ball_dof_adr = int(self.model.jnt_dofadr[ball_joint_id])

        self.render_mode = render_mode
        self.viewer = None

        self.substeps = 10

        self.step_counter = 0
        self.hold_steps = 0
        self.required_hold_steps = 18

        self.grasp_dist_threshold = 0.07
        self.grasp_speed_threshold = 0.04

        self.prev_dist_tip_ball = None
        self.prev_dist_ball_base = None
        self.best_dist_ball_base = None
        self.best_step_base = 0
        self.stall_patience = 50

        self.prev_action = np.zeros(self.act_n, dtype=np.float32)

    # ---------- helpers ----------

    def _tip_pos(self):
        return self.data.xpos[self.tip_body_id].copy()

    def _ball_pos(self):
        return self.data.xpos[self.ball_body_id].copy()

    def _base_pos(self):
        return self.data.xpos[self.base_body_id].copy()

    def _ball_linvel(self):
        return self.data.cvel[self.ball_body_id, 3:].copy()

    def _ball_contact_stats(self):
        """
        Возвращает:
          in_contact - есть ли контакт мяча с любым звеном (кроме пола)
          num_contacts - число разных геометрий щупальца, касающихся мяча
        """
        in_contact = False
        geoms = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            hit_ball1 = c.geom1 == self.ball_geom_id
            hit_ball2 = c.geom2 == self.ball_geom_id
            if not (hit_ball1 or hit_ball2):
                continue
            other = c.geom2 if hit_ball1 else c.geom1
            if other == self.floor_geom_id:
                continue
            in_contact = True
            geoms.add(int(other))
        num_contacts = len(geoms)
        return in_contact, num_contacts

    def _get_obs(self):
        qpos = self.data.qpos.ravel()
        qvel = self.data.qvel.ravel()
        tip = self._tip_pos()
        ball = self._ball_pos()
        base = self._base_pos()
        rel_tip_ball = ball - tip
        rel_ball_base = ball - base
        return np.concatenate(
            [qpos, qvel, tip, ball, rel_tip_ball, rel_ball_base]
        ).astype(np.float32)

    # ---------- reward ----------

    def _compute_reward(self, action):
        tip = self._tip_pos()
        ball = self._ball_pos()
        base = self._base_pos()
        ball_vel = self._ball_linvel()

        dist_tip_ball = float(np.linalg.norm(tip - ball))
        dist_ball_base = float(np.linalg.norm(ball - base))

        if self.prev_dist_tip_ball is None:
            self.prev_dist_tip_ball = dist_tip_ball
        if self.prev_dist_ball_base is None:
            self.prev_dist_ball_base = dist_ball_base

        delta_tip = self.prev_dist_tip_ball - dist_tip_ball
        delta_base = self.prev_dist_ball_base - dist_ball_base

        reward = 0.0

        # фаза "приближения" и фаза "подтягивания"
        near_ball_switch = 0.10

        if dist_tip_ball > near_ball_switch:
            reward += 4.0 * delta_tip
            reward -= 0.06 * dist_tip_ball
        else:
            reward += 1.0 * delta_tip
            reward += 3.0 * delta_base
            reward -= 0.02 * dist_ball_base

        reward -= 0.01 * dist_ball_base

        in_contact, num_contacts = self._ball_contact_stats()

        if in_contact:
            reward += 1.2
            if num_contacts > 1:
                reward += 0.4 * (num_contacts - 1)

        speed = float(np.linalg.norm(ball_vel))
        speed_sq = speed * speed

        base_to_ball = ball - base
        base_to_ball_norm = np.linalg.norm(base_to_ball) + 1e-8
        away_speed = float(np.dot(ball_vel, base_to_ball) / base_to_ball_norm)
        toward_speed = -away_speed

        tip_to_ball = ball - tip
        tip_to_ball_norm = np.linalg.norm(tip_to_ball) + 1e-8

        wrap_cos = float(
            np.dot(base_to_ball, tip_to_ball)
            / (base_to_ball_norm * tip_to_ball_norm)
        )
        wrap_term = max(0.0, -wrap_cos)  # >0, когда шар между базой и кончиком

        # не любим сильные удары по шару
        reward -= 1.5 * speed_sq

        if in_contact:
            reward += 3.0 * delta_base
            reward += 0.8 * max(toward_speed, 0.0)
            reward += 2.0 * wrap_term
            reward += 0.6 * max(0.0, 0.06 - speed)

            if away_speed > 0.003:
                reward -= 15.0 * away_speed

            if speed > 0.04:
                reward -= 15.0 * (speed - 0.04)

            if dist_tip_ball < 0.12 and wrap_term < 0.2 and num_contacts <= 1:
                reward -= 3.0 * (0.2 - wrap_term)

        near_base = dist_ball_base < 0.055
        slow = speed < self.grasp_speed_threshold
        if dist_tip_ball < self.grasp_dist_threshold and slow and in_contact:
            self.hold_steps += 1
        else:
            self.hold_steps = 0

        success = near_base and self.hold_steps >= self.required_hold_steps
        if success:
            reward += 18.0

        # штраф, если шар сильно подлетел
        if ball[2] > self.ball_radius * 2.0:
            reward -= 2.0 * (ball[2] - self.ball_radius * 2.0)

        # штраф за стагнацию
        if abs(delta_tip) < 1e-4 and abs(delta_base) < 1e-4:
            reward -= 0.6

        if self.best_dist_ball_base is None or dist_ball_base < self.best_dist_ball_base:
            self.best_dist_ball_base = dist_ball_base
            self.best_step_base = self.step_counter
        elif (
            self.step_counter - self.best_step_base > self.stall_patience
            and dist_ball_base > 0.12
        ):
            reward -= 7.0
            self.best_step_base = self.step_counter

        # небольшая L2-регуляризация действий
        reward -= 0.002 * float(np.sum(np.square(action)))
        self.prev_action = action.copy()

        self.prev_dist_tip_ball = dist_tip_ball
        self.prev_dist_ball_base = dist_ball_base

        return reward, success, dist_tip_ball, dist_ball_base, speed

    # ---------- Gym API ----------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_counter = 0
        self.hold_steps = 0
        self.prev_dist_tip_ball = None
        self.prev_dist_ball_base = None
        self.best_dist_ball_base = None
        self.best_step_base = 0
        self.prev_action[:] = 0.0

        # легкий шум по начальной позе
        self.data.qpos[:] += 0.02 * self.np_random.normal(size=self.qpos_n)
        mujoco.mj_forward(self.model, self.data)

        base_pos = self._base_pos()
        tip_pos = self._tip_pos()
        base_xy = base_pos[:2]
        tip_xy = tip_pos[:2]
        base_z = base_pos[2]

        link_positions = self.data.xpos[self.link_body_ids]

        # направление база → кончик в плоскости XY
        dir_xy = tip_xy - base_xy
        norm_xy = np.linalg.norm(dir_xy)
        if norm_xy < 1e-6:
            dir_xy = np.array([1.0, 0.0])
            norm_xy = 1.0
        dir_xy_unit = dir_xy / norm_xy
        dist_base_tip_xy = norm_xy

        # нормаль в плоскости XY (перпендикуляр) - туда смещаем шар
        perp_xy = np.array([-dir_xy_unit[1], dir_xy_unit[0]], dtype=float)

        clearance = self.ball_radius + self.base_radius * 1.2

        ball_pos = None
        last_candidate = None

        for _ in range(40):
            # расстояние вдоль оси щупальца
            s = self.np_random.uniform(0.35, 0.9) * dist_base_tip_xy

            # боковой сдвиг - чтобы шар был сбоку, а не над щупальцой
            d_min = 0.08 * self.total_length
            d_max = 0.30 * self.total_length
            d_perp = self.np_random.uniform(d_min, d_max)
            side = 1.0 if self.np_random.random() < 0.5 else -1.0

            ball_xy = base_xy + dir_xy_unit * s + perp_xy * (side * d_perp)

            z = max(base_z, self.ball_radius) + 0.005

            candidate = np.array([ball_xy[0], ball_xy[1], z], dtype=float)
            last_candidate = candidate

            dists = np.linalg.norm(link_positions - candidate, axis=1)
            if float(dists.min()) > clearance:
                ball_pos = candidate
                break

        if ball_pos is None:
            ball_pos = last_candidate

        self.data.qpos[self.ball_qpos_adr : self.ball_qpos_adr + 3] = ball_pos
        self.data.qvel[self.ball_dof_adr : self.ball_dof_adr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)

        tip_pos = self._tip_pos()
        self.prev_dist_tip_ball = float(np.linalg.norm(tip_pos - ball_pos))
        self.prev_dist_ball_base = float(np.linalg.norm(ball_pos - base_pos))
        self.best_dist_ball_base = self.prev_dist_ball_base

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.step_counter += 1

        action = np.clip(action, -1.0, 1.0)
        # 2 компоненты - левой и правый трос
        self.data.ctrl[:] = action * self.ctrl_scale

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, success, dist_tip_ball, dist_ball_base, speed = self._compute_reward(
            action
        )

        terminated = success
        truncated = self.step_counter >= self.max_episode_steps

        info = {
            "success": success,
            "dist_tip_ball": dist_tip_ball,
            "dist_ball_base": dist_ball_base,
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
