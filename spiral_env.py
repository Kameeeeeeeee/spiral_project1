import numpy as np
import gymnasium as gym
import mujoco

from generate_spiral_xml import generate_spiral_tentacle_xml


class SpiralTentacle2TEnv(gym.Env):
    """
    Два троса управляют спиральной щупальцей (24 звена) и шаром-целью.

    В ЭТОЙ ВЕРСИИ:
      - цель стейджа 1: ОБВИТЬ шар и УДЕРЖИВАТЬ его, почти не сдвигая,
        а не подтягивать к основанию;
      - вводим симметрию по лево/право: для агента action[0] всегда трос
        со стороны шара, action[1] - трос с противоположной стороны.
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
            base_radius=base_radius,
            tip_radius=tip_radius,
            total_length=total_length,
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # два троса
        self.act_n = self.model.nu
        assert self.act_n == 2, f"expected 2 actuators, got {self.act_n}"

        self.qpos_n = self.model.nq
        self.qvel_n = self.model.nv

        # мягкие моторы, чтобы не разгоняться
        self.ctrl_scale = 0.6

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.act_n,),
            dtype=np.float32,
        )

        # наблюдение: qpos, qvel, tip, ball, rel_tip_ball, rel_ball_base
        obs_dim = self.qpos_n + self.qvel_n + 3 + 3 + 3 + 3
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.tip_body_id = self.model.body(f"link_{self.num_links - 1}").id
        self.base_body_id = self.model.body("base").id
        self.ball_body_id = self.model.body("obj_sphere_hi").id
        self.ball_geom_id = self.model.geom("obj_sphere_hi_geom").id
        self.floor_geom_id = self.model.geom("floor").id
        self.ball_radius = float(self.model.geom_size[self.ball_geom_id][0])

        self.link_body_ids = [
            self.model.body(f"link_{i}").id for i in range(self.num_links)
        ]

        # индексы hinge суставов - для зеркалирования
        self.hinge_qpos_idx = []
        self.hinge_qvel_idx = []
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                qadr = int(self.model.jnt_qposadr[j])
                dadr = int(self.model.jnt_dofadr[j])
                self.hinge_qpos_idx.append(qadr)
                self.hinge_qvel_idx.append(dadr)

        # свободный joint мяча
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
        self.required_hold_steps = 20

        self.grasp_dist_threshold = 0.07
        self.grasp_speed_threshold = 0.02

        self.prev_dist_tip_ball = None
        self.prev_dist_ball_base = None
        self.prev_action = np.zeros(self.act_n, dtype=np.float32)

        # стартовая позиция и стартовая дистанция до базы
        self.ball_init_pos = None
        self.ball_init_base_dist = None

        # знак симметрии: +1 - штатно, -1 - зеркалим по оси Y
        self.sym_sign = 1.0

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
        return in_contact, len(geoms)

    def _symmetrize_q(self, qpos, qvel):
        if self.sym_sign > 0:
            return qpos, qvel
        qpos_sym = qpos.copy()
        qvel_sym = qvel.copy()
        for idx in self.hinge_qpos_idx:
            qpos_sym[idx] *= -1.0
        for idx in self.hinge_qvel_idx:
            qvel_sym[idx] *= -1.0
        return qpos_sym, qvel_sym

    def _symmetrize_vec3(self, v):
        if self.sym_sign > 0:
            return v
        v_sym = v.copy()
        v_sym[1] *= -1.0
        return v_sym

    def _get_obs(self):
        qpos = self.data.qpos.ravel()
        qvel = self.data.qvel.ravel()
        qpos_sym, qvel_sym = self._symmetrize_q(qpos, qvel)

        tip = self._symmetrize_vec3(self._tip_pos())
        ball = self._symmetrize_vec3(self._ball_pos())
        base = self._symmetrize_vec3(self._base_pos())

        rel_tip_ball = ball - tip
        rel_ball_base = ball - base

        return np.concatenate(
            [qpos_sym, qvel_sym, tip, ball, rel_tip_ball, rel_ball_base]
        ).astype(np.float32)

    # ---------- reward: обвить и не тянуть к базе ----------

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
        delta_base = self.prev_dist_ball_base - dist_ball_base  # >0 если тянем к базе

        # смещение от стартовой позиции
        if self.ball_init_pos is None:
            disp = 0.0
        else:
            disp = float(np.linalg.norm(ball - self.ball_init_pos))

        in_contact, num_contacts = self._ball_contact_stats()
        speed = float(np.linalg.norm(ball_vel))
        speed_sq = speed * speed

        # оценка "обвития"
        base_to_ball = ball - base
        tip_to_ball = ball - tip
        base_to_ball_norm = np.linalg.norm(base_to_ball) + 1e-8
        tip_to_ball_norm = np.linalg.norm(tip_to_ball) + 1e-8
        wrap_cos = float(
            np.dot(base_to_ball, tip_to_ball)
            / (base_to_ball_norm * tip_to_ball_norm)
        )
        wrap_term = max(0.0, -wrap_cos)  # 0..1, >0 когда шар между базой и кончиком

        reward = 0.0

        # 1) подползти к шару
        reward += 4.0 * delta_tip
        reward -= 0.06 * dist_tip_ball

        # небольшой штраф за скорость всегда
        reward -= 0.4 * speed_sq

        if in_contact:
            # сам факт контакта
            reward += 2.0

            # больше звеньев в контакте - лучше (обвитие)
            if num_contacts > 1:
                reward += 0.8 * (num_contacts - 1)

            # поощряем "обвитость"
            reward += 2.0 * wrap_term

            # поощряем, что кончик рядом с шаром, но без фанатизма
            reward += 1.2 * max(0.0, 0.08 - dist_tip_ball)

            # сильный штраф за смещение и скорость мяча
            reward -= 6.0 * disp
            reward -= 8.0 * speed_sq

            # дополнительный штраф именно за подтягивание к базе
            if delta_base > 0.0:
                reward -= 6.0 * delta_base
            if self.ball_init_base_dist is not None:
                pull_frac = max(
                    0.0,
                    (self.ball_init_base_dist - dist_ball_base)
                    / max(self.ball_init_base_dist, 1e-6),
                )
                reward -= 4.0 * pull_frac
        else:
            # пока контакта нет - мягкий штраф за скорость
            reward -= 0.1 * speed_sq

        # успех: обвили, кончик рядом, несколько звеньев, шар почти не двигается
        close_tip = dist_tip_ball < self.grasp_dist_threshold
        slow_ball = speed < self.grasp_speed_threshold
        small_disp = disp < 0.02
        good_wrap = wrap_term > 0.5 and num_contacts >= 2

        if in_contact and close_tip and slow_ball and small_disp and good_wrap:
            self.hold_steps += 1
        else:
            self.hold_steps = 0

        success = self.hold_steps >= self.required_hold_steps
        if success:
            reward += 20.0

        # штраф за выброс мяча вверх
        if ball[2] > self.ball_radius * 2.0:
            reward -= 2.0 * (ball[2] - self.ball_radius * 2.0)

        # если не движемся к шару и нет контакта - ленивое состояние
        if abs(delta_tip) < 1e-4 and not in_contact:
            reward -= 0.5

        # энергосбережение и сглаживание действия
        reward -= 0.001 * float(np.sum(np.square(action)))
        delta_a = action - self.prev_action
        reward -= 0.003 * float(np.sum(np.square(delta_a)))
        self.prev_action = action.copy()

        self.prev_dist_tip_ball = dist_tip_ball
        self.prev_dist_ball_base = dist_ball_base

        return (
            reward,
            success,
            dist_tip_ball,
            disp,
            speed,
            wrap_term,
            in_contact,
        )

    # ---------- Gym API ----------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_counter = 0
        self.hold_steps = 0
        self.prev_dist_tip_ball = None
        self.prev_dist_ball_base = None
        self.prev_action[:] = 0.0
        self.ball_init_pos = None
        self.ball_init_base_dist = None
        self.sym_sign = 1.0

        # чуть рандома в начальной позе
        self.data.qpos[:] += 0.02 * self.np_random.normal(size=self.qpos_n)
        mujoco.mj_forward(self.model, self.data)

        base_pos = self._base_pos()
        tip_pos = self._tip_pos()
        base_xy = base_pos[:2]
        tip_xy = tip_pos[:2]
        base_z = base_pos[2]

        link_positions = self.data.xpos[self.link_body_ids]

        dir_xy = tip_xy - base_xy
        norm_xy = np.linalg.norm(dir_xy)
        if norm_xy < 1e-6:
            dir_xy = np.array([1.0, 0.0])
            norm_xy = 1.0
        dir_xy_unit = dir_xy / norm_xy
        dist_base_tip_xy = norm_xy

        perp_xy = np.array([-dir_xy_unit[1], dir_xy_unit[0]], dtype=float)

        clearance = self.ball_radius + self.base_radius * 1.2

        ball_pos = None
        last_candidate = None

        for _ in range(40):
            s = self.np_random.uniform(0.35, 0.9) * dist_base_tip_xy
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

        self.ball_init_pos = self._ball_pos().copy()
        base_pos = self._base_pos()
        self.ball_init_base_dist = float(
            np.linalg.norm(self.ball_init_pos - base_pos)
        )

        # определяем сторону шара по оси Y для симметрии
        self.sym_sign = (
            1.0 if self.ball_init_pos[1] >= base_pos[1] else -1.0
        )

        tip_pos = self._tip_pos()
        self.prev_dist_tip_ball = float(
            np.linalg.norm(tip_pos - self.ball_init_pos)
        )
        self.prev_dist_ball_base = self.ball_init_base_dist

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.step_counter += 1

        action = np.clip(action, -1.0, 1.0)

        # сопоставляем "канонические" действия с реальными тросами
        # motor_0 - левый (Y>0), motor_1 - правый (Y<0)
        if self.sym_sign > 0:
            left_ctrl = action[0]
            right_ctrl = action[1]
        else:
            left_ctrl = action[1]
            right_ctrl = action[0]

        self.data.ctrl[0] = left_ctrl * self.ctrl_scale
        self.data.ctrl[1] = right_ctrl * self.ctrl_scale

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        (
            reward,
            success,
            dist_tip_ball,
            disp,
            speed,
            wrap_term,
            in_contact,
        ) = self._compute_reward(action)

        terminated = success
        truncated = self.step_counter >= self.max_episode_steps

        info = {
            "success": success,
            "dist_tip_ball": dist_tip_ball,
            "ball_disp": disp,
            "ball_speed": speed,
            "wrap_term": wrap_term,
            "in_contact": in_contact,
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
