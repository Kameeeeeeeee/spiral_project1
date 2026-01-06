from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

import deb_v2 as deb


@dataclass
class EnvCfg:
    seed: int = 0
    domain_randomization: bool = True

    episode_len: int = 450
    control_dt: float = 0.02

    action_mode: str = "delta"  # "delta" or "absolute"
    dmax_per_step: float = 10.0  # N change per control step for delta mode

    # sim2real-ish control bandwidth (you can randomize later)
    ctrl_slew_rate: float = 600.0  # N/s

    # Reward weights
    w_ball_progress: float = 3.0
    w_tip_progress: float = 0.4
    w_wrap: float = 1.2
    w_time: float = 0.002
    w_effort: float = 0.015
    w_anti_away: float = 1.5
    bonus_wrap_pull: float = 0.5
    success_bonus: float = 10.0

    # Success criteria
    success_ball_base: float = 0.035
    success_wrap_frac: float = 0.25


class SpiralEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None, cfg: EnvCfg | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.cfg = cfg if cfg is not None else EnvCfg()

        xml = deb.build_mjcf()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.n_substeps = max(1, int(round(self.cfg.control_dt / float(self.model.opt.timestep))))
        self.dt_effective = float(self.model.opt.timestep) * self.n_substeps

        # IDs from your model naming
        self.motor_left_ids: list[int] = []
        self.motor_right_ids: list[int] = []
        for i in range(deb.N_SEGMENTS - 1):
            il = deb._find_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_L_{i:02d}_{i+1:02d}")
            ir = deb._find_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_R_{i:02d}_{i+1:02d}")
            if il < 0 or ir < 0:
                raise RuntimeError("Missing actuator ids")
            self.motor_left_ids.append(il)
            self.motor_right_ids.append(ir)

        self.joint_qposadrs: list[int] = []
        self.joint_qveladrs: list[int] = []
        for j in range(1, deb.N_SEGMENTS):
            jid = deb._find_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{j:02d}")
            if jid < 0:
                raise RuntimeError("Missing joint id")
            self.joint_qposadrs.append(int(self.model.jnt_qposadr[jid]))
            self.joint_qveladrs.append(int(self.model.jnt_dofadr[jid]))

        ball_x_jid = deb._find_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_slide_x")
        ball_y_jid = deb._find_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_slide_y")
        if ball_x_jid < 0 or ball_y_jid < 0:
            raise RuntimeError("Missing ball slide joint id")
        self.ball_x_qadr = int(self.model.jnt_qposadr[ball_x_jid])
        self.ball_y_qadr = int(self.model.jnt_qposadr[ball_y_jid])
        self.ball_x_vadr = int(self.model.jnt_dofadr[ball_x_jid])
        self.ball_y_vadr = int(self.model.jnt_dofadr[ball_y_jid])

        self.base_body_id = deb._find_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "seg_00")
        self.ball_body_id = deb._find_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.ball_geom_id = deb._find_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        if self.base_body_id < 0 or self.ball_body_id < 0 or self.ball_geom_id < 0:
            raise RuntimeError("Missing base/ball ids")

        self.seg_geom_ids: list[int] = []
        for i in range(deb.N_SEGMENTS):
            gid = deb._find_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"geom_{i:02d}")
            if gid < 0:
                raise RuntimeError("Missing segment geom ids")
            self.seg_geom_ids.append(gid)
        self.seg_geom_to_index = {gid: i for i, gid in enumerate(self.seg_geom_ids)}

        self.ball_bid = self.ball_body_id

        # DR baselines
        self.base_geom_friction = self.model.geom_friction.copy()
        self.base_dof_damping = self.model.dof_damping.copy()
        self.base_dof_frictionloss = self.model.dof_frictionloss.copy()
        self.base_body_mass = self.model.body_mass.copy()
        self.base_body_inertia = self.model.body_inertia.copy()
        self.has_dof_spring = hasattr(self.model, "dof_spring")
        self.has_jnt_stiffness = hasattr(self.model, "jnt_stiffness")
        self.base_dof_spring = self.model.dof_spring.copy() if self.has_dof_spring else None
        self.base_jnt_stiffness = self.model.jnt_stiffness.copy() if self.has_jnt_stiffness else None

        self.dof_ids: list[int] = []
        self.joint_ids: list[int] = []
        for j in range(1, deb.N_SEGMENTS):
            jid = deb._find_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{j:02d}")
            dof = int(self.model.jnt_dofadr[jid])
            self.dof_ids.append(dof)
            self.joint_ids.append(jid)

        # Controller (2 inputs only)
        self.ctrl = deb.ForceController()
        self.ctrl.T_left_seg = [0.0] * (deb.N_SEGMENTS - 1)
        self.ctrl.T_right_seg = [0.0] * (deb.N_SEGMENTS - 1)
        self.ctrl.T_left_prev = 0.0
        self.ctrl.T_right_prev = 0.0

        # DR config from deb_v2
        self.dr_cfg = deb.DomainRandCfg(enabled=self.cfg.domain_randomization, seed=self.cfg.seed, log_on_reset=False)
        self.dr_rng = np.random.default_rng(self.dr_cfg.seed)
        self.spawn_rng = np.random.default_rng(self.dr_cfg.seed + 1)

        # RL spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        obs_dim = 0
        obs_dim += (deb.N_SEGMENTS - 1)  # q
        obs_dim += (deb.N_SEGMENTS - 1)  # qdot
        obs_dim += 2  # Tcmd normalized
        obs_dim += 3  # ball rel base
        obs_dim += 2  # ball vel xy
        obs_dim += 3  # base pos (for normalization stability)
        obs_dim += 3  # tip pos (approx: last body pos)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._obs_dim = obs_dim
        self._step = 0
        self._prev_d_tip_ball: float | None = None
        self._prev_d_ball_base: float | None = None
        self._touched = False

        self._viewer = None

    # This is copied 1:1 in logic from deb_v2.main() nested function, but placed here for import-ability.
    def apply_domain_randomization(self) -> None:
        cfg = self.dr_cfg
        if not cfg.enabled:
            return

        rng = self.dr_rng

        for gid in self.seg_geom_ids:
            f = self.base_geom_friction[gid].copy()
            f[0] *= deb._logu(rng, *cfg.seg_fric_slide)
            f[1] *= deb._logu(rng, *cfg.seg_fric_tors)
            f[2] *= deb._logu(rng, *cfg.seg_fric_roll)
            self.model.geom_friction[gid] = f

        if self.ball_geom_id >= 0:
            f = self.base_geom_friction[self.ball_geom_id].copy()
            f[0] *= deb._logu(rng, *cfg.ball_fric_slide)
            f[1] *= deb._logu(rng, *cfg.ball_fric_tors)
            f[2] *= deb._logu(rng, *cfg.ball_fric_roll)
            self.model.geom_friction[self.ball_geom_id] = f

        for dof in self.dof_ids:
            self.model.dof_damping[dof] = self.base_dof_damping[dof] * deb._logu(rng, *cfg.dof_damping)
            self.model.dof_frictionloss[dof] = self.base_dof_frictionloss[dof] * deb._logu(rng, *cfg.dof_frictionloss)

        if self.has_jnt_stiffness or self.has_dof_spring:
            for jid, dof in zip(self.joint_ids, self.dof_ids):
                scale = deb._logu(rng, *cfg.dof_spring)
                if self.has_jnt_stiffness and self.base_jnt_stiffness is not None:
                    self.model.jnt_stiffness[jid] = self.base_jnt_stiffness[jid] * scale
                if self.has_dof_spring and self.base_dof_spring is not None:
                    self.model.dof_spring[dof] = self.base_dof_spring[dof] * scale

        mscale = deb._logu(rng, *cfg.mass_scale)
        for b in range(self.model.nbody):
            if b == 0 or b == self.ball_bid:
                continue
            jitter = 1.0 + deb._u(rng, -cfg.per_link_mass_jitter, cfg.per_link_mass_jitter)
            s = mscale * jitter
            self.model.body_mass[b] = self.base_body_mass[b] * s
            self.model.body_inertia[b] = self.base_body_inertia[b] * s

        self.ctrl.mu_static = deb._u(rng, *cfg.mu_static)
        self.ctrl.mu_kinetic = deb._u(rng, *cfg.mu_kinetic)
        self.ctrl.cable_bias = deb._u(rng, *cfg.cable_bias)

    def _base_pos(self) -> np.ndarray:
        return self.data.xpos[self.base_body_id].copy()

    def _ball_pos(self) -> np.ndarray:
        return self.data.xpos[self.ball_body_id].copy()

    def _last_body_pos(self) -> np.ndarray:
        last_bid = deb._find_id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"seg_{deb.N_SEGMENTS-1:02d}")
        return self.data.xpos[last_bid].copy()

    def _ball_vel_xy(self) -> np.ndarray:
        vx = float(self.data.qvel[self.ball_x_vadr])
        vy = float(self.data.qvel[self.ball_y_vadr])
        return np.array([vx, vy], dtype=np.float32)

    def _count_wrap_contacts(self) -> int:
        touched = set()
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if g1 == self.ball_geom_id and g2 in self.seg_geom_to_index:
                touched.add(self.seg_geom_to_index[g2])
            elif g2 == self.ball_geom_id and g1 in self.seg_geom_to_index:
                touched.add(self.seg_geom_to_index[g1])
        return len(touched)

    def _get_obs(self) -> np.ndarray:
        q = np.array([self.data.qpos[a] for a in self.joint_qposadrs], dtype=np.float32)
        qd = np.array([self.data.qvel[a] for a in self.joint_qveladrs], dtype=np.float32)

        base = self._base_pos()
        ball = self._ball_pos()
        tip = self._last_body_pos()
        ball_rel = (ball - base).astype(np.float32)
        ball_vel = self._ball_vel_xy()

        tnorm = np.array([self.ctrl.T_left / self.ctrl.Tmax, self.ctrl.T_right / self.ctrl.Tmax], dtype=np.float32)

        obs = np.concatenate(
            [q, qd, tnorm, ball_rel, ball_vel, base.astype(np.float32), tip.astype(np.float32)],
            dtype=np.float32,
        )
        return obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.dr_cfg.seed = int(seed)
            self.dr_rng = np.random.default_rng(self.dr_cfg.seed)
            self.spawn_rng = np.random.default_rng(self.dr_cfg.seed + 1)

        mujoco.mj_resetData(self.model, self.data)
        self.apply_domain_randomization()

        # Use same spawn logic as deb_v2: build_mjcf sets BALL_* globals.
        bx, by = deb._sample_ball_xy(deb.BALL_SPAWN_RADIUS, deb.BALL_MIN_Y_CLEAR, self.spawn_rng)
        self.data.qpos[self.ball_x_qadr] = bx - deb.BALL_BASE_X
        self.data.qpos[self.ball_y_qadr] = by - deb.BALL_BASE_Y

        self.ctrl.T_left = 0.0
        self.ctrl.T_right = 0.0
        self.ctrl.T_left_target = 0.0
        self.ctrl.T_right_target = 0.0
        self.ctrl.T_left_seg = [0.0] * (deb.N_SEGMENTS - 1)
        self.ctrl.T_right_seg = [0.0] * (deb.N_SEGMENTS - 1)
        self.ctrl.T_left_prev = 0.0
        self.ctrl.T_right_prev = 0.0
        self.data.ctrl[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self._step = 0
        base = self._base_pos()
        ball = self._ball_pos()
        tip = self._last_body_pos()
        self._prev_d_tip_ball = float(np.linalg.norm(tip - ball))
        self._prev_d_ball_base = float(np.linalg.norm(ball - base))
        self._touched = False

        return self._get_obs(), {"d_tip_ball": self._prev_d_tip_ball, "d_ball_base": self._prev_d_ball_base}

    def step(self, action: np.ndarray):
        self._step += 1
        a = np.asarray(action, dtype=np.float32).reshape(2)

        if self.cfg.action_mode == "absolute":
            self.ctrl.T_left_target = deb._clip(0.5 * (float(a[0]) + 1.0) * self.ctrl.Tmax, 0.0, self.ctrl.Tmax)
            self.ctrl.T_right_target = deb._clip(0.5 * (float(a[1]) + 1.0) * self.ctrl.Tmax, 0.0, self.ctrl.Tmax)
        else:
            self.ctrl.T_left_target = deb._clip(self.ctrl.T_left_target + float(a[0]) * self.cfg.dmax_per_step, 0.0, self.ctrl.Tmax)
            self.ctrl.T_right_target = deb._clip(self.ctrl.T_right_target + float(a[1]) * self.cfg.dmax_per_step, 0.0, self.ctrl.Tmax)

        max_dT = float(self.cfg.ctrl_slew_rate * self.dt_effective)
        dL = deb._clip(self.ctrl.T_left_target - self.ctrl.T_left, -max_dT, max_dT)
        dR = deb._clip(self.ctrl.T_right_target - self.ctrl.T_right, -max_dT, max_dT)
        self.ctrl.T_left = deb._clip(self.ctrl.T_left + dL, 0.0, self.ctrl.Tmax)
        self.ctrl.T_right = deb._clip(self.ctrl.T_right + dR, 0.0, self.ctrl.Tmax)

        q = [float(self.data.qpos[adr]) for adr in self.joint_qposadrs]
        t_left, t_right = deb.compute_segment_tensions(self.ctrl, q, self.dt_effective)
        deb.apply_segment_tensions_to_motors(self.data, self.motor_left_ids, self.motor_right_ids, t_left, t_right)

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self.ctrl.T_left_prev = self.ctrl.T_left
        self.ctrl.T_right_prev = self.ctrl.T_right

        base = self._base_pos()
        ball = self._ball_pos()
        tip = self._last_body_pos()

        d_tip_ball = float(np.linalg.norm(tip - ball))
        d_ball_base = float(np.linalg.norm(ball - base))

        wrap_count = self._count_wrap_contacts()
        wrap_frac = float(wrap_count) / float(deb.N_SEGMENTS)

        prev_tip = float(self._prev_d_tip_ball if self._prev_d_tip_ball is not None else d_tip_ball)
        prev_ball = float(self._prev_d_ball_base if self._prev_d_ball_base is not None else d_ball_base)
        tip_progress = prev_tip - d_tip_ball
        ball_progress = prev_ball - d_ball_base

        r = 0.0
        r += 0.8 * tip_progress
        r += 4.0 * ball_progress
        r += -0.25 * d_tip_ball
        r += -0.60 * d_ball_base
        r += 2.0 * wrap_frac

        if (not self._touched) and (wrap_count > 0):
            r += 2.0
            self._touched = True

        r += -0.0005
        r += -0.006 * (self.ctrl.T_left + self.ctrl.T_right) / self.ctrl.Tmax

        if ball_progress < 0.0:
            r += 2.0 * ball_progress

        if wrap_frac > 0.20 and ball_progress > 0.0:
            r += 0.8

        self._prev_d_tip_ball = d_tip_ball
        self._prev_d_ball_base = d_ball_base

        success = (d_ball_base < self.cfg.success_ball_base) and (wrap_frac > self.cfg.success_wrap_frac)
        terminated = bool(success)
        truncated = bool(self._step >= self.cfg.episode_len)
        if terminated:
            r += self.cfg.success_bonus

        obs = self._get_obs()
        info = {
            "d_tip_ball": d_tip_ball,
            "d_ball_base": d_ball_base,
            "wrap_count": wrap_count,
            "wrap_frac": wrap_frac,
            "T_left": float(self.ctrl.T_left),
            "T_right": float(self.ctrl.T_right),
            "a0": float(a[0]),
            "a1": float(a[1]),
            "Tsum": float(self.ctrl.T_left + self.ctrl.T_right),
        }

        if self.render_mode == "human":
            self.render()

        return obs, float(r), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return None
        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self._viewer.is_running():
            self._viewer.sync()
        return None

    def close(self):
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None


def make_env(render_mode: str | None = None) -> SpiralEnv:
    cfg = EnvCfg()
    return SpiralEnv(render_mode=render_mode, cfg=cfg)
