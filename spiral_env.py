# spiral_env.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np

from generate_spiral_xml import generate_spiral_tentacle_xml


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _safe_norm(v: np.ndarray, eps: float = 1e-9) -> float:
    return float(np.sqrt(float(np.dot(v, v)) + eps))


@dataclass
class DomainRandRanges:
    friction_link_floor_mu: Tuple[float, float] = (0.8, 1.6)
    friction_link_floor_tors: Tuple[float, float] = (0.002, 0.02)
    friction_link_floor_roll: Tuple[float, float] = (1e-6, 2e-4)

    friction_link_cube_mu: Tuple[float, float] = (0.9, 2.2)
    friction_link_cube_tors: Tuple[float, float] = (0.002, 0.03)
    friction_link_cube_roll: Tuple[float, float] = (1e-6, 3e-4)

    friction_link_link_mu: Tuple[float, float] = (0.6, 1.6)
    friction_link_link_tors: Tuple[float, float] = (0.001, 0.02)
    friction_link_link_roll: Tuple[float, float] = (1e-6, 2e-4)

    solref_link_link_timeconst: Tuple[float, float] = (0.006, 0.02)
    solref_link_link_dampratio: Tuple[float, float] = (0.7, 1.6)

    solref_link_cube_timeconst: Tuple[float, float] = (0.004, 0.015)
    solref_link_cube_dampratio: Tuple[float, float] = (0.7, 1.6)

    solimp_link_link_dmin: Tuple[float, float] = (0.8, 0.98)
    solimp_link_link_dmax: Tuple[float, float] = (0.95, 1.0)
    solimp_link_link_width: Tuple[float, float] = (0.001, 0.01)
    solimp_link_link_midpoint: Tuple[float, float] = (0.1, 0.5)
    solimp_link_link_power: Tuple[float, float] = (2.0, 4.0)

    solimp_link_cube_dmin: Tuple[float, float] = (0.85, 0.99)
    solimp_link_cube_dmax: Tuple[float, float] = (0.95, 1.0)
    solimp_link_cube_width: Tuple[float, float] = (0.001, 0.01)
    solimp_link_cube_midpoint: Tuple[float, float] = (0.1, 0.5)
    solimp_link_cube_power: Tuple[float, float] = (2.0, 4.0)

    cube_mass_scale: Tuple[float, float] = (0.8, 1.25)

    hinge_damping_scale: Tuple[float, float] = (0.8, 1.5)
    hinge_armature_scale: Tuple[float, float] = (0.8, 1.6)

    # FIX (4): narrow early-training DR envelopes for v_max and t_max
    v_max: Tuple[float, float] = (0.20, 0.35)
    t_max: Tuple[float, float] = (800.0, 1200.0)

    u_tau: Tuple[float, float] = (0.06, 0.16)
    du_max: Tuple[float, float] = (0.05, 0.18)
    enc_noise_std: Tuple[float, float] = (0.0, 0.0015)


class SpiralPickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        *,
        render_mode: Optional[str] = None,
        n_segments: int = 24,
        delta_deg: float = 30.0,
        psi_deg: float = 77.6,
        total_length: float = 0.45,
        tip_width: float = 0.0075,
        tip_thickness: float = 0.0024,
        lift_z: float = 0.010,
        tendon_offset_frac: float = 0.55,
        k_tip: float = 3.0,
        damping_mul: float = 1.2,
        frictionloss_tip: float = 0.012,
        armature_mul: float = 1.2,
        m_tip: float = 2e-6,
        motor_gear: float = 2600.0,
        timestep: float = 0.001,
        cube_half: float = 0.02,
        cube_density: float = 800.0,
        cube_friction: str = "1.1 0.02 0.0001",
        max_episode_steps: int = 900,
        substeps: int = 10,
        arena_half: float = 0.45,
        cube_spawn_xy_jitter: float = 0.06,
        goal_a_xy: Tuple[float, float] = (0.18, 0.12),
        goal_b_xy: Tuple[float, float] = (0.18, -0.12),
        goal_z: float = 0.02,
        goal_radius: float = 0.05,
        success_hold_steps: int = 30,
        grasp_min_contacts: int = 2,
        grasp_v_cube_max: float = 0.25,
        grasp_rel_v_tip_max: float = 0.25,
        grasp_max_z: float = 0.10,
        l_max: float = 0.18,
        ell_rest_bias: float = 0.0,
        kp: float = 200000.0,
        kd: float = 6000.0,
        dr_ranges: Optional[DomainRandRanges] = None,
        dr_enable: bool = True,
        w_reach: float = 1.6,
        w_grasp: float = 1.4,
        w_transport: float = 0.0,
        w_place: float = 0.0,
        ctrl_energy_cost: float = 0.0015,
        action_smooth_cost: float = 0.01,
        grasp_v_scale: float = 0.40,
        grasp_t_scale: float = 0.55,
        grasp_filter_tau: float = 0.12,
        vspool_penalty_coef: float = 0.05,
        tension_penalty_coef: float = 0.10,
        v_fly_threshold: float = 1.0,
        z_fly_threshold: float = 0.18,
        fly_penalty: float = 8.0,
        seed: Optional[int] = None,
        r_contact: float = 0.09,
        alpha_success: float = 0.50,
        v_soft: float = 0.22,
        w_d_reach: float = 1.0,
        # FIX (1): keep directional tip-velocity incentive effectively off in phase 0
        w_v_reach: float = 0.0,
        w_prog_reach: float = 0.05,
        prog_clip: float = 0.01,
        w_cb: float = 1.2,
        w_c: float = 0.10,
        w_vc: float = 0.45,
        w_T: float = 0.25,
        w_L: float = 0.12,
    ):
        super().__init__()
        self.render_mode = render_mode

        self.n_segments = int(n_segments)
        self.max_episode_steps = int(max_episode_steps)
        self.substeps = int(substeps)
        self.arena_half = float(arena_half)
        self.cube_spawn_xy_jitter = float(cube_spawn_xy_jitter)

        self.goal_a = np.array([goal_a_xy[0], goal_a_xy[1], goal_z], dtype=np.float64)
        self.goal_b = np.array([goal_b_xy[0], goal_b_xy[1], goal_z], dtype=np.float64)
        self.goal_radius = float(goal_radius)
        self.success_hold_steps = int(success_hold_steps)

        self.grasp_min_contacts = int(grasp_min_contacts)
        self.grasp_v_cube_max = float(grasp_v_cube_max)
        self.grasp_rel_v_tip_max = float(grasp_rel_v_tip_max)
        self.grasp_max_z = float(grasp_max_z)

        self.l_max = float(l_max)
        self.ell_rest_bias = float(ell_rest_bias)
        self.kp = float(kp)
        self.kd = float(kd)

        self.dr_ranges = dr_ranges if dr_ranges is not None else DomainRandRanges()
        self.dr_enable = bool(dr_enable)

        self.w_reach = float(w_reach)
        self.w_grasp = float(w_grasp)
        self.w_transport = float(w_transport)
        self.w_place = float(w_place)
        self.ctrl_energy_cost = float(ctrl_energy_cost)
        self.action_smooth_cost = float(action_smooth_cost)

        self.grasp_v_scale = float(grasp_v_scale)
        self.grasp_t_scale = float(grasp_t_scale)
        self.grasp_filter_tau = float(grasp_filter_tau)

        self.vspool_penalty_coef = float(vspool_penalty_coef)
        self.tension_penalty_coef = float(tension_penalty_coef)

        self.v_fly_threshold = float(v_fly_threshold)
        self.z_fly_threshold = float(z_fly_threshold)
        self.fly_penalty = float(fly_penalty)

        self.r_contact = float(r_contact)
        self.alpha_success = float(alpha_success)
        self.v_soft = float(v_soft)

        self.w_d_reach = float(w_d_reach)
        self.w_v_reach = float(w_v_reach)
        self.w_prog_reach = float(w_prog_reach)
        self.prog_clip = float(prog_clip)

        self.w_cb = float(w_cb)
        self.w_c = float(w_c)
        self.w_vc = float(w_vc)
        self.w_T = float(w_T)
        self.w_L = float(w_L)

        # FIX (3): phase-0 impulse penalties as scaled versions of phase-1 penalties
        self._phase0_impulse_penalty_scale = 0.30

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            self.np_random, _ = gym.utils.seeding.np_random(None)

        xml = generate_spiral_tentacle_xml(
            n_segments=self.n_segments,
            delta_deg=float(delta_deg),
            psi_deg=float(psi_deg),
            total_length=float(total_length),
            tip_width=float(tip_width),
            tip_thickness=float(tip_thickness),
            lift_z=float(lift_z),
            tendon_offset_frac=float(tendon_offset_frac),
            k_tip=float(k_tip),
            damping_mul=float(damping_mul),
            frictionloss_tip=float(frictionloss_tip),
            armature_mul=float(armature_mul),
            m_tip=float(m_tip),
            motor_gear=float(motor_gear),
            timestep=float(timestep),
            cube_half=float(cube_half),
            cube_density=float(cube_density),
            cube_friction=str(cube_friction),
            cube_seed=None,
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        if self.model.nu != 2:
            raise RuntimeError(f"Expected nu=2, got nu={self.model.nu}")

        self.act_id_left = int(self.model.actuator("motor_left").id)
        self.act_id_right = int(self.model.actuator("motor_right").id)

        self.ten_id_left = int(self.model.tendon("tendon_left").id)
        self.ten_id_right = int(self.model.tendon("tendon_right").id)

        self.body_id_base = int(self.model.body("link_00").id)
        self.body_id_tip = int(self.model.body(f"link_{self.n_segments - 1:02d}").id)

        self.body_id_cube = int(self.model.body("cube").id)
        self.geom_id_floor = int(self.model.geom("floor").id)
        self.geom_id_cube = int(self.model.geom("cube_geom").id)

        self.geom_ids_links = [int(self.model.geom(f"geom_{i:02d}").id) for i in range(self.n_segments)]

        cube_jadr = int(self.model.body_jntadr[self.body_id_cube])
        cube_jnum = int(self.model.body_jntnum[self.body_id_cube])
        if cube_jnum < 1:
            raise RuntimeError("Cube body has no joints, expected a freejoint.")
        cube_jid = cube_jadr
        if int(self.model.jnt_type[cube_jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise RuntimeError("Cube first joint is not a free joint, expected <freejoint/>.")
        self.cube_qpos_adr = int(self.model.jnt_qposadr[cube_jid])
        self.cube_dof_adr = int(self.model.jnt_dofadr[cube_jid])

        self.hinge_dof_ids = []
        for j in range(int(self.model.njnt)):
            if int(self.model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE):
                dof = int(self.model.jnt_dofadr[j])
                if 0 <= dof < int(self.model.nv):
                    self.hinge_dof_ids.append(dof)

        self._gear = np.zeros(2, dtype=np.float64)
        self._ctrl_lo = np.zeros(2, dtype=np.float64)
        self._ctrl_hi = np.ones(2, dtype=np.float64)

        for k, act_id in enumerate([self.act_id_left, self.act_id_right]):
            g = float(self.model.actuator_gear[act_id, 0])
            if not np.isfinite(g) or abs(g) < 1e-9:
                g = 1.0
            self._gear[k] = g

            lo = float(self.model.actuator_ctrlrange[act_id, 0])
            hi = float(self.model.actuator_ctrlrange[act_id, 1])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = 0.0, 1.0
            self._ctrl_lo[k] = lo
            self._ctrl_hi[k] = hi

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)

        obs_dim = self.nq + self.nv + 9 + 9 + 3 + 6 + 4 + 3
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._viewer = None

        self._step = 0
        self._phase = 0

        self._L = np.zeros(2, dtype=np.float64)
        self._v = np.zeros(2, dtype=np.float64)
        self._u_prev = np.zeros(2, dtype=np.float64)
        self._u_filt = np.zeros(2, dtype=np.float64)

        self._v_max = 0.35
        self._t_max = 1100.0
        self._u_tau = 0.10
        self._du_max = 0.12
        self._enc_noise = 0.0

        self._v_max_base = self._v_max
        self._t_max_base = self._t_max

        self._grasp_v_scale_state = 1.0
        self._grasp_t_scale_state = 1.0

        self._ell_prev = np.zeros(2, dtype=np.float64)
        self._ell_rest = np.zeros(2, dtype=np.float64)

        self._e_prev = np.zeros(2, dtype=np.float64)

        self._T_last = np.zeros(2, dtype=np.float64)

        self._prev_tip_to_cube = None
        self._last_action_u = np.zeros(2, dtype=np.float64)

        self._cube_half = float(self.model.geom_size[self.geom_id_cube, 0])
        if not np.isfinite(self._cube_half) or self._cube_half <= 0.0:
            self._cube_half = float(cube_half)

        self._dt = float(self.model.opt.timestep) * float(self.substeps)
        if not np.isfinite(self._dt) or self._dt <= 0.0:
            self._dt = 0.001 * float(self.substeps)

        self._mj_min = float(getattr(mujoco, "mjMINVAL", 1e-9))

        self._initial_cube_to_base = 0.0
        self._L_prev_for_reward = np.zeros(2, dtype=np.float64)

    def _pos_base(self) -> np.ndarray:
        return self.data.xpos[self.body_id_base].copy()

    def _pos_tip(self) -> np.ndarray:
        return self.data.xpos[self.body_id_tip].copy()

    def _pos_cube(self) -> np.ndarray:
        return self.data.xpos[self.body_id_cube].copy()

    def _vel_cube_lin(self) -> np.ndarray:
        return self.data.cvel[self.body_id_cube, 3:].copy()

    def _vel_tip_lin(self) -> np.ndarray:
        return self.data.cvel[self.body_id_tip, 3:].copy()

    def _contact_stats(self) -> Dict[str, Any]:
        link_touch = set()
        cube_floor = False

        ncon = int(self.data.ncon)
        for i in range(ncon):
            c = self.data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)

            cube_involved = (g1 == self.geom_id_cube) or (g2 == self.geom_id_cube)
            if not cube_involved:
                continue

            other = g2 if g1 == self.geom_id_cube else g1
            if other == self.geom_id_floor:
                cube_floor = True
                continue

            if other in self.geom_ids_links:
                link_touch.add(other)

        return {"num_link_contacts": int(len(link_touch)), "cube_on_floor": bool(cube_floor)}

    def _is_grasp(self, stats: Dict[str, Any]) -> Tuple[bool, Dict[str, float]]:
        cube_pos = self._pos_cube()
        cube_v = self._vel_cube_lin()
        tip_v = self._vel_tip_lin()
        rel_v = cube_v - tip_v

        cube_speed = float(np.linalg.norm(cube_v))
        rel_speed = float(np.linalg.norm(rel_v))
        z_ok = float(cube_pos[2]) <= float(self.grasp_max_z)

        good_contacts = int(stats["num_link_contacts"]) >= int(self.grasp_min_contacts)
        speed_ok = (cube_speed <= float(self.v_soft)) and (rel_speed <= float(self.grasp_rel_v_tip_max))

        grasp = bool(good_contacts and speed_ok and z_ok)
        return grasp, {"cube_speed": cube_speed, "rel_speed": rel_speed, "z": float(cube_pos[2])}

    def _tendon_lengths(self) -> np.ndarray:
        ell = np.array(
            [float(self.data.ten_length[self.ten_id_left]), float(self.data.ten_length[self.ten_id_right])],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(ell)):
            ell[:] = 0.0
        return ell

    def _update_grasp_limits(self, contact_hint: bool) -> Tuple[float, float]:
        target_v = self.grasp_v_scale if contact_hint else 1.0
        target_t = self.grasp_t_scale if contact_hint else 1.0
        tau = max(1e-6, self.grasp_filter_tau)
        alpha = _clip(self._dt / tau, 0.0, 1.0)
        self._grasp_v_scale_state += alpha * (target_v - self._grasp_v_scale_state)
        self._grasp_t_scale_state += alpha * (target_t - self._grasp_t_scale_state)
        v_eff = self._v_max_base * float(self._grasp_v_scale_state)
        t_eff = self._t_max_base * float(self._grasp_t_scale_state)
        return v_eff, t_eff

    def _update_spools_and_ctrl(self, u_cmd: np.ndarray, contact_hint: bool) -> None:
        u = np.clip(u_cmd.astype(np.float64), -1.0, 1.0)

        if self._u_tau > 1e-6:
            alpha = _clip(self._dt / self._u_tau, 0.0, 1.0)
        else:
            alpha = 1.0
        u_f = self._u_filt + alpha * (u - self._u_filt)

        du = np.clip(u_f - self._u_prev, -self._du_max, self._du_max)
        u_rl = np.clip(self._u_prev + du, -1.0, 1.0)

        self._u_filt[:] = u_rl
        self._u_prev[:] = u_rl

        v_eff, t_eff = self._update_grasp_limits(contact_hint)
        v_des = v_eff * u_rl
        self._v[:] = v_des
        self._L[:] = np.clip(self._L + self._dt * self._v, 0.0, self.l_max)

        ell = self._tendon_lengths()
        ell_meas = ell.copy()
        if self._enc_noise > 0.0:
            ell_meas += self.np_random.normal(0.0, self._enc_noise, size=(2,)).astype(np.float64)

        ell_des = (self._ell_rest - self._L) + self.ell_rest_bias

        e = ell_meas - ell_des
        de = (e - self._e_prev) / max(self._dt, 1e-9)
        self._e_prev[:] = e

        T = self.kp * e + self.kd * de
        T = np.clip(T, 0.0, t_eff)
        self._T_last[:] = T

        ctrl = np.zeros(2, dtype=np.float64)
        for k in range(2):
            g = self._gear[k]
            raw = T[k] / g
            ctrl[k] = _clip(float(raw), float(self._ctrl_lo[k]), float(self._ctrl_hi[k]))

        self.data.ctrl[self.act_id_left] = float(ctrl[0])
        self.data.ctrl[self.act_id_right] = float(ctrl[1])

        self._ell_prev[:] = ell_meas

    def _rand_uniform(self, lo: float, hi: float) -> float:
        return float(self.np_random.uniform(lo, hi))

    def _apply_domain_randomization(self) -> None:
        rr = self.dr_ranges

        mu_lf = self._rand_uniform(*rr.friction_link_floor_mu)
        t_lf = self._rand_uniform(*rr.friction_link_floor_tors)
        r_lf = self._rand_uniform(*rr.friction_link_floor_roll)

        mu_lc = self._rand_uniform(*rr.friction_link_cube_mu)
        t_lc = self._rand_uniform(*rr.friction_link_cube_tors)
        r_lc = self._rand_uniform(*rr.friction_link_cube_roll)

        mu_ll = self._rand_uniform(*rr.friction_link_link_mu)
        t_ll = self._rand_uniform(*rr.friction_link_link_tors)
        r_ll = self._rand_uniform(*rr.friction_link_link_roll)

        for gid in self.geom_ids_links:
            self.model.geom_friction[gid, 0] = mu_ll
            self.model.geom_friction[gid, 1] = t_ll
            self.model.geom_friction[gid, 2] = r_ll

        self.model.geom_friction[self.geom_id_floor, 0] = mu_lf
        self.model.geom_friction[self.geom_id_floor, 1] = t_lf
        self.model.geom_friction[self.geom_id_floor, 2] = r_lf

        self.model.geom_friction[self.geom_id_cube, 0] = mu_lc
        self.model.geom_friction[self.geom_id_cube, 1] = t_lc
        self.model.geom_friction[self.geom_id_cube, 2] = r_lc

        ll_time = self._rand_uniform(*rr.solref_link_link_timeconst)
        ll_damp = self._rand_uniform(*rr.solref_link_link_dampratio)
        lc_time = self._rand_uniform(*rr.solref_link_cube_timeconst)
        lc_damp = self._rand_uniform(*rr.solref_link_cube_dampratio)

        ll_solimp = np.array(
            [
                self._rand_uniform(*rr.solimp_link_link_dmin),
                self._rand_uniform(*rr.solimp_link_link_dmax),
                self._rand_uniform(*rr.solimp_link_link_width),
                self._rand_uniform(*rr.solimp_link_link_midpoint),
                self._rand_uniform(*rr.solimp_link_link_power),
            ],
            dtype=np.float64,
        )
        lc_solimp = np.array(
            [
                self._rand_uniform(*rr.solimp_link_cube_dmin),
                self._rand_uniform(*rr.solimp_link_cube_dmax),
                self._rand_uniform(*rr.solimp_link_cube_width),
                self._rand_uniform(*rr.solimp_link_cube_midpoint),
                self._rand_uniform(*rr.solimp_link_cube_power),
            ],
            dtype=np.float64,
        )

        for gid in self.geom_ids_links:
            self.model.geom_solref[gid, 0] = ll_time
            self.model.geom_solref[gid, 1] = ll_damp
            self.model.geom_solimp[gid, :] = ll_solimp

        self.model.geom_solref[self.geom_id_cube, 0] = lc_time
        self.model.geom_solref[self.geom_id_cube, 1] = lc_damp
        self.model.geom_solimp[self.geom_id_cube, :] = lc_solimp

        bid = self.body_id_cube
        base_mass = float(self.model.body_mass[bid])
        if not np.isfinite(base_mass) or base_mass <= self._mj_min:
            base_mass = 0.02

        s = self._rand_uniform(*rr.cube_mass_scale)
        new_mass = max(self._mj_min, base_mass * s)
        self.model.body_mass[bid] = new_mass

        I = self.model.body_inertia[bid].copy()
        I = np.maximum(I, self._mj_min)
        self.model.body_inertia[bid, :] = np.maximum(self._mj_min, I * s)

        damp_scale = self._rand_uniform(*rr.hinge_damping_scale)
        arm_scale = self._rand_uniform(*rr.hinge_armature_scale)
        for dof in self.hinge_dof_ids:
            self.model.dof_damping[dof] = max(self._mj_min, float(self.model.dof_damping[dof]) * damp_scale)
            self.model.dof_armature[dof] = max(self._mj_min, float(self.model.dof_armature[dof]) * arm_scale)

        self._v_max = self._rand_uniform(*rr.v_max)
        self._t_max = self._rand_uniform(*rr.t_max)
        self._u_tau = self._rand_uniform(*rr.u_tau)
        self._du_max = self._rand_uniform(*rr.du_max)
        self._enc_noise = self._rand_uniform(*rr.enc_noise_std)

        self._v_max_base = self._v_max
        self._t_max_base = self._t_max

    def _phase_update(self, stats: Dict[str, Any], dist_tip_cube: float) -> None:
        if self._phase == 0:
            if int(stats["num_link_contacts"]) >= self.grasp_min_contacts and dist_tip_cube < self.r_contact:
                self._phase = 1
        else:
            if int(stats["num_link_contacts"]) == 0 and dist_tip_cube > (self.r_contact * 1.6):
                self._phase = 0

    def _compute_reward(
        self,
        action_u: np.ndarray,
        ctrl_energy: float,
        stats: Dict[str, Any],
        grasp: bool,
        grasp_metrics: Dict[str, float],
    ) -> Tuple[float, Dict[str, Any]]:
        base = self._pos_base()
        tip = self._pos_tip()
        cube = self._pos_cube()
        v_cube = self._vel_cube_lin()
        v_tip = self._vel_tip_lin()

        rel_cube_tip = cube - tip
        dist_tip_cube = _safe_norm(rel_cube_tip)
        dir_to_cube = rel_cube_tip / max(dist_tip_cube, 1e-9)

        dist_cube_base = _safe_norm(cube - base)
        cube_speed = float(np.linalg.norm(v_cube))
        cube_z = float(cube[2])

        if self._prev_tip_to_cube is None:
            self._prev_tip_to_cube = dist_tip_cube

        reach_progress = float(self._prev_tip_to_cube - dist_tip_cube)
        reach_progress = _clip(reach_progress, -self.prog_clip, self.prog_clip)

        self._phase_update(stats, dist_tip_cube)

        phase_onehot = np.zeros(4, dtype=np.float32)
        phase_onehot[int(self._phase)] = 1.0

        r = 0.0

        t_sum = float(self._T_last[0] + self._T_last[1])
        t_scale = max(1e-6, self._t_max_base)
        v_norm = float(np.linalg.norm(self._v))
        v_scale = max(1e-6, self._v_max_base)
        dL = float(np.sum(np.abs(self._L - self._L_prev_for_reward)))

        if self._phase == 0:
            r += -self.w_d_reach * dist_tip_cube
            # FIX (1): remove directed tip-velocity incentive in phase 0
            # (keep parameter for API compatibility, but do not apply it here)
            r += self.w_prog_reach * reach_progress

            # FIX (3): apply weak impulse penalties already in phase 0
            s0 = float(self._phase0_impulse_penalty_scale)
            r -= s0 * self.vspool_penalty_coef * (v_norm / v_scale)
            r -= s0 * self.tension_penalty_coef * (t_sum / t_scale)
            r += -s0 * self.w_L * dL
            r += -s0 * self.w_T * (t_sum / t_scale)
        else:
            k = float(int(stats["num_link_contacts"]))
            r += -self.w_cb * dist_cube_base
            r += self.w_c * k
            r += -self.w_vc * cube_speed
            r += -self.w_T * (t_sum / t_scale)
            r += -self.w_L * dL
            r -= self.vspool_penalty_coef * (v_norm / v_scale)
            r -= self.tension_penalty_coef * (t_sum / t_scale)

        flew = (cube_speed > self.v_fly_threshold) or (cube_z > self.z_fly_threshold)
        if flew:
            r -= self.fly_penalty

        r -= self.ctrl_energy_cost * ctrl_energy
        r -= self.action_smooth_cost * float(np.sum(np.square(action_u - self._last_action_u)))

        if not np.isfinite(r):
            r = -10.0

        if abs(float(cube[0])) > self.arena_half or abs(float(cube[1])) > self.arena_half:
            r -= 6.0

        if float(cube[2]) > 0.25:
            r -= 2.0 * (float(cube[2]) - 0.25)

        self._prev_tip_to_cube = dist_tip_cube

        info = {
            "phase": int(self._phase),
            "phase_onehot": phase_onehot,
            "dist_tip_to_cube": float(dist_tip_cube),
            "dist_cube_to_base": float(dist_cube_base),
            "cube_speed": float(cube_speed),
            "cube_z": float(cube_z),
            "num_contacts": int(stats["num_link_contacts"]),
            "cube_on_floor": bool(stats["cube_on_floor"]),
            "in_grasp": bool(grasp),
            "fly_event": bool(flew),
            "t_left": float(self._T_last[0]),
            "t_right": float(self._T_last[1]),
            "dL": float(dL),
        }
        return float(r), info

    def _get_obs(self, phase_onehot: np.ndarray, grasp: bool, stats: Dict[str, Any]) -> np.ndarray:
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()

        base = self._pos_base()
        tip = self._pos_tip()
        cube = self._pos_cube()

        v_cube = self._vel_cube_lin()

        rel_cube_tip = cube - tip
        rel_goalb_cube = self.goal_b - cube
        rel_goala_cube = self.goal_a - cube

        aggregates = np.array(
            [
                1.0 if grasp else 0.0,
                float(int(stats["num_link_contacts"])),
                1.0 if bool(stats["cube_on_floor"]) else 0.0,
            ],
            dtype=np.float32,
        )

        drive = np.concatenate([self._L, self._v, self._u_filt], axis=0).astype(np.float32)

        obs = np.concatenate(
            [
                qpos.astype(np.float32),
                qvel.astype(np.float32),
                base.astype(np.float32),
                tip.astype(np.float32),
                cube.astype(np.float32),
                rel_cube_tip.astype(np.float32),
                rel_goalb_cube.astype(np.float32),
                rel_goala_cube.astype(np.float32),
                v_cube.astype(np.float32),
                drive,
                phase_onehot.astype(np.float32),
                aggregates,
            ],
            axis=0,
        )

        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        mujoco.mj_resetData(self.model, self.data)

        self._step = 0
        self._phase = 0

        self._L[:] = 0.0
        self._v[:] = 0.0
        self._u_prev[:] = 0.0
        self._u_filt[:] = 0.0

        self._grasp_v_scale_state = 1.0
        self._grasp_t_scale_state = 1.0

        self._prev_tip_to_cube = None
        self._last_action_u[:] = 0.0

        self._L_prev_for_reward[:] = self._L

        if self.dr_enable:
            self._apply_domain_randomization()

        qpos_noise = self.np_random.normal(0.0, 0.015, size=(self.nq,)).astype(np.float64)
        self.data.qpos[:] = self.data.qpos[:] + qpos_noise
        self.data.qvel[:] = 0.0

        xy = self.goal_a[:2].copy()
        xy += self.np_random.uniform(-self.cube_spawn_xy_jitter, self.cube_spawn_xy_jitter, size=(2,)).astype(np.float64)
        xy[0] = _clip(float(xy[0]), -self.arena_half * 0.8, self.arena_half * 0.8)
        xy[1] = _clip(float(xy[1]), -self.arena_half * 0.8, self.arena_half * 0.8)
        z = max(self._cube_half + 0.003, float(self.goal_a[2]))

        self.data.qpos[self.cube_qpos_adr + 0] = float(xy[0])
        self.data.qpos[self.cube_qpos_adr + 1] = float(xy[1])
        self.data.qpos[self.cube_qpos_adr + 2] = float(z)

        self.data.qpos[self.cube_qpos_adr + 3 : self.cube_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qvel[self.cube_dof_adr : self.cube_dof_adr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)

        ell = self._tendon_lengths()
        self._ell_rest[:] = ell
        self._ell_prev[:] = ell

        ell_des = (self._ell_rest - self._L) + self.ell_rest_bias
        self._e_prev[:] = (ell - ell_des)

        self._T_last[:] = 0.0

        stats = self._contact_stats()
        grasp, _ = self._is_grasp(stats)

        base = self._pos_base()
        cube = self._pos_cube()
        self._initial_cube_to_base = float(_safe_norm(cube - base))

        phase_oh = np.zeros(4, dtype=np.float32)
        phase_oh[0] = 1.0
        obs = self._get_obs(phase_oh, grasp, stats)

        info = {
            "phase": int(self._phase),
            "success": False,
            "dist_tip_to_cube": float(_safe_norm(self._pos_cube() - self._pos_tip())),
            "dist_cube_to_base": float(_safe_norm(self._pos_cube() - self._pos_base())),
            "num_contacts": int(stats["num_link_contacts"]),
            "cube_speed": float(np.linalg.norm(self._vel_cube_lin())),
            "in_grasp": bool(grasp),
            "t_left": float(self._T_last[0]),
            "t_right": float(self._T_last[1]),
        }
        return obs, info

    def step(self, action: np.ndarray):
        self._step += 1

        u = np.clip(np.asarray(action, dtype=np.float64).reshape(2,), -1.0, 1.0)

        # FIX (2): remove 1-step lag by using contacts from the current (pre-control) state
        stats_pre = self._contact_stats()
        contact_hint = bool(int(stats_pre["num_link_contacts"]) >= 1)

        self._update_spools_and_ctrl(u, contact_hint=contact_hint)
        ctrl_energy = float(np.sum(np.square(self.data.ctrl[:2])))

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
                break

        stats_post = self._contact_stats()
        grasp, grasp_metrics = self._is_grasp(stats_post)

        reward, info = self._compute_reward(u, ctrl_energy, stats_post, grasp, grasp_metrics)

        dist_cube_base = float(info["dist_cube_to_base"])
        cube_speed = float(info["cube_speed"])
        contacts_ok = int(info["num_contacts"]) >= int(self.grasp_min_contacts)
        speed_ok = cube_speed < float(self.v_soft)
        closer_ok = dist_cube_base < float(self.alpha_success) * max(1e-9, float(self._initial_cube_to_base))
        success = bool(closer_ok and speed_ok and contacts_ok)

        terminated = bool(success)

        cube = self._pos_cube()
        out_of_arena = (abs(float(cube[0])) > self.arena_half) or (abs(float(cube[1])) > self.arena_half)
        flew = float(cube[2]) > 0.35
        nan_state = (not np.all(np.isfinite(self.data.qpos))) or (not np.all(np.isfinite(self.data.qvel)))

        truncated = False
        if self._step >= self.max_episode_steps:
            truncated = True
        if out_of_arena or flew or nan_state:
            truncated = True
            reward -= 10.0

        info.update(
            {
                "success": bool(success),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "drive_v_max": float(self._v_max_base),
                "drive_t_max": float(self._t_max_base),
                "drive_tau": float(self._u_tau),
                "drive_du_max": float(self._du_max),
                "enc_noise": float(self._enc_noise),
                "v_spool": float(np.linalg.norm(self._v)),
                "initial_dist_cube_to_base": float(self._initial_cube_to_base),
            }
        )

        self._last_action_u = u.copy()
        self._L_prev_for_reward[:] = self._L

        obs = self._get_obs(info["phase_onehot"], bool(grasp), stats_post)

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            time.sleep(0.02)
        else:
            self._viewer.sync()

    def close(self):
        self._viewer = None
