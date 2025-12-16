from __future__ import annotations

import math
from typing import List

from generate_hex_mesh import ensure_unit_hex_mesh_obj, unit_y_span_over_x_span


def _compute_spiral_geometry(
    num_links: int,
    total_length: float,
    taper_angle_deg: float,
    delta_theta_deg: float,
    base_width_target: float,
) -> tuple[List[float], List[float], float, float, float]:
    delta_theta = math.radians(delta_theta_deg)

    b = 0.22 * (taper_angle_deg / 15.0)
    q = math.exp(b * delta_theta)

    if abs(q - 1.0) < 1e-9:
        q = 1.0 + 1e-6

    qn = q**num_links
    denom = qn - 1.0
    if denom <= 0.0:
        raise ValueError("invalid spiral denominator")

    tip_len = total_length * (q - 1.0) / denom
    lengths_base2tip = [tip_len * (q ** (num_links - 1 - i)) for i in range(num_links)]
    widths_base2tip = [base_width_target * (q ** (-i)) for i in range(num_links)]

    return lengths_base2tip, widths_base2tip, widths_base2tip[0], widths_base2tip[-1], q


def generate_spiral_tentacle_xml(
    num_links: int = 24,
    total_length: float = 0.45,
    taper_angle_deg: float = 15.0,
    delta_theta_deg: float = 30.0,
    base_width_target: float = 0.06,
    thickness: float = 0.01,
    link_density: float = 300.0,
    motor_gear: float = 1000.0,
    joint_damping: float = 0.4,
    joint_frictionloss: float = 0.02,
    distributed_cable: bool = True,
    cable_stiffness_per_seg: float = 0.0,
    cable_damping_per_seg: float = 0.0,
    cable_frictionloss_per_seg: float = 0.15,
    tension_max: float = 80.0,
    contact_layer_frac: float = 0.07,
) -> str:
    hex_path = ensure_unit_hex_mesh_obj("assets/hex_base.obj")

    verts_2d = [
        (0.156, 0.00),
        (0.10, 0.50),
        (-0.10, 0.50),
        (-0.25, 0.00),
        (-0.10, -0.50),
        (0.10, -0.50),
    ]
    y_over_x = unit_y_span_over_x_span(verts_2d)
    y_over_x = max(1e-9, float(y_over_x))

    link_lengths, link_widths, base_width, tip_width, q = _compute_spiral_geometry(
        num_links=num_links,
        total_length=total_length,
        taper_angle_deg=taper_angle_deg,
        delta_theta_deg=delta_theta_deg,
        base_width_target=base_width_target,
    )

    base_radius = base_width / 2.0
    tip_radius = tip_width / 2.0

    stiff_base = 40.0
    stiff_tip = 2.0
    stiff_ratio = (stiff_tip / stiff_base) ** (1.0 / max(num_links - 1, 1))
    joint_stiffness: List[float] = [stiff_base * (stiff_ratio**i) for i in range(num_links)]

    range_base = 40.0
    range_tip = 180.0
    joint_range_min: List[float] = []
    joint_range_max: List[float] = []
    for i in range(num_links):
        t = i / max(num_links - 1, 1)
        r = range_base * (1.0 - t) + range_tip * t
        joint_range_min.append(-r)
        joint_range_max.append(r)

    xml: List[str] = []

    xml.append('<mujoco model="spiral_tentacle_hex_spirob_2c">')
    xml.append('  <compiler angle="degree" coordinate="local" inertiafromgeom="true" boundmass="0.0001" boundinertia="0.0000001"/>')
    xml.append('  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" solver="Newton" iterations="60"/>')

    xml.append('  <default>')
    xml.append(f'    <joint damping="{joint_damping:.4f}" frictionloss="{joint_frictionloss:.4f}" springref="0"/>')
    xml.append('    <geom condim="4" solref="0.010 0.90" solimp="0.92 0.99 0.002" friction="1.2 0.03 0.001" margin="0"/>')
    xml.append('  </default>')

    xml.append('  <asset>')
    xml.append('    <material name="mat_tpu" rgba="0.85 0.85 0.88 1"/>')
    xml.append('    <material name="mat_floor" rgba="0.78 0.78 0.78 1"/>')
    xml.append('    <material name="mat_obj" rgba="0 1 0 1"/>')
    xml.append(f'    <mesh name="hex_base" file="{hex_path}"/>')
    for i in range(num_links):
        L = link_lengths[i]
        W = link_widths[i]
        sx = L
        sy = W / y_over_x
        sz = thickness
        xml.append(f'    <mesh name="hex_{i}" file="{hex_path}" scale="{sx:.6f} {sy:.6f} {sz:.6f}"/>')
    xml.append('  </asset>')

    xml.append('  <worldbody>')
    xml.append('    <geom name="floor" type="plane" size="2 2 0.1" material="mat_floor" friction="1.0 0.02 0.001" solref="0.012 0.95" solimp="0.95 0.99 0.002"/>')

    total_len = sum(link_lengths)
    R_approx = total_len / (2.0 * math.pi)

    base_x = -R_approx
    base_z = thickness * 1.5

    xml.append(f'    <body name="base" pos="{base_x:.6f} 0 {base_z:.6f}">')
    xml.append('      <geom type="sphere" size="0.03" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>')

    xml.append(f'      <site name="tendon_left_base"  pos="0 {base_radius:.6f} 0" size="0.002"/>')
    xml.append(f'      <site name="tendon_right_base" pos="0 {-base_radius:.6f} 0" size="0.002"/>')

    indent = "      "
    for i in range(num_links):
        L = link_lengths[i]
        W = link_widths[i]
        body_name = f"link_{i}"
        joint_name = f"joint_{i}"
        geom_name = f"link_geom_{i}"
        mesh_name = f"hex_{i}"

        if i == 0:
            pos_str = "0 0 0"
        else:
            prev_L = link_lengths[i - 1]
            pos_str = f"{prev_L:.6f} 0 0"

        margin = max(1e-4, contact_layer_frac * W)

        xml.append(f'{indent}<body name="{body_name}" pos="{pos_str}">')
        xml.append(
            f'{indent}  <joint name="{joint_name}" type="hinge" axis="0 0 1" pos="0 0 0" '
            f'range="{joint_range_min[i]:.1f} {joint_range_max[i]:.1f}" stiffness="{joint_stiffness[i]:.3f}"/>'
        )
        xml.append(
            f'{indent}  <geom name="{geom_name}" type="mesh" mesh="{mesh_name}" pos="0 0 0" density="{link_density:.1f}" '
            f'material="mat_tpu" friction="1.4 0.03 0.001" solref="0.010 0.90" solimp="0.92 0.99 0.002" condim="4" margin="{margin:.6f}"/>'
        )

        half_L = 0.5 * L
        t = i / max(num_links - 1, 1)
        arm_factor = 0.4 * (1.0 - t) + 0.9 * t
        y_off = 0.5 * W * arm_factor

        xml.append(f'{indent}  <site name="tendon_left_{i}"  pos="{half_L:.6f} {y_off:.6f} 0" size="0.002"/>')
        xml.append(f'{indent}  <site name="tendon_right_{i}" pos="{half_L:.6f} {-y_off:.6f} 0" size="0.002"/>')

        indent += "  "

    for _ in range(num_links):
        indent = indent[:-2]
        xml.append(f'{indent}</body>')
    xml.append('    </body>')

    obj_r = max(0.008, tip_radius * 1.2)
    xml.append(f'    <body name="obj_sphere" pos="{base_x + 0.65 * total_length:.6f} {0.18 * base_width_target:.6f} {obj_r:.6f}">')
    xml.append('      <joint name="obj_sphere_free" type="free"/>')
    xml.append(
        f'      <geom name="obj_sphere_geom" type="sphere" size="{obj_r:.6f}" density="300" '
        'friction="1.1 0.02 0.001" solref="0.010 0.90" solimp="0.92 0.99 0.002" condim="4" material="mat_obj"/>'
    )
    xml.append('    </body>')

    xml.append('  </worldbody>')

    xml.append('  <contact>')
    for i in range(num_links - 1):
        xml.append(f'    <exclude body1="link_{i}" body2="link_{i+1}"/>')
    xml.append('  </contact>')

    xml.append('  <tendon>')
    xml.append('    <spatial name="tendon_left" limited="false" width="0.002">')
    xml.append('      <site site="tendon_left_base"/>')
    for i in range(num_links):
        xml.append(f'      <site site="tendon_left_{i}"/>')
    xml.append('    </spatial>')

    xml.append('    <spatial name="tendon_right" limited="false" width="0.002">')
    xml.append('      <site site="tendon_right_base"/>')
    for i in range(num_links):
        xml.append(f'      <site site="tendon_right_{i}"/>')
    xml.append('    </spatial>')

    if distributed_cable:
        xml.append(
            f'    <spatial name="tendon_left_seg_0" limited="false" stiffness="{cable_stiffness_per_seg:.6f}" '
            f'damping="{cable_damping_per_seg:.6f}" frictionloss="{cable_frictionloss_per_seg:.6f}">'
            f'<site site="tendon_left_base"/><site site="tendon_left_0"/></spatial>'
        )
        xml.append(
            f'    <spatial name="tendon_right_seg_0" limited="false" stiffness="{cable_stiffness_per_seg:.6f}" '
            f'damping="{cable_damping_per_seg:.6f}" frictionloss="{cable_frictionloss_per_seg:.6f}">'
            f'<site site="tendon_right_base"/><site site="tendon_right_0"/></spatial>'
        )
        for i in range(1, num_links):
            xml.append(
                f'    <spatial name="tendon_left_seg_{i}" limited="false" stiffness="{cable_stiffness_per_seg:.6f}" '
                f'damping="{cable_damping_per_seg:.6f}" frictionloss="{cable_frictionloss_per_seg:.6f}">'
                f'<site site="tendon_left_{i-1}"/><site site="tendon_left_{i}"/></spatial>'
            )
            xml.append(
                f'    <spatial name="tendon_right_seg_{i}" limited="false" stiffness="{cable_stiffness_per_seg:.6f}" '
                f'damping="{cable_damping_per_seg:.6f}" frictionloss="{cable_frictionloss_per_seg:.6f}">'
                f'<site site="tendon_right_{i-1}"/><site site="tendon_right_{i}"/></spatial>'
            )

    xml.append('  </tendon>')

    xml.append('  <actuator>')
    xml.append(
        f'    <motor name="motor_left" tendon="tendon_left" gear="{motor_gear:.3f}" '
        f'ctrllimited="true" ctrlrange="-1 1" forcerange="{-tension_max:.6f} {tension_max:.6f}"/>'
    )
    xml.append(
        f'    <motor name="motor_right" tendon="tendon_right" gear="{motor_gear:.3f}" '
        f'ctrllimited="true" ctrlrange="-1 1" forcerange="{-tension_max:.6f} {tension_max:.6f}"/>'
    )

    if distributed_cable:
        for i in range(num_links):
            xml.append(
                f'    <motor name="motor_left_seg_{i}" tendon="tendon_left_seg_{i}" gear="{motor_gear:.3f}" '
                f'ctrllimited="true" ctrlrange="-1 1" forcerange="{-tension_max:.6f} {tension_max:.6f}"/>'
            )
            xml.append(
                f'    <motor name="motor_right_seg_{i}" tendon="tendon_right_seg_{i}" gear="{motor_gear:.3f}" '
                f'ctrllimited="true" ctrlrange="-1 1" forcerange="{-tension_max:.6f} {tension_max:.6f}"/>'
            )

    xml.append('  </actuator>')
    xml.append('</mujoco>')

    return "\n".join(xml)


if __name__ == "__main__":
    xml = generate_spiral_tentacle_xml()
    with open("spiral_tentacle_hex.xml", "w", encoding="utf-8") as f:
        f.write(xml)
    print("Saved spiral_tentacle_hex.xml")
