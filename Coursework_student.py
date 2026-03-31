import numpy as np
import math
import os
from simulator_setup import setup_simulator
from plotting import plot_results

# Load simulator setup
setup_data = setup_simulator()
sim = setup_data['sim']
plant = setup_data['plant']
plant_context = setup_data['plant_context']
j_motor1 = setup_data['j_motor1']
j_motor2 = setup_data['j_motor2']
j_spring1 = setup_data['j_spring1']
j_spring2 = setup_data['j_spring2']
j_wall = setup_data['j_wall']
L1 = setup_data['L1']
L2 = setup_data['L2']
phi1_init = setup_data['phi1_init']
phi2_init = setup_data['phi2_init']
k1 = setup_data['k1']
k2 = setup_data['k2']
wall_x = setup_data['wall_x']
wall_thickness = setup_data['wall_thickness']
wall_k = setup_data['wall_k']
wall_damping_extra = setup_data['wall_damping_extra']
wall_fric_normal = setup_data['wall_fric_normal']
wall_mu_static = setup_data['wall_mu_static']
wall_mu_coulomb = setup_data['wall_mu_coulomb']
wall_v_stiction = setup_data['wall_v_stiction']
TORQUE_LIMIT = setup_data['TORQUE_LIMIT']
USE_FALSE_TRAJ = setup_data['USE_FALSE_TRAJ']
PRINT_SUMMARY = setup_data['PRINT_SUMMARY']

jl1 = j_motor1
jl2 = j_motor2
jdel1 = j_spring1
jdel2 = j_spring2

# ============================================
# SYSTEM PARAMETERS
# ============================================
# Spring stiffness vector (for converting between torque and deflection)
K_spring = np.array([k1, k2])

# Link/rotor masses (must match simulator_setup.py)
m1, m2 = 1.5, 1.5
m_rotor2 = 0.4       # rotor2 mass at joint 2 pivot
g = 9.81
lc1, lc2 = L1 / 2.0, L2 / 2.0   # centre-of-mass offsets

# ============================================
# TRAJECTORY PARAMETERS
# ============================================
# Wall-drawing region
z_top = 1.55          # top of the vertical line
z_bot = 0.10          # bottom of the vertical line

# Phase timing
T_HOLD     = 0.5      # Phase 1: hold initial pose
T_APPROACH = 1.5      # Phase 2: move to approach config (ends at T_HOLD + T_APPROACH)
T_CONTACT  = 2.5      # Phase 3: make gentle contact (ends at T_HOLD+T_APPROACH+T_CONTACT)
T_DRAW     = 40.0     # Phase 4: draw the line (slow descent)
T_final    = T_HOLD + T_APPROACH + T_CONTACT + T_DRAW  # total sim time

# Contact force magnitude (N) applied in x-direction via J^T
F_CONTACT_X = 15.0    # enough to maintain contact, not enough to shove wall

# ============================================
# CONTROLLER GAINS
# ============================================

# --- Outer loop: link-side PD (computes desired spring torque) ---
# These gains act on the link position/velocity errors.
# Higher = stiffer tracking, but risk oscillation with the springs.
Kp_free = np.array([20.0, 15.0])    # proportional gains for free-space motion
Kd_free = np.array([0.4, 0.4])    # derivative gains for free-space motion

Kp_draw = np.array([80.0, 75.0])   # proportional gains during wall contact/drawing
Kd_draw = np.array([28.0, 18.0])    # derivative gains during wall contact/drawing

# --- Inner loop: motor-side PD (tracks desired motor angle) ---
# These gains compensate for the motor inertia and damping.
# They should be modest: too high causes motor-spring resonance.
Ktheta = np.array([5.0, 3.0])       # motor position gain
Dtheta = np.array([0.5, 0.3])       # motor velocity damping

# Saturation limit for desired spring torque (before converting to theta_des)
TAU_S_MAX = 2000.0


# ============================================
# CONTROLLER STATE (global, persists across calls)
# ============================================
_init = False
_q_hold = None        # initial link-side pose (captured at t=0)
_q_target = None      # approach configuration (near wall, at z_top)
_ws = None            # wall surface x-coordinate at init
_phase_prev = -1


# ============================================
# KINEMATICS & DYNAMICS HELPERS
# ============================================

def forward_kinematics(q1, q2):
    """Compute end-effector [x, z] from link angles."""
    x = L1 * math.cos(q1) + L2 * math.cos(q1 + q2)
    z = -L1 * math.sin(q1) - L2 * math.sin(q1 + q2)
    return np.array([x, z])


def inverse_kinematics(x_target, z_target, elbow_up=False):
    """
    Analytical IK for 2-link planar arm.
    elbow_up=False gives q2 > 0 (elbow-down), suitable for reaching right and below.
    """
    dist_sq = x_target**2 + z_target**2
    dist = math.sqrt(dist_sq)

    if dist > (L1 + L2) or dist < abs(L1 - L2):
        print("Target out of reach!")
        return None

    cos_q2 = (dist_sq - L1**2 - L2**2) / (2 * L1 * L2)
    cos_q2 = max(-1.0, min(1.0, cos_q2))

    if elbow_up:
        q2 = -math.acos(cos_q2)
    else:
        q2 = math.acos(cos_q2)

    alpha = math.atan2(z_target, x_target)
    beta = math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    q1 = alpha - beta

    return np.array([q1, q2])


def jacobian(q1, q2):
    """
    Jacobian J such that [xdot, zdot]^T = J * [q1dot, q2dot]^T.
    """
    s1 = math.sin(q1)
    c1 = math.cos(q1)
    s12 = math.sin(q1 + q2)
    c12 = math.cos(q1 + q2)
    return np.array([
        [-L1 * s1 - L2 * s12, -L2 * s12],
        [ L1 * c1 + L2 * c12,  L2 * c12],
    ])


def gravity_compensation(q1, q2):
    """
    Compute gravity torque vector for the 2-link arm.
    
    This returns the torque needed to HOLD the arm against gravity,
    i.e., tau_g such that the arm is in static equilibrium when tau = tau_g.
    
    For the Drake model (x-z plane, gravity in -z):
        tau_g1 = g*cos(q1)*(m1*lc1 + m_rotor2*L1 + m2*L1) + m2*g*lc2*cos(q1+q2)
        tau_g2 = m2*g*lc2*cos(q1+q2)
    """
    c1 = math.cos(q1)
    c12 = math.cos(q1 + q2)

    tauG1 = (g * c1 * (m1 * lc1 + m_rotor2 * L1 + m2 * L1)
             + m2 * g * lc2 * c12)
    tauG2 = m2 * g * lc2 * c12
    return np.array([tauG1, tauG2])


def smoothstep(t, T):
    """Hermite smoothstep: 0 at t<=0, 1 at t>=T, smooth in between."""
    if t <= 0.0:
        return 0.0
    if t >= T:
        return 1.0
    u = t / T
    return u * u * (3.0 - 2.0 * u)


# ============================================
# CASCADED COMPLIANT-JOINT CONTROLLER
# ============================================
#
# Architecture:
#
#   Link-side dynamics:    M(q)q̈ + C(q,q̇)q̇ + K(q-θ) + τ_g(q) = 0
#   Motor-side dynamics:   Jθ̈ + Dθ̇ + K(θ-q) = τ_m
#
# The spring torque on the link is τ_spring = K(θ-q) = -K·δ  (where δ = q - θ = spring_delta)
# The spring torque on the motor is K(q-θ) = K·δ
#
# OUTER LOOP (link-side):
#   Computes the desired spring torque τ_s_des that the springs should exert
#   on the links to achieve the desired motion:
#       τ_s_des = τ_g(q_des) + Kp*(q_des - q) + Kd*(q̇_des - q̇) + J^T * F_contact
#
#   This is essentially: "what torque do the springs need to produce to make
#   the links follow the trajectory?"
#
# CONVERSION (τ_s → θ_des):
#   Since τ_spring = K(θ - q), to get the desired spring torque:
#       τ_s_des = K(θ_des - q_des)
#       θ_des = q_des + τ_s_des / K
#
# INNER LOOP (motor-side):
#   Tracks the desired motor angle θ_des:
#       τ_m = τ_s_des + Kθ*(θ_des - θ) - Dθ*θ̇
#
#   The τ_s_des term is a feedforward (the motor must produce at least this
#   much torque to maintain the spring deflection). The PD terms correct
#   tracking errors on the motor side.
#

def controller(q, qd, theta, theta_dot, delta, deltad, t, wall_pos):
    """
    Cascaded compliant-joint controller.
    
    Args:
        q:          Link-side joint angles [q1, q2]
        qd:         Link-side joint velocities [q1dot, q2dot]  
        theta:      Motor-side joint angles [θ1, θ2]
        theta_dot:  Motor-side joint velocities [θ1dot, θ2dot]
        delta:      Spring deflections [δ1, δ2] = q - θ
        deltad:     Spring deflection rates [δ1dot, δ2dot]
        t:          Current simulation time
        wall_pos:   Wall prismatic joint position
        
    Returns:
        tau_m:      Motor torques [τ_m1, τ_m2]
        q_des:      Desired link angles (for logging)
        qd_des:     Desired link velocities (for logging)
    """
    global _init, _q_hold, _q_target, _ws, _phase_prev

    # Current end-effector position and Jacobian
    ee = forward_kinematics(q[0], q[1])
    J = jacobian(q[0], q[1])

    # Current wall surface position
    ws = wall_x + float(wall_pos) - wall_thickness / 2.0

    # --- Initialization (first call) ---
    if not _init:
        _q_hold = q.copy()
        _ws = ws
        _q_target = inverse_kinematics(_ws - 0.005, z_top, elbow_up=False)
        _init = True
        print(f"\n=== CONTROLLER INIT ===")
        print(f"  q_hold = {_q_hold}")
        print(f"  ee_init = {ee}")
        print(f"  wall surface = {_ws:.4f}")
        print(f"  q_target (approach) = {_q_target}")

    # --- Phase determination ---
    t1 = T_HOLD
    t2 = T_HOLD + T_APPROACH
    t3 = T_HOLD + T_APPROACH + T_CONTACT
    t4 = t3 + T_DRAW

    if t < t1:
        phase = 1
    elif t < t2:
        phase = 2
    elif t < t3:
        phase = 3
    elif t < t4:
        phase = 4
    else:
        phase = 5

    if phase != _phase_prev:
        print(f"\n--- PHASE {phase} at t={t:.2f}s | q={np.round(q,3)} | ee={np.round(ee,3)} ---")
        _phase_prev = phase

    # ================================================================
    # OUTER LOOP: Compute desired spring torque (τ_s_des)
    # ================================================================
    # This is the torque that the springs should exert on the links.
    # Formula: τ_s_des = τ_g(q_des) + Kp*(q_des - q) + Kd*(qd_des - qd) + J^T*F
    #
    # Note: We negate because the spring torque on the link is -K*δ = K(θ-q).
    # So τ_s_des here represents the desired K(θ-q) value. We compute
    # the "link command" first, then set τ_s = -link_command.

    if phase == 1:
        # ---- Phase 1: Hold initial pose ----
        q_des = _q_hold.copy()
        qd_des = np.zeros(2)
        F_contact = np.array([0.0, 0.0])

        tau_g = gravity_compensation(q_des[0], q_des[1])
        tau_pd = Kp_free * (q_des - q) + Kd_free * (qd_des - qd)
        tau_link_cmd = tau_g + tau_pd
        tau_s_des = -tau_link_cmd   # spring must produce opposite torque

    elif phase == 2:
        # ---- Phase 2: Move to approach configuration ----
        s = smoothstep(t - t1, T_APPROACH)
        q_des = _q_hold + s * (_q_target - _q_hold)
        qd_des = np.zeros(2)
        F_contact = np.array([0.0, 0.0])

        tau_g = gravity_compensation(q_des[0], q_des[1])
        tau_pd = Kp_free * (q_des - q) + Kd_free * (qd_des - qd)
        tau_link_cmd = tau_g + tau_pd
        tau_s_des = -tau_link_cmd

    elif phase == 3:
        # ---- Phase 3: Make gentle contact with wall ----
        s = smoothstep(t - t2, T_CONTACT)
        q_wall = inverse_kinematics(_ws, z_top, elbow_up=False)
        if q_wall is None:
            q_wall = _q_target.copy()
        q_des = _q_target + s * (q_wall - _q_target)
        qd_des = np.zeros(2)

        # Gradually ramp up contact force
        F_contact = np.array([F_CONTACT_X * s, 0.0])

        tau_g = gravity_compensation(q_des[0], q_des[1])
        tau_pd = Kp_free * (q_des - q) + Kd_free * (qd_des - qd)
        tau_link_cmd = tau_g + tau_pd + J.T @ F_contact
        tau_s_des = -tau_link_cmd

    elif phase == 4:
        # ---- Phase 4: Draw vertical line (descend along wall) ----
        s_draw = smoothstep(t - t3, T_DRAW)
        q_draw_top = inverse_kinematics(_ws - 0.005, z_top, elbow_up=False)
        q_draw_bot = inverse_kinematics(_ws - 0.005, z_bot, elbow_up=False)

        if q_draw_top is None:
            q_draw_top = _q_target.copy()
        if q_draw_bot is None:
            q_draw_bot = _q_target.copy()

        q_des = q_draw_top + s_draw * (q_draw_bot - q_draw_top)
        qd_des = np.zeros(2)

        # Maintain constant contact force into the wall
        F_contact = np.array([F_CONTACT_X, 0.0])

        tau_g = gravity_compensation(q_des[0], q_des[1])
        tau_pd = Kp_draw * (q_des - q) + Kd_draw * (qd_des - qd)
        tau_link_cmd = tau_g + tau_pd + J.T @ F_contact
        tau_s_des = -tau_link_cmd

    else:
        # ---- Phase 5: Hold final position ----
        q_des = q.copy()
        qd_des = np.zeros(2)
        F_contact = np.array([F_CONTACT_X, 0.0])

        tau_g = gravity_compensation(q_des[0], q_des[1])
        tau_pd = Kp_draw * (q_des - q) + Kd_draw * (qd_des - qd)
        tau_link_cmd = tau_g + tau_pd + J.T @ F_contact
        tau_s_des = -tau_link_cmd

    # Clip desired spring torque to prevent extreme deflections
    tau_s_des = np.clip(tau_s_des, -TAU_S_MAX, TAU_S_MAX)

    # ================================================================
    # CONVERSION: τ_s_des → θ_des
    # ================================================================
    # The spring torque on the link is: τ_spring = K(θ - q)
    # We want τ_spring = τ_s_des (but note τ_s_des was set to -tau_link_cmd)
    # Actually, the spring exerts K(θ-q) on the motor side and -K(θ-q) on the link.
    # The link needs: -K(θ-q) = tau_link_cmd  =>  K(θ-q) = -tau_link_cmd = τ_s_des
    # So: θ_des = q_des + τ_s_des / K  ... but careful with signs.
    #
    # More precisely: We want the spring deflection such that
    #   K * (θ_des - q_des) = τ_s_des
    #   θ_des = q_des + τ_s_des / K
    #
    # But τ_s_des = -tau_link_cmd = -(τ_g + τ_pd + J^T*F)
    # The spring needs to push the link with tau_link_cmd, so the motor
    # must be offset to create that deflection.

    theta_des = q_des + tau_s_des / K_spring

    # ================================================================
    # INNER LOOP: Motor-side PD to track θ_des
    # ================================================================
    # τ_m = τ_s_des + Kθ*(θ_des - θ) - Dθ*θ̇
    #
    # - τ_s_des is feedforward: the motor must at minimum produce
    #   this torque to maintain the desired spring deflection.
    # - Kθ*(θ_des - θ) corrects motor position error.
    # - Dθ*θ̇ provides motor-side velocity damping.

    tau_m = tau_s_des + Ktheta * (theta_des - theta) - Dtheta * theta_dot

    # Final torque saturation
    tau_m = np.clip(tau_m, -TORQUE_LIMIT, TORQUE_LIMIT)

    # Debug printing (every integer second)
    if abs(t - round(t)) < 0.011:
        print(f"t={t:5.1f} [Ph{phase}] EE=({ee[0]:+.3f},{ee[1]:+.3f}) "
              f"spr=({delta[0]:+.4f},{delta[1]:+.4f}) "
              f"tau=({tau_m[0]:+7.1f},{tau_m[1]:+7.1f})")

    return tau_m, q_des, qd_des


# ============================================
# SIMULATION LOOP
# ============================================

print("\n" + "="*60)
print("RUNNING SIMULATION - CASCADED CONTROLLER")
print("="*60)

time_log = []
q_log = []
q_des_log = []
qd_log = []
qd_des_log = []
qdd_log = []
qdd_des_log = []
delta_log = []
delta_des_log = []
theta_log = []
theta_des_log = []
thetadot_log = []
thetadot_des_log = []
ee_log = []
ee_des_log = []
tau_log = []
tau_link_log = []
contact_log = []

sim.set_target_realtime_rate(1.0)
dt_vis = 0.002     # 2ms timestep (matching your friend's code for stability)
t = 0.0

while t < T_final:

    #  ----Acquire Feedback, Please don't modify this section----
    theta = np.array([jl1.get_angle(plant_context), jl2.get_angle(plant_context)])
    thetadot = np.array([jl1.get_angular_rate(plant_context), jl2.get_angular_rate(plant_context)])
    
    spring_delta = np.array([jdel1.get_angle(plant_context), jdel2.get_angle(plant_context)])
    spring_deltad = np.array([jdel1.get_angular_rate(plant_context), jdel2.get_angular_rate(plant_context)])
    wall_pos = j_wall.get_translation(plant_context)
    wall_vel = j_wall.get_translation_rate(plant_context)

    wall_drive = -wall_k * wall_pos - wall_damping_extra * wall_vel
    normal_force = max(wall_fric_normal, wall_k * wall_pos)
    fric_static = wall_mu_static * normal_force
    fric_coulomb = wall_mu_coulomb * normal_force
    if abs(wall_vel) < wall_v_stiction and abs(wall_drive) < fric_static:
        wall_fric = -wall_drive
    else:
        wall_fric = -fric_coulomb * np.sign(wall_vel)

    wall_force = wall_drive + wall_fric

    # Compute link-side quantities
    q = theta + spring_delta           # link angle = motor angle + deflection
    qd = thetadot + spring_deltad      # link velocity

    ee_current = forward_kinematics(q[0], q[1])

    # ---- Call the cascaded controller ----
    tau, q_des, qd_des = controller(
        q, qd, theta, thetadot,
        spring_delta, spring_deltad,
        t, wall_pos
    )

    # ---- Apply torques (don't modify wall_force) ----
    u = np.array([tau[0], tau[1], wall_force])
    plant.get_actuation_input_port().FixValue(plant_context, u)
    sim.AdvanceTo(t)
    t += dt_vis

    # ---- Logging (don't modify) ----
    q_wall_pos = j_wall.get_translation(plant_context)
    wall_surface_x_nominal = (wall_x + q_wall_pos) - wall_thickness / 2.0
    dist_to_wall = wall_surface_x_nominal - ee_current[0]

    if dist_to_wall <= 0.016:
        contact_log.append(ee_current.copy())

    # Desired task-space for logging
    traj_x_des = forward_kinematics(q_des[0], q_des[1])
    p_des = traj_x_des

    # Convert task-space desired velocities to joint-space for logging
    try:
        J_inv = np.linalg.pinv(jacobian(q[0], q[1]))
        qd_des_joint = qd_des  # already in joint space
    except:
        qd_des_joint = np.zeros(2)

    qdd_des = np.zeros(2)
    delta_des = np.zeros(2)
    theta_des_log_val = q_des.copy()
    thetadot_des = qd_des_joint.copy()

    tau_link = np.array([k1 * spring_delta[0], k2 * spring_delta[1]])

    time_log.append(t)
    q_log.append(q.copy())
    q_des_log.append(q_des.copy())
    qdd_actual = (qd - qd_log[-1]) / dt_vis if qd_log else np.zeros(2)
    qd_log.append(qd.copy())
    qd_des_log.append(qd_des.copy())
    qdd_log.append(qdd_actual)
    qdd_des_log.append(qdd_des.copy())
    delta_log.append(spring_delta.copy())
    delta_des_log.append(delta_des.copy())
    theta_log.append(theta.copy())
    theta_des_log.append(theta_des_log_val.copy())
    thetadot_log.append(thetadot.copy())
    thetadot_des_log.append(thetadot_des.copy())
    ee_log.append(ee_current.copy())
    ee_des_log.append(p_des.copy())
    tau_log.append(tau.copy())
    tau_link_log.append(tau_link.copy())

q_final = q_log[-1]
ee_final = forward_kinematics(q_final[0], q_final[1])
if ee_final is None:
    ee_final = np.array([0.0, 0.0])

if PRINT_SUMMARY:
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print("Final joint angles  : phi1={:.4f} rad, phi2={:.4f} rad".format(q_final[0], q_final[1]))
    print("Final EE position   : x={:.4f} m, z={:.4f} m".format(ee_final[0], ee_final[1]))
    des_final = np.array(ee_des_log[-1])
    print("Target EE position  : x={:.4f} m, z={:.4f} m".format(des_final[0], des_final[1]))
    print("Position error      : {:.4f} m".format(np.linalg.norm(ee_final - des_final)))

plot_results(time_log, q_log, q_des_log, qd_log, qd_des_log, qdd_log, qdd_des_log,
             delta_log, delta_des_log, theta_log, theta_des_log, thetadot_log,
             thetadot_des_log, ee_log, ee_des_log, tau_log, tau_link_log, contact_log, PRINT_SUMMARY)

print("\n" + "="*60)
print("MeshCat visualization is running...")
print("="*60)