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

q_err_integral = np.zeros(2)

q_des_prev = np.array([phi1_init, phi2_init])

def forward_kinematics(J1, J2):
    
    x = L1 * math.cos(J1) + L2 * math.cos(J1 + J2)
    z = L1 * math.sin(J1) + L2 * math.sin(J1 + J2)
    #print("Current EE position: x={:.4f} m, z={:.4f} m".format(x, z))
    return np.array([x, z])

def inverse_kinematics(x_target, z_target, elbow_up=False):
    dist_sq = x_target**2 + z_target**2
    dist = math.sqrt(dist_sq)

    # Check if the point is reachable (L1 + L2 = 2.0)
    if dist > (L1 + L2) or dist < abs(L1 - L2):
        print("Target out of reach!")
        return None

    # Calculate q2
    # Law of Cosines: cos(q2) = (x^2 + z^2 - L1^2 - L2^2) / (2 * L1 * L2)
    cos_q2 = (dist_sq - L1**2 - L2**2) / (2 * L1 * L2)
    
    # Account for floating point errors (stay within [-1, 1])
    cos_q2 = max(-1.0, min(1.0, cos_q2))
    
    if elbow_up:
        q2 = -math.acos(cos_q2)
    else:
        q2 = math.acos(cos_q2)

    # Calculate q1
    # q1 = angle_to_point - angle_within_triangle
    alpha = math.atan2(z_target, x_target)
    beta = math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    q1 = alpha - beta

    return np.array([q1, q2])
 
def trajectory_planner(t: float):
    # Wall is at x=1.1, so 1.12 gives a gentle, stable contact force
    x_wall_contact = 1.12  
    z_start = -1.0
    
    # Parameters for the Drawing Motion
    freq = 0.1 # 10 seconds per full up-and-down cycle
    omega = 2.0 * math.pi * freq
    amp_z = 0.5 
    z0 = -0.5 # Z will oscillate between -1.0 and 0.0
    
    # Initial EE position (from forward kinematics at t=0)
    q_init = np.array([phi1_init, phi2_init])
    ee_init = forward_kinematics(q_init[0], q_init[1])
    
    t_transition = 2.0   # seconds to move from start to the wall
    
    if t < t_transition:
        # Smooth transition using cubic interpolation (3s^2 - 2s^3)
        s = t / t_transition
        s_smooth = 3 * (s**2) - 2 * (s**3)
        ds_dt = (6 * s - 6 * s**2) / t_transition
        
        x_des = ee_init[0] + s_smooth * (x_wall_contact - ee_init[0])
        z_des = ee_init[1] + s_smooth * (z_start - ee_init[1])
        
        xd_des = ds_dt * (x_wall_contact - ee_init[0])
        zd_des = ds_dt * (z_start - ee_init[1])
        xdd_des = 0.0
        zdd_des = 0.0
    else:
        # Draw a vertical line on the wall
        t_line = t - t_transition
        x_des = x_wall_contact
        
        # Sine wave oscillating around z0
        z_des = z0 + amp_z * math.sin(omega * t_line - math.pi/2)
        
        xd_des = 0.0
        zd_des = amp_z * omega * math.cos(omega * t_line - math.pi/2)
        
        xdd_des = 0.0
        zdd_des = -amp_z * (omega**2) * math.sin(omega * t_line - math.pi/2)

    return x_des, z_des, xd_des, zd_des, xdd_des, zdd_des

def jacobian(phi1: float, phi2: float, L1: float, L2: float):
    """
    Jacobian matrix for 2-link planar arm.
    
    Derivation:
    J = [∂x/∂φ1, ∂x/∂φ2]
        [∂z/∂φ1, ∂z/∂φ2]
    
    ∂x/∂φ1 = -L1*sin(phi1) - L2*sin(phi1 + phi2)
    ∂x/∂φ2 = -L2*sin(phi1 + phi2)
    ∂z/∂φ1 = L1*cos(phi1) + L2*cos(phi1 + phi2)
    ∂z/∂φ2 = L2*cos(phi1 + phi2)
    """
    J = np.array([
        [-L1 * math.sin(phi1) - L2 * math.sin(phi1 + phi2), -L2 * math.sin(phi1 + phi2)],
        [ L1 * math.cos(phi1) + L2 * math.cos(phi1 + phi2),  L2 * math.cos(phi1 + phi2)],
    ])
    return J

def controller(q, qd, theta, theta_dot, x_des, z_des, xd_des, zd_des, dt):
    global q_des_prev

    # 1. Inverse Kinematics
    q_target = inverse_kinematics(x_des, z_des, elbow_up=False)
    if q_target is None:
        q_target = q_des_prev.copy()

    # Bug 2 fix: clip against previous *desired* q, not actual q
    max_q_jump = 0.1
    q_des = np.clip(q_target, q_des_prev - max_q_jump, q_des_prev + max_q_jump)
    q_des_prev = q_des.copy()  # ← update for next call

    # 2. Desired joint velocities (for logging only — NOT used in damping term)
    J_des = jacobian(q_des[0], q_des[1], L1, L2)
    try:
        qd_des =  np.array([xd_des, zd_des])
    except:
        qd_des = np.zeros(2)

    # 3. Gravity compensation at desired link position
    m1, m2 = 1.5, 1.5
    m_rotor2 = 0.4
    g = 9.81
    lc1, lc2 = L1 / 2, L2 / 2

    tauG1 = (g * math.cos(q_des[0]) * (m1 * lc1 + m_rotor2 * L1 + m2 * L1)
             + m2 * g * lc2 * math.cos(q_des[0] + q_des[1]))
    tauG2 = m2 * g * lc2 * math.cos(q_des[0] + q_des[1])
    tauG = np.array([tauG1, tauG2])

    k_mat = np.diag([k1, k2])
    theta_des = q_des + np.linalg.solve(k_mat, tauG)



    # 6. PD with spring-mode damping
    # K_P must satisfy K_P < 4*J/dt^2:
    #   joint1: 4*0.00128/0.02^2 = 12.8  → use 6
    #   joint2: 4*0.00102/0.02^2 = 10.2  → use 2  (soft spring = lower limit)
    k_p = np.diag([45.8, 30])
    k_d = np.diag([0.3, 0.3])

    
    k_d = k_d @ theta_dot 
    k_p = k_p @ (theta_des - theta)
    tau = tauG + k_p  - k_d 
    tau = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)

    print( "kp = {}, kd = {}".format(k_p, k_d), "tauG =",tauG, "tau =", tau)
    #print("Controller Debug: q_des = {}, theta_des = {}, tau = {}".format(q_des, theta_des, tau))

    return tau, q_des, qd_des


# ============================================
# SIMULATION LOOP WITH COMPLETE CONTROL
# ============================================

print("\n" + "="*60)
print("RUNNING SIMULATION")
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
dt_vis = 0.02
t = 0.0
T_final = 20.0

thetadot_prev = np.zeros(2)

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
    

    # ---------------------------------------
    # 1. Acquire Feedback
    thetaddot = (thetadot - thetadot_prev) / dt_vis
    thetadot_prev = thetadot.copy()

    q = theta + spring_delta
    qd = thetadot + spring_deltad

    ee_current = forward_kinematics(q[0], q[1])

    traj_data = trajectory_planner(t)
    x_des, z_des, xd_des, zd_des, xdd_des, zdd_des = traj_data

    # Contact-aware x clamping: never command deeper than the current wall
    # surface. Without this the motor spins while the link is blocked and
    # the spring winds up to many radians, saturating the torque output.
    wall_surface_x_now = (wall_x + wall_pos) - wall_thickness / 2.0
    if x_des > wall_surface_x_now - 0.005:
        x_des = wall_surface_x_now - 0.005   # 5 mm clearance
        xd_des = 0.0                           # no desired velocity into wall
    traj_data = (x_des, z_des, xd_des, zd_des, xdd_des, zdd_des)

        # Check for contact forces on the end effector (link2) and the wall
    contact_results = plant.get_contact_results_output_port().Eval(plant_context)
    tip_force = np.array([0.0, 0.0, 0.0])
    for i in range(contact_results.num_point_pair_contacts()):
        info = contact_results.point_pair_contact_info(i)
        body_A = plant.get_body(info.bodyA_index()).name()
        body_B = plant.get_body(info.bodyB_index()).name()
        
        force_on_A = info.contact_force()
        # Only count if the contact is specifically between link2 and wall
        if (body_A == "link2" and body_B == "wall"):
            tip_force += force_on_A
        elif (body_B == "link2" and body_A == "wall"):
            tip_force -= force_on_A


    
    tau, q_des, qd_des = controller(q, qd, theta, thetadot ,x_des, z_des, xd_des, zd_des, dt_vis)    

    # -----Please don't modify the wall_force in here ------
    u = np.array([tau[0], tau[1] , wall_force])
    plant.get_actuation_input_port().FixValue(plant_context, u)
    sim.AdvanceTo(t)
    t += dt_vis



    # -----Logging data, please don't modify this section ------
    q_wall_pos = j_wall.get_translation(plant_context)
    wall_surface_x_nominal = (wall_x + q_wall_pos) - wall_thickness / 2.0
    dist_to_wall = wall_surface_x_nominal - ee_current[0]
    
    if dist_to_wall <= 0.016: 
         contact_log.append(ee_current.copy())
    
    # Extract desired task-space position from the trajectory data
    x_des, z_des, xd_des, zd_des, xdd_des, zdd_des = traj_data
    p_des = np.array([x_des, z_des])

    #convert task spaince into joint space 
    try:
        J_inv = np.linalg.pinv(jacobian(q[0], q[1], L1, L2))
        qd_des_joint = J_inv @ np.array([xd_des, zd_des])
    except:
        qd_des_joint = np.zeros(2)
    
    # Set dummy variables for the values we aren't actively computing
    qdd_des = np.zeros(2)          # Desired joint acceleration
    delta_des = np.zeros(2)        # Desired spring deflection (ideally 0)
    theta_des = q_des.copy()       # Desired motor angle
    thetadot_des = qd_des_joint.copy() 
    
    # Calculate the actual torque experienced by the links due to the springs
    tau_link = np.array([k1 * spring_delta[0], k2 * spring_delta[1]])
    
    # Reassign qd_des so the logger picks up the correct joint velocities
    qd_des = qd_des_joint

    
    # ----Cancel the comment when you need to plot your data. -----

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
    theta_des_log.append(theta_des.copy())
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

#----Cancel the comment when you need to plot your data. -----
plot_results(time_log, q_log, q_des_log, qd_log, qd_des_log, qdd_log, qdd_des_log, 
             delta_log, delta_des_log, theta_log, theta_des_log, thetadot_log, 
             thetadot_des_log, ee_log, ee_des_log, tau_log, tau_link_log, contact_log, PRINT_SUMMARY)

print("\n" + "="*60)
print("MeshCat visualization is running...")
print("="*60)
input("\nPress Enter to quit...")