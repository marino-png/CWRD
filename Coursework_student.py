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
    
    x0 = 1.12  #
    z0 = -0.5   
    
    # 2. Parameters for the Motion
    freq = 0.1 # 10 seconds per full cycle
    omega = 2.0 * math.pi * freq
    amp_x = 0.3 # Oscillates between x=1.02 and x=1.22
    amp_z = 0.3 # Oscillates between z=-0.5 and z=0.5
    
    # 3. Position: Circular/Elliptical path
    # Using 1.57 (pi/2) phase shift for z makes it a circle
    x_des = x0 + amp_x * math.sin(omega * t)
    z_des = z0 + amp_z * math.sin(omega * t - 1.57)
    
    # 4. Velocity (First Derivative)
    xd_des = amp_x * omega * math.cos(omega * t)
    zd_des = amp_z * omega * math.cos(omega * t + 1.57)

    # 5. Acceleration (Second Derivative)
    xdd_des = -amp_x * (omega**2) * math.sin(omega * t)
    zdd_des = -amp_z * (omega**2) * math.sin(omega * t - 1.57)

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

def controller(q: np.ndarray,
               qd: np.ndarray,
               x_des: float,
               z_des: float,
               xd_des: float,
               zd_des: float,
               xdd_des: float,
               zdd_des: float,
               dt: float,
               theta_dot: float,
               theta_ddot: float,
               F_des: np.ndarray = None,
               tip_force: np.ndarray = None):
    
    # 1. Kinematics & Errors
    J_cur = jacobian(q[0], q[1], L1, L2)
    c_current = forward_kinematics(q[0], q[1])
    c_des = np.array([x_des, z_des])
    Ce = c_des - c_current
    
    # joint space and velocity error
    J_cur = jacobian(q[0], q[1], L1, L2)
    qd_des = np.linalg.pinv(J_cur) @ np.array([xd_des, zd_des])
    global q_des_prev
    q_des = q_des_prev + qd_des * dt
    q_des_prev = q_des.copy()
    q_err = q_des - q
    qd_err = qd_des - qd# 2. Gains (Adjust these to prevent the wall from moving too much)


    # Task-space velocity error
    c_dot = J_cur @ qd
    c_dot_des = np.array([xd_des, zd_des])
    c_dot_err = c_dot_des - c_dot

    #task pace acceleration error
    c_ddot_des = np.array([xdd_des,zdd_des])
    
    if not hasattr(controller, "c_dot_prev"):
            controller.c_dot_prev = c_dot.copy()
    c_ddot = (c_dot - controller.c_dot_prev) / dt
    controller.c_dot_prev = c_dot.copy()

    # fliter
    if not hasattr(controller, "c_ddot_filt_prev"):
            controller.c_ddot_filt_prev = c_ddot.copy()
    alpha_ddot = 0.5
    c_ddot = alpha_ddot * controller.c_ddot_filt_prev + (1.0 - alpha_ddot) * c_ddot
    controller.c_ddot_filt_prev = c_ddot.copy()

    c_ddot_err = c_ddot_des - c_ddot

    # 3. Gravity Cancellation for the links
    m1, m2 = 1.5, 1.5 # From simulator_setup.py
    g = 9.81
    lc1, lc2 = L1/2, L2/2
    r = 0.15
    r_rotor = 0.08

    # Motor inertias (solid sphere, used in J term)
    Jm = np.array([2/5 * 0.5 * r_rotor**2,   # rotor1: 0.00128 kg·m²
                2/5 * 0.4 * r_rotor**2])  # rotor2: 0.001024 kg·m²

    # Motor damping (from RevoluteJoint damping parameter)
    Dm = np.array([0.1, 0.1])    # N·m·s/rad

    # Inertia matrix M(q)
    I1 = 2/5*m1*r**2   
    I2 = 2/5*m2*r**2  
    M11 = I1 + I2 + m1*lc1**2 + m2*(L1**2 + lc2**2 + 2*L1*lc2*math.cos(q[1]))
    M12 = I2 + m2*(lc2**2 + L1*lc2*math.cos(q[1]))
    M21 = I2 + m2*(lc2**2 + L1*lc2*math.cos(q[1]))
    M22 = I2 + m2*lc2**2
    M = np.array([[M11, M12],
                  [M21, M22]])
    
    # Coriolis matrix C(q, qdot)
    C11 = -m2 * L1 * lc2 * math.sin(q[1]) * qd[1]
    C12 = -m2 * L1 * lc2 * math.sin(q[1]) * (qd[0] + qd[1])
    C21 =  m2 * L1 * lc2 * math.sin(q[1]) * qd[0]
    C22 = 0.0
    C = np.array([[C11, C12],
                [C21, C22]])
    
    # Gravity vector G(q_des)

    tauG1comp = -g * (m1 * lc1 * math.cos(q_des[0])
                    + m2 * L1  * math.cos(q_des[0])
                    + m2 * lc2 * math.cos(q_des[0] + q_des[1])
                    )

    tauG2comp = -g * (m2 * lc2 * math.cos(q_des[0] + q_des[1])
                        ) 

    # Gravity vector G(q)
    tauG1canc = -g * (m1 * lc1 * math.cos(q[0]) + m2 * L1 * math.cos(q[0]) + m2 * lc2 * math.cos(q[0] + q[1]))
    tauG2canc = -g * (m2 * lc2 * math.cos(q[0] + q[1]))


    tauGcomp = np.array([tauG1comp, tauG2comp])
    tauGcanc = np.array([tauG1canc, tauG2canc])

        
    # Operational Space Control 
    J_dot = np.array([
        [-L1 * math.cos(q[0]) * qd[0] - L2 * math.cos(q[0] + q[1]) * (qd[0] + qd[1]), -L2 * math.cos(q[0] + q[1]) * (qd[0] + qd[1])],
        [-L1 * math.sin(q[0]) * qd[0] - L2 * math.sin(q[0] + q[1]) * (qd[0] + qd[1]), -L2 * math.sin(q[0] + q[1]) * (qd[0] + qd[1])]
    ])

    J_inv = np.linalg.pinv(J_cur)
    J_inv_T = J_inv.T

    Lambda = J_inv_T @ M @ J_inv
    Gamma = J_inv_T @ C @ J_inv - Lambda @ J_dot @ J_inv
    eta = J_inv_T @ tauGcanc
    
    if np.linalg.norm(tip_force) > 1.0:   # in contact
        Mc = np.diag([1.0, 1.0])    
        Kc = np.diag([20.0, 20.0])  
        Dc = np.diag([30.0, 30.0])
    else:                                   # free motion
        Mc = np.diag([1.0, 1.0])    
        Kc = np.diag([2.0, 100.0])    
        Dc = np.diag([60.0, 40.0])  # High damping is the key — kills the bounce
            
    #c_ddot = c_ddot_des + Dc @ c_dot_des + Kc @ Ce
    J_dot_qd = J_dot @ qd
    # Impedance controller

    #fc = Lambda @ c_ddot+ Gamma @ c_dot + eta + Mc @ c_ddot_err + Dc @ c_dot_err + Kc @ Ce
    fc = Lambda @ c_ddot+ Gamma @ c_dot + eta + Jm @ theta_dot + Dm @ theta_ddot
    tau = J_cur.T @ fc


    # Saturation and numerical safety
    tau = np.clip(tau, -float(TORQUE_LIMIT), float(TORQUE_LIMIT))
    tau = np.nan_to_num(tau, nan=0.0, posinf=float(TORQUE_LIMIT), neginf=-float(TORQUE_LIMIT))
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
    thetadotdot = (thetadot - thetadot_prev) / dt_vis
    thetadot_prev = thetadot.copy()

    q = theta + spring_delta
    qd = thetadot + spring_deltad

    ee_current = forward_kinematics(q[0], q[1])

    traj_data = trajectory_planner(t)
    x_des, z_des, xd_des, zd_des, xdd_des, zdd_des = traj_data

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


    des_force = np.array([1.0, 0.0, 0.0])
    tau, q_des, qd_des = controller(q, qd, x_des, z_des, xd_des, zd_des, xdd_des, zdd_des, dt_vis,thetadot, thetadotdot,  F_des=des_force, tip_force=tip_force)
    
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
