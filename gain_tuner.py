"""
gain_tuner.py  –  Automated PD Gain Search for ASR Coursework
==============================================================
Runs N simulations with different (kp1, kp2, kd1, kd2) combinations,
records per-trial metrics, and saves everything to gain_tuning_results.csv.

Usage
-----
    python gain_tuner.py --strategy random --n 50 --T 10   # quick first pass
    python gain_tuner.py --strategy grid   --T 10           # 320-pt coarse grid (~2 min)
    python gain_tuner.py --strategy dense  --T 10           # 1200-pt dense grid (~8 min)
    python gain_tuner.py --strategy fine   --T 20 --center 57.25,3.74,1.45,0.90
    python gain_tuner.py --strategy both   --n 30 --T 10   # grid + random

The script prints a live summary line per trial and writes every result to CSV
immediately, so you can kill it early and still use partial data.

After the run a short "best gains" block is printed to the console.
"""

import argparse
import csv
import itertools
import math
import os
import random
import sys
import time

import numpy as np

# ── Drake imports ─────────────────────────────────────────────────────────────
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    Cylinder,
    CoulombFriction,
    DiagramBuilder,
    FixedOffsetFrame,
    PrismaticJoint,
    RevoluteJoint,
    RevoluteSpring,
    RigidTransform,
    RotationMatrix,
    Simulator,
    Sphere,
    SpatialInertia,
    UnitInertia,
)


# ══════════════════════════════════════════════════════════════════════════════
# ROBOT / ENVIRONMENT CONSTANTS  –  must match simulator_setup.py exactly
# ══════════════════════════════════════════════════════════════════════════════
L1, L2 = 1.0, 1.0
PHI1_INIT = math.pi * -1 / 2
PHI2_INIT = math.pi *  1 / 9
K1, K2   = 60.5, 20.01          # spring stiffnesses [N·m/rad]
WALL_X         = 1.1            # wall nominal x-position [m]
_WALL_THICKNESS = 0.03
TORQUE_LIMIT   = 200.0

# Wall physics — DO NOT CHANGE (coursework spec)
_WALL_MASS         = 150.0
_WALL_DAMPING      = 180.0
_WALL_K            = 0.0         # wall_k in setup
_WALL_DAMP_EXTRA   = 350.0
_WALL_MU_STATIC    = 0.6
_WALL_MU_COULOMB   = 0.45
_WALL_FRIC_NORMAL  = 800.0
_WALL_V_STICTION   = 0.005
_WALL_WIDTH_Y      = 0.70
_WALL_HEIGHT_Z     = 4.00

G   = 9.81
M1, M2      = 1.5, 1.5
M_ROTOR2    = 0.4
LC1, LC2    = L1 / 2, L2 / 2


# ══════════════════════════════════════════════════════════════════════════════
# HEADLESS SIMULATOR BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def _build_headless_sim():
    """
    Build a fresh Drake sim without Meshcat.
    set_target_realtime_rate(0) → runs as fast as CPU allows.
    """
    axis = [0, 1, 0]
    dt   = 0.001

    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=dt)
    plant.mutable_gravity_field().set_gravity_vector([0.0, 0.0, -G])

    def _body(name, mass, r):
        return plant.AddRigidBody(
            name,
            SpatialInertia(mass=mass, p_PScm_E=[0, 0, 0],
                           G_SP_E=UnitInertia.SolidSphere(r)),
        )

    link1  = _body("link1",  M1,  0.16)
    link2  = _body("link2",  M2,  0.16)
    rotor1 = _body("rotor1", 0.5, 0.08)
    rotor2 = _body("rotor2", M_ROTOR2, 0.08)

    wm   = plant.AddFrame(FixedOffsetFrame("world_mount", plant.world_frame(), RigidTransform([0, 0, 0])))
    lb1  = plant.AddFrame(FixedOffsetFrame("link1_base",  link1.body_frame(),  RigidTransform([-L1/2, 0, 0])))
    lt1  = plant.AddFrame(FixedOffsetFrame("link1_tip",   link1.body_frame(),  RigidTransform([+L1/2, 0, 0])))
    lb2  = plant.AddFrame(FixedOffsetFrame("link2_base",  link2.body_frame(),  RigidTransform([-L2/2, 0, 0])))

    jm1 = plant.AddJoint(RevoluteJoint("motor_joint_1", wm,  rotor1.body_frame(), axis, damping=0.1))
    plant.AddJointActuator("act_m1", jm1)
    js1 = plant.AddJoint(RevoluteJoint("spring_joint_1", rotor1.body_frame(), lb1, axis, damping=0.0))
    plant.AddForceElement(RevoluteSpring(js1, nominal_angle=0.0, stiffness=K1))

    jm2 = plant.AddJoint(RevoluteJoint("motor_joint_2", lt1, rotor2.body_frame(), axis, damping=0.1))
    plant.AddJointActuator("act_m2", jm2)
    js2 = plant.AddJoint(RevoluteJoint("spring_joint_2", rotor2.body_frame(), lb2, axis, damping=0.0))
    plant.AddForceElement(RevoluteSpring(js2, nominal_angle=0.0, stiffness=K2))

    X_cyl = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2))
    plant.RegisterCollisionGeometry(link1, X_cyl, Cylinder(0.03, L1),   "l1_col", CoulombFriction(0.8, 0.6))
    plant.RegisterCollisionGeometry(link2, X_cyl, Cylinder(0.015, L2),  "l2_col", CoulombFriction(0.8, 0.6))

    # Wall
    wall_body = plant.AddRigidBody(
        "wall_body",
        SpatialInertia(mass=_WALL_MASS, p_PScm_E=[0, 0, 0],
                       G_SP_E=UnitInertia.SolidBox(_WALL_THICKNESS, _WALL_WIDTH_Y, _WALL_HEIGHT_Z)),
    )
    wo = plant.AddFrame(FixedOffsetFrame(
        "wall_origin", plant.world_frame(), RigidTransform([WALL_X, 0.0, 0.0])
    ))
    jw = plant.AddJoint(PrismaticJoint(
        "wall_joint", wo, wall_body.body_frame(), [1, 0, 0], damping=_WALL_DAMPING
    ))
    plant.AddJointActuator("act_wall", jw)
    jw.set_position_limits([0.0], [np.inf])
    ws = Box(_WALL_THICKNESS, _WALL_WIDTH_Y, _WALL_HEIGHT_Z)
    plant.RegisterCollisionGeometry(wall_body, RigidTransform(), ws, "wall_col", CoulombFriction(0.9, 0.7))

    plant.Finalize()

    diagram = builder.Build()
    sim     = Simulator(diagram)
    sim.set_target_realtime_rate(0.0)          # ← headless, no rate limit

    ctx  = sim.get_mutable_context()
    pctx = plant.GetMyMutableContextFromRoot(ctx)

    jm1.set_angle(pctx, PHI1_INIT)
    jm2.set_angle(pctx, PHI2_INIT)
    js1.set_angle(pctx, 0.0)
    js2.set_angle(pctx, 0.0)
    plant.get_actuation_input_port().FixValue(pctx, [0.0, 0.0, 0.0])
    sim.Initialize()

    return dict(sim=sim, plant=plant, pctx=pctx,
                jm1=jm1, jm2=jm2, js1=js1, js2=js2, jw=jw)


# ══════════════════════════════════════════════════════════════════════════════
# KINEMATICS  (identical to Coursework_student.py)
# ══════════════════════════════════════════════════════════════════════════════
def _fk(q1, q2):
    x = L1 * math.cos(q1) + L2 * math.cos(q1 + q2)
    z = L1 * math.sin(q1) + L2 * math.sin(q1 + q2)
    return np.array([x, z])

def _ik(x_t, z_t):
    d2   = x_t**2 + z_t**2
    dist = math.sqrt(d2)
    if dist > (L1 + L2) or dist < abs(L1 - L2):
        return None
    cq2 = max(-1.0, min(1.0, (d2 - L1**2 - L2**2) / (2 * L1 * L2)))
    q2   = math.acos(cq2)
    alpha = math.atan2(z_t, x_t)
    beta  = math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    return np.array([alpha - beta, q2])

def _jac(phi1, phi2):
    return np.array([
        [-L1*math.sin(phi1) - L2*math.sin(phi1+phi2), -L2*math.sin(phi1+phi2)],
        [ L1*math.cos(phi1) + L2*math.cos(phi1+phi2),  L2*math.cos(phi1+phi2)],
    ])

def _traj(t):
    x_wall_contact = 1.12
    z_start = -1.0
    omega = 2.0 * math.pi * 0.1
    amp_z = 0.5
    z0 = -0.5
    
    ee_init = _fk(PHI1_INIT, PHI2_INIT)
    tt = 2.0
    
    if t < tt:
        s = t / tt
        s_smooth = 3 * (s**2) - 2 * (s**3)
        ds_dt = (6 * s - 6 * s**2) / tt
        
        xd = ee_init[0] + s_smooth * (x_wall_contact - ee_init[0])
        zd = ee_init[1] + s_smooth * (z_start - ee_init[1])
        xdd = ds_dt * (x_wall_contact - ee_init[0])
        zdd = ds_dt * (z_start - ee_init[1])
        xadd = 0.0
        zadd = 0.0
    else:
        t_line = t - tt
        xd = x_wall_contact
        zd = z0 + amp_z * math.sin(omega * t_line - math.pi/2)
        xdd = 0.0
        zdd = amp_z * omega * math.cos(omega * t_line - math.pi/2)
        xadd = 0.0
        zadd = -amp_z * omega**2 * math.sin(omega * t_line - math.pi/2)
        
    return xd, zd, xdd, zdd, xadd, zadd


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE TRIAL
# ══════════════════════════════════════════════════════════════════════════════
def run_trial(kp1: float, kp2: float, kd1: float, kd2: float,
              T_final: float = 20.0, dt_vis: float = 0.02) -> dict:
    """
    Run one full simulation with the given diagonal PD gains.
    Returns a flat dict of scalar metrics.

    Note: builds a fresh simulator per trial — Drake contexts are not reusable.
    """
    env  = _build_headless_sim()
    sim  = env['sim'];  plant = env['plant'];  pctx = env['pctx']
    jm1  = env['jm1'];  jm2  = env['jm2']
    js1  = env['js1'];  js2  = env['js2']
    jw   = env['jw']

    k_p  = np.diag([kp1, kp2])
    k_d  = np.diag([kd1, kd2])
    k_sp = np.diag([K1,  K2])

    q_des_prev     = np.array([PHI1_INIT, PHI2_INIT])
    thetadot_prev  = np.zeros(2)
    t              = 0.0

    # ── per-step accumulators ──────────────────────────────────────────────
    ee_err_list      = []   # ||ee_actual - ee_desired||
    q_err_list       = []   # ||q - q_des||
    theta_err_list   = []   # ||theta - theta_des||
    delta_list       = []   # ||spring_delta||  (= ||q - theta||)
    tau_list         = []   # ||tau||
    wall_disp_list   = []   # |wall_pos|
    tau1_log         = []   # raw tau1 (for mean/std)
    tau2_log         = []   # raw tau2
    contact_steps    = 0
    tau_sat_steps    = 0
    total_steps      = 0
    diverged         = False
    diverge_time     = float('nan')

    while t < T_final:
        # ── Feedback ──────────────────────────────────────────────────────
        theta    = np.array([jm1.get_angle(pctx), jm2.get_angle(pctx)])
        thetadot = np.array([jm1.get_angular_rate(pctx), jm2.get_angular_rate(pctx)])
        sdelta   = np.array([js1.get_angle(pctx), js2.get_angle(pctx)])
        sdeltad  = np.array([js1.get_angular_rate(pctx), js2.get_angular_rate(pctx)])
        wall_pos = jw.get_translation(pctx)
        wall_vel = jw.get_translation_rate(pctx)

        # Wall force (DO NOT MODIFY)
        wall_drive   = -_WALL_K * wall_pos - _WALL_DAMP_EXTRA * wall_vel
        normal_force = max(_WALL_FRIC_NORMAL, _WALL_K * wall_pos)
        fric_static  = _WALL_MU_STATIC  * normal_force
        fric_coulomb = _WALL_MU_COULOMB * normal_force
        if abs(wall_vel) < _WALL_V_STICTION and abs(wall_drive) < fric_static:
            wall_fric = -wall_drive
        else:
            wall_fric = -fric_coulomb * np.sign(wall_vel)
        wall_force = wall_drive + wall_fric

        q  = theta + sdelta
        qd = thetadot + sdeltad
        ee = _fk(q[0], q[1])

        # ── Divergence guard ───────────────────────────────────────────────
        if not np.all(np.isfinite(q)) or np.any(np.abs(q) > 60.0):
            diverged     = True
            diverge_time = t
            break

        # ── Trajectory ────────────────────────────────────────────────────
        x_des, z_des, xd_des, zd_des, _, _ = _traj(t)
        p_des = np.array([x_des, z_des])

        # ── Controller  (same logic as Coursework_student.py) ─────────────
        q_tgt = _ik(x_des, z_des)
        if q_tgt is None:
            q_tgt = q_des_prev.copy()
        q_des = np.clip(q_tgt, q_des_prev - 0.1, q_des_prev + 0.1)
        q_des_prev = q_des.copy()

        tauG1 = (G * math.cos(q_des[0]) * (M1*LC1 + M_ROTOR2*L1 + M2*L1)
                 + M2*G*LC2*math.cos(q_des[0] + q_des[1]))
        tauG2 = M2 * G * LC2 * math.cos(q_des[0] + q_des[1])
        tauG  = np.array([tauG1, tauG2])

        theta_des = q_des + np.linalg.solve(k_sp, tauG)

        # BUG FIX: use local `thetadot`, not a global shadow
        tau = k_p @ (theta_des - theta) - k_d @ thetadot
        tau = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)

        u = np.array([tau[0], tau[1], wall_force])
        plant.get_actuation_input_port().FixValue(pctx, u)
        sim.AdvanceTo(t)
        t += dt_vis
        total_steps += 1

        # ── Metrics accumulation ──────────────────────────────────────────
        ee_err_list.append(float(np.linalg.norm(ee - p_des)))
        q_err_list.append(float(np.linalg.norm(q - q_des)))
        theta_err_list.append(float(np.linalg.norm(theta - theta_des)))
        delta_list.append(float(np.linalg.norm(sdelta)))
        tau_mag = float(np.linalg.norm(tau))
        tau_list.append(tau_mag)
        tau1_log.append(float(tau[0]))
        tau2_log.append(float(tau[1]))
        wall_disp_list.append(float(abs(wall_pos)))

        if np.any(np.abs(tau) >= TORQUE_LIMIT * 0.99):
            tau_sat_steps += 1

        # Contact: same threshold as student code
        surf_x    = (WALL_X + wall_pos) - _WALL_THICKNESS / 2.0
        dist_wall = surf_x - ee[0]
        if dist_wall <= 0.016:
            contact_steps += 1

    # ── Aggregate ──────────────────────────────────────────────────────────
    def s(arr, fn):
        a = np.asarray(arr, dtype=float)
        return float(fn(a)) if a.size > 0 else float('nan')

    ns = max(total_steps, 1)
    ee_arr = np.asarray(ee_err_list)

    return {
        # ── inputs ──────────────────────────────────────────────────────
        'kp1': kp1, 'kp2': kp2, 'kd1': kd1, 'kd2': kd2,
        # ── stability ───────────────────────────────────────────────────
        'diverged':         int(diverged),
        'diverge_time_s':   diverge_time,
        'steps_completed':  total_steps,
        # ── EE tracking error ───────────────────────────────────────────
        'ee_err_mean':      s(ee_err_list, np.mean),
        'ee_err_max':       s(ee_err_list, np.max),
        'ee_err_rms':       float(np.sqrt(np.mean(ee_arr**2))) if ee_arr.size else float('nan'),
        'ee_err_final':     ee_err_list[-1] if ee_err_list else float('nan'),
        # ── link angle error (q) ────────────────────────────────────────
        'q_err_mean':       s(q_err_list, np.mean),
        'q_err_max':        s(q_err_list, np.max),
        'q_err_std':        s(q_err_list, np.std),
        # ── motor angle error (theta) ───────────────────────────────────
        'theta_err_mean':   s(theta_err_list, np.mean),
        'theta_err_max':    s(theta_err_list, np.max),
        # ── spring deflection ───────────────────────────────────────────
        'delta_mean':       s(delta_list, np.mean),
        'delta_max':        s(delta_list, np.max),
        'delta_std':        s(delta_list, np.std),
        # ── torques ─────────────────────────────────────────────────────
        'tau_mean':         s(tau_list, np.mean),
        'tau_max':          s(tau_list, np.max),
        'tau1_mean':        s(tau1_log, np.mean),
        'tau2_mean':        s(tau2_log, np.mean),
        'tau_sat_fraction': tau_sat_steps / ns,
        # ── wall movement ───────────────────────────────────────────────
        'wall_disp_mean':   s(wall_disp_list, np.mean),
        'wall_disp_max':    s(wall_disp_list, np.max),
        'wall_disp_std':    s(wall_disp_list, np.std),
        # ── contact ─────────────────────────────────────────────────────
        'contact_fraction': contact_steps / ns,
        'contact_steps':    contact_steps,
    }


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORE  (lower = better)
# ══════════════════════════════════════════════════════════════════════════════
def composite_score(m: dict) -> float:
    """
    Weighted combination of metrics.  Lower = better gain set.

    Weights reflect the coursework objective:
      - Wall movement is the primary failure mode  → highest weight
      - EE tracking error is the secondary goal
      - Torque saturation means the controller is fighting the dynamics
      - Lack of contact means the robot isn't even touching the wall

    Adjust weights here if your priorities differ.
    """
    if m['diverged']:
        return 1e9 + (20.0 - m.get('diverge_time_s', 0.0))   # earlier divergence = worse

    score = (
        12.0 * m['wall_disp_max']           # wall pushed away          → primary penalty
      +  8.0 * m['ee_err_rms']              # RMS tracking error        → secondary
      +  5.0 * m['tau_sat_fraction']        # torque saturation         → controller effort
      +  3.0 * m['delta_max']              # excessive spring deflection
      +  4.0 * (1.0 - m['contact_fraction'])  # reward sustained contact
      +  2.0 * m['q_err_mean']             # link angle error
    )
    return score


# ══════════════════════════════════════════════════════════════════════════════
# GAIN CANDIDATE GENERATORS
# ══════════════════════════════════════════════════════════════════════════════
def _random_candidates(n: int, seed: int = 42) -> list:
    """Random sampling with physically motivated bounds."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        kp1 = rng.uniform(3.0,  80.0)
        kp2 = rng.uniform(2.0,  40.0)
        kd1 = rng.uniform(0.1,   6.0)
        kd2 = rng.uniform(0.05,  3.0)
        out.append((kp1, kp2, kd1, kd2))
    return out

def _grid_coarse() -> list:
    """Coarse grid: 5x4x4x4 = 320 points. Fast landscape survey (~2 min at T=10)."""
    kp1_vals = [5,  15, 30, 50,  70]
    kp2_vals = [2,  8,  20, 38]
    kd1_vals = [0.2, 0.8, 1.8, 3.5]
    kd2_vals = [0.05, 0.3, 0.8, 1.8]
    return list(itertools.product(kp1_vals, kp2_vals, kd1_vals, kd2_vals))

def _grid_dense() -> list:
    """Dense grid: 8x6x5x5 = 1200 points. Comprehensive (~8 min at T=10)."""
    kp1_vals = [3,  8,  15, 25, 40, 55, 70, 80]
    kp2_vals = [2,  5,  12, 22, 32, 40]
    kd1_vals = [0.1, 0.5, 1.2, 2.5, 4.5]
    kd2_vals = [0.05, 0.25, 0.6, 1.2, 2.2]
    return list(itertools.product(kp1_vals, kp2_vals, kd1_vals, kd2_vals))

def _grid_fine(center: tuple, spread: float = 0.4, steps: int = 5) -> list:
    """
    Fine grid zoomed around a known good point (steps^4 = 625 points by default).
    Pass --center kp1,kp2,kd1,kd2 from the best trial found previously.
    spread=0.4 means +/-40% around each centre value.
    """
    kp1_c, kp2_c, kd1_c, kd2_c = center
    def _rng(c, sp, n):
        lo = max(c * (1 - sp), 0.01)
        hi = c * (1 + sp)
        return [round(lo + i * (hi - lo) / (n - 1), 4) for i in range(n)]
    return list(itertools.product(
        _rng(kp1_c, spread, steps), _rng(kp2_c, spread, steps),
        _rng(kd1_c, spread, steps), _rng(kd2_c, spread, steps)
    ))

def generate_candidates(strategy: str, n_random: int,
                        center: tuple = None) -> list:
    if strategy == 'random':
        return _random_candidates(n_random)
    if strategy == 'grid':        # coarse grid — best first pass
        return _grid_coarse()
    if strategy == 'dense':       # thorough — run unattended
        return _grid_dense()
    if strategy == 'fine':        # zoom in around best known point
        if center is None:
            raise ValueError('--strategy fine requires --center kp1,kp2,kd1,kd2')
        return _grid_fine(center)
    if strategy == 'both':
        return _grid_coarse() + _random_candidates(n_random)
    raise ValueError(f"Unknown strategy '{strategy}'. Use: random | grid | dense | fine | both")


# ══════════════════════════════════════════════════════════════════════════════
# CSV FIELD NAMES  (order preserved in output file)
# ══════════════════════════════════════════════════════════════════════════════
FIELDNAMES = [
    'trial', 'score', 'kp1', 'kp2', 'kd1', 'kd2',
    'diverged', 'diverge_time_s', 'steps_completed',
    'ee_err_mean', 'ee_err_max', 'ee_err_rms', 'ee_err_final',
    'q_err_mean', 'q_err_max', 'q_err_std',
    'theta_err_mean', 'theta_err_max',
    'delta_mean', 'delta_max', 'delta_std',
    'tau_mean', 'tau_max', 'tau1_mean', 'tau2_mean', 'tau_sat_fraction',
    'wall_disp_mean', 'wall_disp_max', 'wall_disp_std',
    'contact_fraction', 'contact_steps',
    'elapsed_s',
]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Automated PD gain tuner for ASR robot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategies
----------
  random  – N random trials in the full search space (fast exploration)
  grid    – 320-point coarse grid   (~2 min at T=10)
  dense   – 1200-point dense grid   (~8 min at T=10, run unattended)
  fine    – 625-point zoom around --center kp1,kp2,kd1,kd2
  both    – coarse grid + N random

Typical workflow
----------------
  1. python gain_tuner.py --strategy random --n 50 --T 10
  2. python gain_tuner.py --strategy grid   --T 10
  3. python gain_tuner.py --strategy fine   --T 20 --center 57.25,3.74,1.45,0.90
"""
    )
    parser.add_argument('--strategy', default='random',
                        choices=['random', 'grid', 'dense', 'fine', 'both'],
                        help="Sampling strategy (default: random)")
    parser.add_argument('--n',  type=int, default=50,
                        help="Number of random trials for random/both (default: 50)")
    parser.add_argument('--T',  type=float, default=20.0,
                        help="Simulation duration in seconds per trial (default: 20)")
    parser.add_argument('--out', default='gain_tuning_results.csv',
                        help="Output CSV filename (default: gain_tuning_results.csv)")
    parser.add_argument('--center', default=None,
                        help="Centre gains for fine strategy: kp1,kp2,kd1,kd2")
    args = parser.parse_args()

    center = None
    if args.center:
        try:
            parts = [float(x) for x in args.center.split(',')]
            if len(parts) != 4:
                raise ValueError
            center = tuple(parts)
        except ValueError:
            raise SystemExit("--center must be four comma-separated floats: kp1,kp2,kd1,kd2")
    candidates = generate_candidates(args.strategy, args.n, center=center)
    total      = len(candidates)
    out_path   = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)

    print("=" * 65)
    print(" ASR GAIN TUNER")
    print(f"  Strategy  : {args.strategy}  ({total} trials)")
    print(f"  T_final   : {args.T} s  per trial")
    print(f"  Output    : {out_path}")
    print("=" * 65)

    best_score  = float('inf')
    best_gains  = None
    best_metrics = None

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for i, (kp1, kp2, kd1, kd2) in enumerate(candidates):
            t0 = time.time()
            print(
                f"[{i+1:4d}/{total}]  "
                f"kp=[{kp1:6.2f}, {kp2:6.2f}]  "
                f"kd=[{kd1:5.3f}, {kd2:5.3f}]  ...",
                end='', flush=True
            )

            try:
                m = run_trial(kp1, kp2, kd1, kd2, T_final=args.T)
            except Exception as exc:
                # Don't let one bad trial stop the entire sweep
                print(f"  ERROR: {exc}")
                m = {k: float('nan') for k in FIELDNAMES
                     if k not in ('trial', 'kp1', 'kp2', 'kd1', 'kd2',
                                  'score', 'elapsed_s')}
                m.update(kp1=kp1, kp2=kp2, kd1=kd1, kd2=kd2,
                         diverged=1, steps_completed=0, diverge_time_s=0.0)

            elapsed = time.time() - t0
            score   = composite_score(m)

            row = {'trial': i + 1, 'score': round(score, 6), 'elapsed_s': round(elapsed, 2)}
            row.update(m)
            # Fill any missing keys so csv writer doesn't complain
            for fn in FIELDNAMES:
                row.setdefault(fn, float('nan'))
            writer.writerow(row)
            f.flush()                # write immediately → safe to ctrl-c

            flag = ''
            if score < best_score:
                best_score   = score
                best_gains   = (kp1, kp2, kd1, kd2)
                best_metrics = m
                flag = '  ◄ BEST'

            print(
                f"  score={score:8.4f}  "
                f"wall_max={m.get('wall_disp_max', float('nan')):.4f}m  "
                f"ee_rms={m.get('ee_err_rms', float('nan')):.4f}m  "
                f"contact={m.get('contact_fraction', float('nan')):.0%}  "
                f"{elapsed:.1f}s{flag}"
            )

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(" TUNING COMPLETE")
    print("=" * 65)
    if best_gains:
        kp1, kp2, kd1, kd2 = best_gains
        print(f"\nBest score  : {best_score:.4f}")
        print(f"Best gains  : kp1={kp1:.3f}  kp2={kp2:.3f}  kd1={kd1:.4f}  kd2={kd2:.4f}")
        print("\nKey metrics for best trial:")
        for key in ('ee_err_rms', 'wall_disp_max', 'contact_fraction',
                    'tau_sat_fraction', 'delta_max', 'diverged'):
            val = best_metrics.get(key, float('nan'))
            print(f"  {key:<22s}: {val}")
        print(f"\nPaste into Coursework_student.py:")
        print(f"  k_p = np.diag([{kp1:.2f}, {kp2:.2f}])")
        print(f"  k_d = np.diag([{kd1:.3f}, {kd2:.3f}])")
    print(f"\nFull results saved to:\n  {out_path}")


if __name__ == '__main__':
    main()