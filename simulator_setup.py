import numpy as np
import math
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    SpatialInertia,
    UnitInertia,
    RevoluteJoint,
    PrismaticJoint,
    RevoluteSpring,
    FixedOffsetFrame,
    RigidTransform,
    RotationMatrix,
    Simulator,
    Meshcat,
    MeshcatVisualizer,
    Box,
    Cylinder,
    CoulombFriction,
    Sphere,
)

def setup_simulator():
    print("="*60)
    print("ROBOTICS CONTROL COURSEWORK - STUDENT VERSION")
    print("="*60)
    print("\nPART 1: Loading robot and setting up environment...")

    L1, L2 = 1.0, 1.0
    pi = math.pi
    phi1_init = pi * -1 / 2
    phi2_init = pi * 1 / 9
    z0 = 0.0
    axis = [0, 1, 0]
    dt = 0.001
    k1, k2 = 60.5, 60.01
    wall_x = 1.1
    wall_thickness = 0.03
    wall_width_y = 0.70
    wall_height_z = 4.00
    PRINT_SUMMARY = False
    TORQUE_LIMIT = 200.0
    USE_FALSE_TRAJ = False

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=dt)
    plant.mutable_gravity_field().set_gravity_vector([0.0, 0.0, -9.81])

    def make_body_at_com(name: str, mass: float, inertia_radius: float):
        return plant.AddRigidBody(
            name,
            SpatialInertia(
                mass=mass,
                p_PScm_E=[0.0, 0.0, 0.0],
                G_SP_E=UnitInertia.SolidSphere(inertia_radius),
            ),
        )

    link1 = make_body_at_com("link1", mass=1.5, inertia_radius=0.16)        
    link2 = make_body_at_com("link2", mass=1.5, inertia_radius=0.16)
    rotor1 = make_body_at_com("rotor1", mass=0.5, inertia_radius=0.08)
    rotor2 = make_body_at_com("rotor2", mass=0.4, inertia_radius=0.08)

    world_mount = plant.AddFrame(
        FixedOffsetFrame("world_mount", plant.world_frame(), RigidTransform([0, 0, z0]))
    )
    
    link1_base = plant.AddFrame(
        FixedOffsetFrame("link1_base", link1.body_frame(), RigidTransform([-L1/2, 0, 0]))
    )
    link1_tip = plant.AddFrame(
        FixedOffsetFrame("link1_tip", link1.body_frame(), RigidTransform([+L1/2, 0, 0]))
    )
    
    link2_base = plant.AddFrame(
        FixedOffsetFrame("link2_base", link2.body_frame(), RigidTransform([-L2/2, 0, 0]))
    )

    rotor1_center = plant.AddFrame(FixedOffsetFrame("rotor1_center", rotor1.body_frame(), RigidTransform([0,0,0])))

    j_motor1 = plant.AddJoint(RevoluteJoint("motor_joint_1", world_mount, rotor1.body_frame(), axis, damping=0.1))
    plant.AddJointActuator("act_m1", j_motor1)

    j_spring1 = plant.AddJoint(RevoluteJoint("spring_joint_1", rotor1.body_frame(), link1_base, axis, damping=0.0))
    plant.AddForceElement(RevoluteSpring(j_spring1, nominal_angle=0.0, stiffness=k1))

    j_motor2 = plant.AddJoint(RevoluteJoint("motor_joint_2", link1_tip, rotor2.body_frame(), axis, damping=0.1))
    plant.AddJointActuator("act_m2", j_motor2)

    j_spring2 = plant.AddJoint(RevoluteJoint("spring_joint_2", rotor2.body_frame(), link2_base, axis, damping=0.0))
    plant.AddForceElement(RevoluteSpring(j_spring2, nominal_angle=0.0, stiffness=k2))

    X_cyl_to_x = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2))

    plant.RegisterVisualGeometry(
        link1, X_cyl_to_x, Cylinder(0.03, L1),
        "link1_vis", np.array([0.90, 0.90, 0.85, 1.0])
    )
    plant.RegisterCollisionGeometry(
        link1, X_cyl_to_x, Cylinder(0.03, L1),
        "link1_col", CoulombFriction(0.8, 0.6),
    )
    plant.RegisterVisualGeometry(
        link2, X_cyl_to_x, Cylinder(0.025, L2),
        "link2_vis", np.array([0.90, 0.90, 0.85, 1.0])
    )
    plant.RegisterCollisionGeometry(
        link2, X_cyl_to_x, Cylinder(0.015, L2),
        "link2_col", CoulombFriction(0.8, 0.6),
    )
    plant.RegisterVisualGeometry(
        rotor1, RigidTransform(), Sphere(0.05),
        "rotor1_vis", np.array([0.20, 0.80, 1.00, 1.0])
    )
    plant.RegisterVisualGeometry(
        rotor2, RigidTransform(), Sphere(0.045),
        "rotor2_vis", np.array([0.20, 0.80, 1.00, 1.0])
    )

    wall_mass = 150.0
    wall_damping = 180.0
    wall_k = 0.0
    wall_damping_extra = 350.0
    wall_mu_static = 0.6
    wall_mu_coulomb = 0.45
    wall_fric_normal = 800.0
    wall_v_stiction = 0.005

    wall_body = plant.AddRigidBody(
        "wall_body",
        SpatialInertia(
            mass=wall_mass,
            p_PScm_E=[0.0, 0.0, 0.0],
            G_SP_E=UnitInertia.SolidBox(wall_thickness, wall_width_y, wall_height_z),
        )
    )

    wall_origin_frame = plant.AddFrame(
        FixedOffsetFrame("wall_origin", plant.world_frame(), RigidTransform([wall_x, 0.0, 0.0]))
    )

    j_wall = plant.AddJoint(
        PrismaticJoint("wall_joint", wall_origin_frame, wall_body.body_frame(), [1, 0, 0], damping=wall_damping)
    )
    plant.AddJointActuator("act_wall", j_wall)

    j_wall.set_position_limits([0.0], [np.inf])

    wall_shape = Box(wall_thickness, wall_width_y, wall_height_z)

    plant.RegisterVisualGeometry(
        wall_body, RigidTransform(), wall_shape,
        "right_wall_vis", np.array([0.60, 0.65, 0.75, 1.0])
    )

    plant.RegisterCollisionGeometry(
        wall_body, RigidTransform(), wall_shape,
        "right_wall_col", CoulombFriction(0.9, 0.7)
    )

    plant.Finalize()

    meshcat = Meshcat()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()
    sim = Simulator(diagram)
    context = sim.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    j_motor1.set_angle(plant_context, phi1_init)
    j_motor2.set_angle(plant_context, phi2_init)
    j_spring1.set_angle(plant_context, 0.0)
    j_spring2.set_angle(plant_context, 0.0)

    plant.get_actuation_input_port().FixValue(plant_context, [0.0, 0.0, 0.0])

    sim.Initialize()

    print("✓ Robot model loaded successfully")

    return {
        'sim': sim,
        'plant': plant,
        'plant_context': plant_context,
        'j_motor1': j_motor1,
        'j_motor2': j_motor2,
        'j_spring1': j_spring1,
        'j_spring2': j_spring2,
        'j_wall': j_wall,
        'L1': L1,
        'L2': L2,
        'phi1_init': phi1_init,
        'phi2_init': phi2_init,
        'k1': k1,
        'k2': k2,
        'wall_x': wall_x,
        'wall_thickness': wall_thickness,
        'wall_k': wall_k,
        'wall_damping_extra': wall_damping_extra,
        'wall_fric_normal': wall_fric_normal,
        'wall_mu_static': wall_mu_static,
        'wall_mu_coulomb': wall_mu_coulomb,
        'wall_v_stiction': wall_v_stiction,
        'TORQUE_LIMIT': TORQUE_LIMIT,
        'USE_FALSE_TRAJ': USE_FALSE_TRAJ,
        'PRINT_SUMMARY': PRINT_SUMMARY
    }
