import numpy as np
import math
import os

def plot_results(time_log, q_log, q_des_log, qd_log, qd_des_log, qdd_log, qdd_des_log, 
                 delta_log, delta_des_log, theta_log, theta_des_log, thetadot_log, 
                 thetadot_des_log, ee_log, ee_des_log, tau_log, tau_link_log, contact_log, PRINT_SUMMARY):
    try:
        import matplotlib.pyplot as plt

        time_array = np.array(time_log)
        q_array = np.array(q_log)
        q_des_array = np.array(q_des_log)
        qd_array = np.array(qd_log)
        qd_des_array = np.array(qd_des_log)
        qdd_array = np.array(qdd_log)
        qdd_des_array = np.array(qdd_des_log)
        delta_array = np.array(delta_log)
        delta_des_array = np.array(delta_des_log)
        theta_array = np.array(theta_log)
        theta_des_array = np.array(theta_des_log)
        thetadot_array = np.array(thetadot_log)
        thetadot_des_array = np.array(thetadot_des_log)
        ee_array = np.array(ee_log)
        ee_des_array = np.array(ee_des_log)
        tau_array = np.array(tau_log)
        tau_link_array = np.array(tau_link_log)

        def downsample_indices(length, max_points=5000):
            if length <= max_points:
                return np.arange(length)
            return np.unique(np.linspace(0, length - 1, max_points).astype(int))

        idx = downsample_indices(len(time_array))
        time_array = time_array[idx]
        q_array = q_array[idx]
        q_des_array = q_des_array[idx]
        qd_array = qd_array[idx]
        qd_des_array = qd_des_array[idx]
        qdd_array = qdd_array[idx]
        qdd_des_array = qdd_des_array[idx]
        delta_array = delta_array[idx]
        delta_des_array = delta_des_array[idx]
        theta_array = theta_array[idx]
        theta_des_array = theta_des_array[idx]
        thetadot_array = thetadot_array[idx]
        thetadot_des_array = thetadot_des_array[idx]
        tau_array = tau_array[idx]
        tau_link_array = tau_link_array[idx]

        def set_pi_ticks(ax, data, step):
            finite_data = data[np.isfinite(data)]
            if finite_data.size == 0:
                return
            min_val = np.min(finite_data)
            max_val = np.max(finite_data)
            tick_min = np.floor(min_val / step) * step
            tick_max = np.ceil(max_val / step) * step
            if tick_max <= tick_min:
                tick_max = tick_min + step
            max_ticks = 200
            span_ticks = int(round((tick_max - tick_min) / step)) + 1
            if span_ticks > max_ticks:
                return
            ticks = np.arange(tick_min, tick_max + step * 0.5, step)
            ax.set_yticks(ticks)
            ax.set_ylim(tick_min, tick_max)
            tick_labels = []
            den = max(1, int(round(math.pi / step)))
            for t in ticks:
                if abs(t) < 1e-9:
                    tick_labels.append("0")
                    continue
                num = int(round(t / math.pi * den))
                if num == den:
                    tick_labels.append("π")
                    continue
                if num == -den:
                    tick_labels.append("-π")
                    continue
                g = math.gcd(abs(num), den)
                num_red = num // g
                den_red = den // g
                if den_red == 1:
                    tick_labels.append(f"{num_red}π")
                else:
                    if num_red == 1:
                        tick_labels.append(f"π/{den_red}")
                    elif num_red == -1:
                        tick_labels.append(f"-π/{den_red}")
                    else:
                        tick_labels.append(f"{num_red}π/{den_red}")
            ax.set_yticklabels(tick_labels)

        fig = plt.figure(figsize=(14, 18))
        fig.suptitle('2-DOF Planar Manipulator Results', fontsize=14, fontweight='bold')
        gs = fig.add_gridspec(5, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(ee_array[:, 0], ee_array[:, 1], 'b-', linewidth=2, label='Actual trajectory')
        
        contact_array = np.array(contact_log) if contact_log else np.empty((0, 2))
        if len(contact_array) > 0:
            contact_idx = downsample_indices(len(contact_array), max_points=2000)
            contact_array = contact_array[contact_idx]
        if len(contact_array) > 0:
            ax1.plot(contact_array[:, 0], contact_array[:, 1], marker='o', color='purple', linestyle='None', markersize=3, label='Wall Contact')
            
        ax1.plot(ee_des_array[:, 0], ee_des_array[:, 1], 'r--', linewidth=1.5, label='Desired trajectory')
        ax1.plot(ee_array[0, 0], ee_array[0, 1], 'go', markersize=10, label='Start')
        ax1.plot(ee_array[-1, 0], ee_array[-1, 1], 'rs', markersize=10, label='End')
        x_des_mark, z_des_mark = ee_des_array[-1, 0], ee_des_array[-1, 1]
        ax1.plot(x_des_mark, z_des_mark, 'k*', markersize=15, label='Desired final')
        ax1.set_xlabel('X Position (m)', fontsize=11)
        ax1.set_ylabel('Z Position (m)', fontsize=11)
        ax1.set_title('XZ Plane', fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time_array, tau_array[:, 0], 'b-', linewidth=2, label='tau1 (motor)')
        ax2.plot(time_array, tau_array[:, 1], 'r-', linewidth=2, label='tau2 (motor)')
        ax2.plot(time_array, tau_link_array[:, 0], 'b--', linewidth=1.5, label='tau_link1')
        ax2.plot(time_array, tau_link_array[:, 1], 'r--', linewidth=1.5, label='tau_link2')
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('Torque (N·m)', fontsize=11)
        ax2.set_title('Joint Torques', fontsize=12, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(time_array, q_array[:, 0], 'b-', linewidth=2, label='q1 actual')
        ax3.plot(time_array, q_des_array[:, 0], 'b--', linewidth=1.5, label='q1 desired')
        ax3.plot(time_array, theta_array[:, 0], 'r-', linewidth=1.5, label='theta1 motor')
        ax3.plot(time_array, theta_des_array[:, 0], 'r--', linewidth=1.5, label='theta1 desired')
        ax3.set_ylabel('q1 (rad)', fontsize=11)
        ax3.set_title('Joint 1 Angle', fontsize=12, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        set_pi_ticks(ax3, np.hstack([q_array[:, 0], theta_array[:, 0], theta_des_array[:, 0], q_des_array[:, 0]]), math.pi / 4)

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(time_array, q_array[:, 1], 'b-', linewidth=2, label='q2 actual')
        ax4.plot(time_array, q_des_array[:, 1], 'b--', linewidth=1.5, label='q2 desired')
        ax4.plot(time_array, theta_array[:, 1], 'r-', linewidth=1.5, label='theta2 motor')
        ax4.plot(time_array, theta_des_array[:, 1], 'r--', linewidth=1.5, label='theta2 desired')
        ax4.set_ylabel('q2 (rad)', fontsize=11)
        ax4.set_title('Joint 2 Angle', fontsize=12, fontweight='bold')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        set_pi_ticks(ax4, np.hstack([q_array[:, 1], theta_array[:, 1], theta_des_array[:, 1], q_des_array[:, 1]]), math.pi / 4)

        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(time_array, qd_array[:, 0], 'b-', linewidth=2, label='qd1 actual')
        ax5.plot(time_array, qd_des_array[:, 0], 'b--', linewidth=1.5, label='qd1 desired')
        ax5.plot(time_array, thetadot_array[:, 0], 'r-', linewidth=1.5, label='thetadot1 motor')
        ax5.plot(time_array, thetadot_des_array[:, 0], 'r--', linewidth=1.5, label='thetadot1 desired')
        ax5.set_xlabel('Time (s)', fontsize=11)
        ax5.set_ylabel('qd1 (rad/s)', fontsize=11)
        ax5.set_title('Joint 1 Velocity', fontsize=12, fontweight='bold')
        ax5.legend(loc='best')
        ax5.grid(True, alpha=0.3)
        set_pi_ticks(ax5, np.hstack([qd_array[:, 0], thetadot_array[:, 0], thetadot_des_array[:, 0], qd_des_array[:, 0]]), math.pi / 4)

        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(time_array, qd_array[:, 1], 'b-', linewidth=2, label='qd2 actual')
        ax6.plot(time_array, qd_des_array[:, 1], 'b--', linewidth=1.5, label='qd2 desired')
        ax6.plot(time_array, thetadot_array[:, 1], 'r-', linewidth=1.5, label='thetadot2 motor')
        ax6.plot(time_array, thetadot_des_array[:, 1], 'r--', linewidth=1.5, label='thetadot2 desired')
        ax6.set_xlabel('Time (s)', fontsize=11)
        ax6.set_ylabel('qd2 (rad/s)', fontsize=11)
        ax6.set_title('Joint 2 Velocity', fontsize=12, fontweight='bold')
        ax6.legend(loc='best')
        ax6.grid(True, alpha=0.3)
        set_pi_ticks(ax6, np.hstack([qd_array[:, 1], thetadot_array[:, 1], thetadot_des_array[:, 1], qd_des_array[:, 1]]), math.pi / 4)

        ax7 = fig.add_subplot(gs[3, 0])
        ax7.plot(time_array, delta_array[:, 0], 'b-', linewidth=2, label='delta1 actual')
        ax7.plot(time_array, delta_des_array[:, 0], 'r--', linewidth=1.5, label='delta1 desired')
        ax7.set_xlabel('Time (s)', fontsize=11)
        ax7.set_ylabel('delta1 (rad)', fontsize=11)
        ax7.set_title('Joint 1 Deflection', fontsize=12, fontweight='bold')
        ax7.legend(loc='best')
        ax7.grid(True, alpha=0.3)
        set_pi_ticks(ax7, np.hstack([delta_array[:, 0], delta_des_array[:, 0]]), math.pi / 4)

        ax8 = fig.add_subplot(gs[3, 1])
        ax8.plot(time_array, delta_array[:, 1], 'b-', linewidth=2, label='delta2 actual')
        ax8.plot(time_array, delta_des_array[:, 1], 'r--', linewidth=1.5, label='delta2 desired')
        ax8.set_xlabel('Time (s)', fontsize=11)
        ax8.set_ylabel('delta2 (rad)', fontsize=11)
        ax8.set_title('Joint 2 Deflection', fontsize=12, fontweight='bold')
        ax8.legend(loc='best')
        ax8.grid(True, alpha=0.3)
        set_pi_ticks(ax8, np.hstack([delta_array[:, 1], delta_des_array[:, 1]]), math.pi / 4)

        ax9 = fig.add_subplot(gs[4, 0])
        if len(contact_array) > 0:
            ax9.plot(contact_array[:, 0], contact_array[:, 1], 'r.', markersize=2, label='Contact Pts')
            if len(contact_array) > 5:
                 window = 10
                 if len(contact_array) > window:
                     weights = np.repeat(1.0, window)/window
                     x_smooth = np.convolve(contact_array[:, 0], weights, 'valid')
                     z_smooth = np.convolve(contact_array[:, 1], weights, 'valid')
                     ax9.plot(x_smooth, z_smooth, 'k-', linewidth=1.5, label='Smoothed Path')
            ax9.set_xlabel('Penetration X (m)', fontsize=11)
            ax9.set_xlim(0, 3)
            ax9.set_ylabel('Height Z (m)', fontsize=11)
            ax9.set_title('Wall Contact Profile', fontsize=12, fontweight='bold')
            ax9.legend(loc='best')
        else:
            ax9.text(0.5, 0.5, "No Contact Detected", ha='center', va='center')
            ax9.set_title('Wall Contact Profile', fontsize=12, fontweight='bold')
        ax9.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.expanduser("/home/marino/Desktop/CWRobotDynamics/RDC_coursework/control_results.png"), dpi=150)

        if PRINT_SUMMARY:
            print("✓ Plots generated and saved successfully")
            ee_errors = np.linalg.norm(ee_array - ee_des_array, axis=1)
            print("EE tracking error (mean): {:.4f} m".format(np.mean(ee_errors)))
            print("EE tracking error (max):  {:.4f} m".format(np.max(ee_errors)))
            print("EE tracking error (final): {:.4f} m".format(ee_errors[-1]))
    except ImportError:
        print("○ Matplotlib not available - skipping plots")
    except Exception as e:
        print(f"✗ Error generating plots: {e}")
