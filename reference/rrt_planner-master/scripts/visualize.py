import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_cartesian_trajectory(path, robot):
    """
    Visualize full robot arm trajectory in Cartesian space.
    Each timestep shows the full joint chain.
    """
    fig = plt.figure()
    fig.set_size_inches(20, 20)
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title("Full Arm Cartesian Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ee_positions = []

    for q in path:
        joint_positions = robot.get_joint_positions(q)
        ee_positions.append(joint_positions[-1])

        ax.plot(joint_positions[:, 0],
                joint_positions[:, 1],
                joint_positions[:, 2],
                '-o', color='lightblue', alpha=0.6)

    ee_positions = np.array(ee_positions)
    ax.plot(ee_positions[:, 0],
            ee_positions[:, 1],
            ee_positions[:, 2],
            '-o', color='red', label='End Effector Trajectory', linewidth=2)

    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([-0.6, 0.6])
    ax.set_zlim([-0.1, 1.1])
    
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
