import modern_robotics as mr
import numpy as np
from module import motion_model, trajectory_generatator


class Robot:
    def __init__(self):
        M_0e = np.array(
            [[1, 0, 0, 0.033], [0, 1, 0, 0], [0, 0, 1, 0.6546], [0, 0, 0, 1]]
        )
        Tb_0 = np.array(
            [[1, 0, 0, 0.1662], [0, 1, 0, 0], [0, 0, 1, 0.0026], [0, 0, 0, 1]]
        )
        x = 0
        y = 0
        phi = 0
        T_sb = np.array(
            [
                [np.cos(phi), -np.sin(phi), 0, x],
                [np.sin(phi), np.cos(phi), 0, y],
                [0, 0, 1, 0.0963],
                [0, 0, 0, 1],
            ]
        )
        T0_e = np.dot(Tb_0, M_0e)
        T_se = np.dot(T_sb, T0_e)
        Tse_initial = T_se
        s45 = np.sin(np.pi / 4)

        self.Tsc_initial = np.array(
            [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0.025], [0, 0, 0, 1]]
        )  # Initial configuration of the cube
        self.Tsc_final = np.array(
            [[0, 1, 0, 0], [-1, 0, 0, -1], [0, 0, 1, 0.025], [0, 0, 0, 1]]
        )  # Final configuration of the cube

        self.Tce_grasp = np.array(
            [[-s45, 0, s45, 0], [0, 1, 0, 0], [-s45, 0, -s45, 0], [0, 0, 0, 1]]
        )  # Configuration of the end-effector relative to the cube while grasping
        self.Tce_standoff = np.array(
            [[-s45, 0, s45, 0], [0, 1, 0, 0], [-s45, 0, -s45, 0.05], [0, 0, 0, 1]]
        )  # Standoff configuration of the end-effector above the cube

    def next_state(
        self,
        current_state,
        joint_and_wheel_velocities,
        delta_t,
        max_joint_and_wheel_velocity,
    ):
        return motion_model.next_state(
            current_state,
            joint_and_wheel_velocities,
            delta_t,
            max_joint_and_wheel_velocity,
        )

    def TrajectoryGenerator(self):
        return trajectory_generatator.TrajectoryGenerator(
            self.Tse_initial,
            self.Tsc_initial,
            self.Tsc_final,
            self.Tce_grasp,
            self.Tce_standoff,
            self.k,
        )
