import modern_robotics as mr
import numpy as np
from module import motion_model, trajectory_generatator, feedback_control
from module.trajectory_generatator import state2config
class Robot:
    def __init__(self, K_p, K_i):
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
        Tb_e = np.dot(Tb_0, M_0e)
        T_se = np.dot(T_sb, Tb_e)
        # self.Tse_initial = T_se
        s45 = np.sin(np.pi / 4)
        
        
        self.initial_config = np.array([0.5, -0.2, 0.1, 0, 0, 0.2, -1.6, 0, 0, 0, 0, 0, 0])
        self.Tse_initial = state2config(self.initial_config)






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

        self.delta_t = 0.01

        self.K_p = K_p
        self.K_i = K_i
        self.X_err_prev_intergral = 0
        self.k = 1

    def next_state(
        self,
        current_state,
        joint_and_wheel_velocities,
        delta_t,
        max_arm,
        max_wheel
    ):
        return motion_model.NextState(
            current_state,
            joint_and_wheel_velocities,
            delta_t,
            max_arm,
            max_wheel
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

    def controller(self, X_d, X_d_next, current_state):

        V_b, cmd , X_err, X_err_intergral =  feedback_control.controller_1(
            X_d,
            X_d_next,
            current_state,
            self.delta_t,
            self.K_p,
            self.K_i,
            self.X_err_prev_intergral
        )
        self.X_err_prev_intergral = X_err_intergral
        return V_b, cmd , X_err