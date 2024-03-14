import csv
from math import pi
import numpy as np
import modern_robotics as mr
import matplotlib.pyplot as plt
def rearrange_back(list):
    """
    rearrange the 1x12 vector to a matrix T
    r11, r12, r13, r21, r22, r23, r31, r32, r33, px, py, pz, gripper state
    """
    return np.array(
        [
            [list[0], list[1], list[2], list[9]],
            [list[3], list[4], list[5], list[10]],
            [list[6], list[7], list[8], list[11]],
            [0, 0, 0, 1],
        ]
    )

def state_vector2matrix(state):
    """
    state:(odemetry_new, arm_joint_angles_new, wheel_angles_new, gripper_state_new)
    return matrix T
    """
    M0_e = np.array([[1, 0, 0, 0.033],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0.6546],
                     [0, 0, 0, 1]])

    Blist = np.array([[0, 0, 1, 0, 0.033, 0],
                      [0, -1, 0, -0.5076, 0, 0],
                      [0, -1, 0, -0.3526, 0, 0],
                      [0, -1, 0, -0.2176, 0, 0],
                      [0, 0, 1, 0, 0, 0]]).T

    theta_list = np.array(state[3:8])
    T_b0 = np.array([[1, 0, 0, 0.1662], [0, 1, 0, 0], [0, 0, 1, 0.0026], [0, 0, 0, 1]])
    phi, x, y = state[0], state[1],state[2]
    T_0e = mr.FKinBody(M0_e, Blist, theta_list)
    T_sb = np.array(
        [[np.cos(phi), -np.sin(phi), 0, x], [np.sin(phi), np.cos(phi), 0, y], [0, 0, 1, 0.0963], [0, 0, 0, 1]])
    return  T_sb@T_b0@T_0e
def controller(X_d, X_d_next, current_state, delta_t, K_p, K_i):
    """
    return
    V_b
    cmd
    X_err
    """
    #robot config
    r = 0.0475
    l = 0.235
    w = 0.15

    M0_e = np.array([[1, 0, 0, 0.033],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.6546],
                    [0, 0, 0, 1]])

    Blist = np.array([[0, 0, 1, 0, 0.033, 0],
                      [0, -1, 0, -0.5076, 0, 0],
                      [0, -1, 0, -0.3526, 0, 0],
                      [0, -1, 0, -0.2176, 0, 0],
                      [0, 0, 1, 0, 0, 0]]).T

    theta_list = np.array(current_state[3:8])
    #
    X_d = state_vector2matrix(X_d)
    X_d_next = state_vector2matrix(X_d_next)

    V_d = 1./delta_t * mr.MatrixLog6(np.linalg.inv(X_d)@X_d_next)
    V_d = mr.se3ToVec(V_d)

    # compute X
    X = state_vector2matrix(current_state)

    #
    Ad_V_d = mr.Adjoint(np.linalg.inv(X)@X_d)@V_d
    X_err = mr.MatrixLog6(np.linalg.inv(X)@X_d)
    X_err = mr.se3ToVec(X_err)

    # control
    V_b = Ad_V_d + K_p@X_err + delta_t * K_i@X_err

    # jacobian

    J_arm = mr.JacobianBody(Blist, theta_list)

    F = r/4 * np.array([[0,0,0,0],[0,0,0,0],[-1/(l+w),1/(l+w),1/(l+w),-1/(l+w)],[1,1,1,1],[-1,1,-1,1],[0,0,0,0]])
    T_b0 = np.array([[1, 0, 0, 0.1662], [0, 1, 0, 0], [0, 0, 1, 0.0026], [0, 0, 0, 1]])
    T_0e = mr.FKinBody(M0_e, Blist, theta_list)
    J_base = mr.Adjoint(np.linalg.inv(T_0e)@np.linalg.inv(T_b0))@F
    Je = np.hstack((J_base, J_arm))

    Je_inv = np.linalg.pinv(Je)

    cmd = Je_inv@V_b,
    return V_b, cmd, X_err
# def main():
#     X_d = np.array([[0,0,1,0.5],
#                     [0,1,0,0],
#                     [-1,0,0,0.5],
#                     [0,0,0,1]])
#     X_d_next = np.array([[0,0,1,0.6],
#                          [0,1,0,0],
#                          [-1,0,0,0.3],
#                          [0,0,0,1]])
#     current_state = [0,0,0,0,0,0.2,-1.6,0, ] + [0,0,0,0,0]
#     delta_t = 0.01
#     K_p = np.zeros((6,6))
#     K_i = np.zeros((6,6))
#     V_b, cmd , X_err= controller(X_d, X_d_next, current_state, delta_t, K_p, K_i)
#     print(V_b, cmd)
#
# if __name__ == '__main__':
#     main()