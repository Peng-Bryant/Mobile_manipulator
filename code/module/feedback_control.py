import csv
from math import pi
import numpy as np
import modern_robotics as mr
import matplotlib.pyplot as plt
def controller(X_d, X_d_next, X, delta_t, K_p, K_i):
    #robot config
    r = 0.0475
    l = 0.235
    w = 0.15

    T_b0 = np.array([[1, 0, 0, 0.1662],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.0026],
                    [0, 0, 0, 1]])

    M_0e = np.array([[1, 0, 0, 0.033],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.6546],
                    [0, 0, 0, 1]])

    #
    V_d = 1./delta_t * mr.MatrixLog6(np.linalg.inv(X_d)@X_d_next)
    V_d = mr.se3ToVec(V_d)
    Ad_V_d = mr.Adjoint(np.linalg.inv(X)@X_d)@V_d
    X_err = mr.MatrixLog6(np.linalg.inv(X)@X_d)
    X_err = mr.se3ToVec(X_err)

    # control
    V_b = Ad_V_d + K_p@X_err + delta_t * K_i@X_err

    # jacobian
    Blist = np.array([[0, 0, 1, 0, 0.033, 0],
                      [0, -1, 0, -0.5076, 0, 0],
                      [0, -1, 0, -0.3526, 0, 0],
                      [0, -1, 0, -0.2176, 0, 0],
                      [0, 0, 1, 0, 0, 0]]).T
    theta_list = np.array([0,0,0.2,-1.6, 0])
    J_arm = mr.JacobianBody(Blist, theta_list)

    F = r/4 * np.array([[0,0,0,0],[0,0,0,0],[-1/(l+w),1/(l+w),1/(l+w),-1/(l+w)],[1,1,1,1],[-1,1,-1,1],[0,0,0,0]])
    T_0e = mr.FKinBody(M_0e, Blist, theta_list)
    J_base = mr.Adjoint(np.linalg.inv(T_0e)@np.linalg.inv(T_b0))@F
    Je = np.hstack((J_base, J_arm))
    Je_inv = np.linalg.pinv(Je)

    cmd = Je_inv@V_b
    return V_b, cmd
def main():
    X_d = np.array([[0,0,1,0.5],
                    [0,1,0,0],
                    [-1,0,0,0.5],
                    [0,0,0,1]])
    X_d_next = np.array([[0,0,1,0.6],
                         [0,1,0,0],
                         [-1,0,0,0.3],
                         [0,0,0,1]])
    X = np.array([[0.170,0,0.985,0.387],
                  [0,1,0,0],
                  [-0.985,0,0.170,0.570],
                  [0,0,0,1]]) # calculated from config
    delta_t = 0.01
    K_p = np.zeros((6,6))
    K_i = np.zeros((6,6))
    V_b, cmd = controller(X_d, X_d_next, X, delta_t, K_p, K_i)
    print(V_b, cmd)

if __name__ == '__main__':
    main()