import csv
from math import pi
import numpy as np
import modern_robotics as mr
import matplotlib.pyplot as plt


def controller(X_d, X_d_next, curr_state, delta_t, K_p, K_i):
    '''
    Input:
    X_d: desired configuration
    X_d_next: desired configuration at next time step
    X: ee current configuration
    curr_state: current state of the robot: "odemetry, arm joint angles, wheel angles, gripper state"
    delta_t: time step
    K_p: proportional gain
    K_i: integral gain
    Output:
    V_b: desired body twist
    '''
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
    
    X = state2config(curr_state)
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
    
    #extract the joint thetalist from the current state
    theta_list = curr_state[3:8]

    J_arm = mr.JacobianBody(Blist, theta_list)

    F = r/4 * np.array([[0,0,0,0],[0,0,0,0],[-1/(l+w),1/(l+w),1/(l+w),-1/(l+w)],[1,1,1,1],[-1,1,-1,1],[0,0,0,0]])
    T_0e = mr.FKinBody(M_0e, Blist, theta_list)
    J_base = mr.Adjoint(np.linalg.inv(T_0e)@np.linalg.inv(T_b0))@F

    Je = np.hstack((J_base, J_arm))
    Je_inv = np.linalg.pinv(Je)

    cmd = Je_inv@V_b
    print('control input',cmd)
    print('V_b_',V_b)
    print('Je',Je)
    return V_b, cmd , X_err

def state2config(curr_state):
    odemetry = curr_state[0:3]
    arm_joint_angles = curr_state[3:8]
    M_0e = np.array([[1, 0, 0, 0.033], [0, 1, 0, 0], [0, 0, 1, 0.6546], [0, 0, 0, 1]])

    # Jacobian matrices
    Blist = np.array([[0, 0, 1, 0, 0.033, 0],
                      [0, -1, 0, -0.5076, 0, 0],
                      [0, -1, 0, -0.3526, 0, 0],
                      [0, -1, 0, -0.2176, 0, 0],
                      [0, 0, 1, 0, 0, 0]]).T

    theta_list = arm_joint_angles  # Extract joint angles
    J_arm = mr.JacobianBody(Blist, theta_list)
    r = 0.0475
    l = 0.235
    w = 0.15
    T0_e = mr.FKinBody(M_0e, Blist, theta_list)

    Tb_0 = np.array([[1, 0, 0, 0.1662], [0, 1, 0, 0], [0, 0, 1, 0.0026], [0, 0, 0, 1]])
    x = odemetry[0]
    y = odemetry[1]
    phi = odemetry[2]
    T_sb = np.array([[np.cos(phi), -np.sin(phi), 0, x], [np.sin(phi), np.cos(phi), 0, y], [0, 0, 1, 0.0963], [0, 0, 0, 1]])
    T_se = np.dot(T_sb, Tb_0)@T0_e

    return T_se




def controller_0(X_d, X_d_next, curr_state, delta_t, K_p, K_i):
    '''
    This is a tolerence mition 
    Input:
    X_d: desired configuration
    X_d_next: desired configuration at next time step
    curr_state: current state of the robot: "odemetry, arm joint angles, wheel angles, gripper state"
    delta_t: time step
    K_p: proportional gain
    K_i: integral gain
    Output:
    V_b: desired body twist
    '''
    
    # Robot configuration
    r = 0.0475
    l = 0.235
    w = 0.15

    X = state2config(curr_state)

    # Transform matrices
    T_b0 = np.array([[1, 0, 0, 0.1662],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0.0026],
                     [0, 0, 0, 1]])

    M_0e = np.array([[1, 0, 0, 0.033],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0.6546],
                     [0, 0, 0, 1]])

    # Compute feedforward velocity and error twist
    V_d = 1./delta_t * mr.MatrixLog6(np.linalg.inv(X_d)@X_d_next)
    V_d = mr.se3ToVec(V_d)
    Ad_V_d = mr.Adjoint(np.linalg.inv(X)@X_d)@V_d

    X_err = mr.MatrixLog6(np.linalg.inv(X)@X_d)
    X_err = mr.se3ToVec(X_err)

    # Control law
    V_b = Ad_V_d + K_p@X_err + delta_t * K_i@X_err

    # Jacobian matrices
    Blist = np.array([[0, 0, 1, 0, 0.033, 0],
                      [0, -1, 0, -0.5076, 0, 0],
                      [0, -1, 0, -0.3526, 0, 0],
                      [0, -1, 0, -0.2176, 0, 0],
                      [0, 0, 1, 0, 0, 0]]).T

    theta_list = curr_state[3:8]  # Extract joint angles
    J_arm = mr.JacobianBody(Blist, theta_list)
    
    F = r/4 * np.array([[0,0,0,0],[0,0,0,0],[-1/(l+w),1/(l+w),1/(l+w),-1/(l+w)],[1,1,1,1],[-1,1,-1,1],[0,0,0,0]])
    T_0e = mr.FKinBody(M_0e, Blist, theta_list)
    J_base = mr.Adjoint(np.linalg.inv(T_0e)@np.linalg.inv(T_b0))@F

    Je = np.hstack((J_base, J_arm))
    
    # Compute the pseudoinverse with a tolerance to mitigate singularity issues
    U, S, V_T = np.linalg.svd(Je)
    tolerance = 0.0001  # Set a tolerance threshold
    S_inv = np.array([1/s if s > tolerance else 0 for s in S])
    S_inv_padded = np.zeros_like(Je.T)  # Create a zero matrix with the transpose shape of Je
    min_dim = min(Je.shape)  # Minimum dimension of Je
    S_inv_padded[:min_dim, :min_dim] = S_inv  # Place S_inv into the top-left corner

    # Now compute the pseudoinverse with correct dimension handling
    Je_inv = V_T.T @ S_inv_padded @ U.T

    # Compute the commanded velocities
    cmd = Je_inv @ V_b
    print('control input',cmd)
    print('V_b_',V_b)
    print('Je',Je)


    return V_b, cmd

def controller_1(X_d, X_d_next,curr_state, delta_t, K_p, K_i):
    '''
    This is the selective weighted pseudoinverse method

    Input:
    X_d: desired configuration
    X_d_next: desired configuration at next time step
    curr_state: current state of the robot (odometry, arm joint angles, wheel angles, gripper state)
    delta_t: time step
    K_p: proportional gain matrix
    K_i: integral gain matrix
    Output:
    V_b: desired body twist
    cmd: commanded velocities for wheels and arm joints
    '''
    # Robot configuration
    r = 0.0475
    l = 0.235
    w = 0.15

    X = state2config(curr_state)

    # Transformation matrices
    T_b0 = np.array([[1, 0, 0, 0.1662],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0.0026],
                     [0, 0, 0, 1]])

    M_0e = np.array([[1, 0, 0, 0.033],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0.6546],
                     [0, 0, 0, 1]])

    # Compute the feedforward velocity and the error twist
    V_d = 1./delta_t * mr.MatrixLog6(np.linalg.inv(X_d) @ X_d_next)
    V_d = mr.se3ToVec(V_d)
    Ad_V_d = mr.Adjoint(np.linalg.inv(X) @ X_d) @ V_d

    X_err = mr.MatrixLog6(np.linalg.inv(X) @ X_d)
    X_err = mr.se3ToVec(X_err)

    # Control law
    V_b = Ad_V_d + K_p @ X_err + delta_t * K_i @ X_err

    # Jacobian for the arm and base
    Blist = np.array([[0, 0, 1, 0, 0.033, 0],
                      [0, -1, 0, -0.5076, 0, 0],
                      [0, -1, 0, -0.3526, 0, 0],
                      [0, -1, 0, -0.2176, 0, 0],
                      [0, 0, 1, 0, 0, 0]]).T
    theta_list = curr_state[3:8]  # Extract the joint angles
    J_arm = mr.JacobianBody(Blist, theta_list)

    F = r / 4 * np.array([[0, 0, 0, 0], [0, 0, 0, 0], [-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)], [1, 1, 1, 1], [-1, 1, -1, 1], [0, 0, 0, 0]])
    T_0e = mr.FKinBody(M_0e, Blist, theta_list)
    J_base = mr.Adjoint(np.linalg.inv(T_0e) @ np.linalg.inv(T_b0)) @ F

    Je = np.hstack((J_base, J_arm))

    # Selectively Weighted Pseudoinverse Calculation

    # Define weighting matrices 
    W_base = np.diag([1, 1, 1, 1] + [0.1] * 5)  # Higher weight for base movement
    W_arm = np.diag([0.1] * 4 + [1, 1, 1, 1, 1])  # Higher weight for arm movement
    
    # Adjust weighting based on your task preference or dynamic conditions
    W = W_base  # This example prioritizes base movement

    Je_weighted = Je @ W

    # Compute the pseudoinverse with a tolerance to mitigate singularity issues

    U, S, V_T = np.linalg.svd(Je_weighted)
    tolerance = 0.01  # Set a tolerance threshold
    S_inv = np.array([1/s if s > tolerance else 0 for s in S])

    # Correctly shape S_inv for non-square matrices
    S_inv_padded = np.zeros_like(Je_weighted.T)  # Create a zero matrix with the transpose shape of Je
    min_dim = min(Je_weighted.shape)  # Minimum dimension of Je
    S_inv_padded[:min_dim, :min_dim] = S_inv  # Place S_inv into the top-left corner

    Je_inv_weighted = V_T.T @ S_inv_padded @ U.T

    cmd = Je_inv_weighted @ V_b

    print('control input',cmd)
    print('V_b_',V_b)
    print('Je_weighted',Je_weighted)
    print('Je',Je)

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
    
    curren_state = np.array([0, 0, 0, 0, 0, 0.2, -1.6, 0,0,0,0,0,0])
    # X_est = state2config(curren_state)
    # print(X_est)

    delta_t = 0.01
    K_p = np.zeros((6,6))
    K_i = np.zeros((6,6))
    V_b, cmd = controller(X_d, X_d_next,curren_state, delta_t, K_p, K_i)
    # print(V_b, cmd)

if __name__ == '__main__':
    main()