import csv
from math import pi
import numpy as np
import modern_robotics as mr
import matplotlib.pyplot as plt

Tsc_initial_ = np.array(
    [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0.025], [0, 0, 0, 1]]
)  # Initial configuration of the cube
Tsc_final_ = np.array(
    [[0, 1, 0, 0], [-1, 0, 0, -1], [0, 0, 1, 0.025], [0, 0, 0, 1]]
)  # Final configuration of the cube

def controller(X_d, X_d_next, curr_state, delta_t, K_p, K_i, X_err_prev_intergral):
    """
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


    Kp, Ki = (2,0.1)
    """
    # robot config
    r = 0.0475
    l = 0.235
    w = 0.15

    T_b0 = np.array([[1, 0, 0, 0.1662], [0, 1, 0, 0], [0, 0, 1, 0.0026], [0, 0, 0, 1]])

    M_0e = np.array([[1, 0, 0, 0.033], [0, 1, 0, 0], [0, 0, 1, 0.6546], [0, 0, 0, 1]])

    X = state2config(curr_state)
    #
    V_d = (1.0 / delta_t) * mr.MatrixLog6(np.linalg.inv(X_d) @ X_d_next)
    V_d = mr.se3ToVec(V_d)
    Ad_V_d = mr.Adjoint(np.linalg.inv(X) @ X_d) @ V_d

    X_err = mr.MatrixLog6(np.linalg.inv(X) @ X_d)
    X_err = mr.se3ToVec(X_err)

    X_err_intergral = X_err_prev_intergral + X_err * delta_t
    # control
    # if feedforward:
    # V_b = Ad_V_d + K_p@X_err + K_i@X_err_intergral
    # if feedback:

    V_b = K_p @ X_err + K_i @ X_err_intergral

    # jacobian
    Blist = np.array(
        [
            [0, 0, 1, 0, 0.033, 0],
            [0, -1, 0, -0.5076, 0, 0],
            [0, -1, 0, -0.3526, 0, 0],
            [0, -1, 0, -0.2176, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]
    ).T

    # extract the joint thetalist from the current state
    theta_list = curr_state[3:8]

    J_arm = mr.JacobianBody(Blist, theta_list)

    F = (
        r
        / 4
        * np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1 / (l + w), 1 / (l + w), 1 / (l + w), -1 / (l + w)],
                [1, 1, 1, 1],
                [-1, 1, -1, 1],
                [0, 0, 0, 0],
            ]
        )
    )
    T_0e = mr.FKinBody(M_0e, Blist, theta_list)
    J_base = mr.Adjoint(np.linalg.inv(T_0e) @ np.linalg.inv(T_b0)) @ F

    Je = np.hstack((J_base, J_arm))
    Je_inv = np.linalg.pinv(Je)

    cmd = Je_inv @ V_b
    print("control input", cmd)
    print("V_b_", V_b)
    print("Je", Je)
    return V_b, cmd, X_err, X_err_intergral


def FeedbackControl(Xd, Xdnext, curr_state, timestep, Kp, Ki, Xerr_integral):
    """
    The function calculates the task-space feedforward plus feedback control law.

    Inputs:
    • The current actual end-effector configuration X (i.e. Tse)
    • The current reference end-effector configuration Xd (i.e. Tse,d)
    • The reference end-effector configuration at the next timestep, Xdnext (aka Tse,d,next)
    • The PI gain matrices Kp and Ki
    • The timestep ∆t between reference trajectory configurations
    • The joint angles of the manipulator : thetalist
    • The integral of error in the end-effector configuration over the time :Xerr_integral

    Outputs:Je, , Xerr, Xerr_integral
    • The desired body twist Vd
    • The commanded end-effector twist V expressed in the end-effector frame {e}
    • The mobile manipulator Jacobian Je(θ): Je
    • the wheel and joints velocities (u, θdot) :u_theta_dot
    • The end effector configuration error Xerr
    • The integral of end effector configuration error Xerr_integral


    para
    Ki,Kp = (2,0)(2,0.01)
    """
    thetalist = curr_state[3:8]
    X = state2config(curr_state)
    # Setting the environment
    Blist = np.array(
        [
            [0, 0, 1, 0, 0.033, 0],
            [0, -1, 0, -0.5076, 0, 0],
            [0, -1, 0, -0.3526, 0, 0],
            [0, -1, 0, -0.2176, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]
    ).T

    # Listing the dimensions of the chassis
    r = 0.0475
    l = 0.47 / 2
    w = 0.3 / 2
    F = (r / 4) * np.array(
        [
            [-1 / (l + w), 1 / (l + w), 1 / (l + w), -1 / (l + w)],
            [1, 1, 1, 1],
            [-1, 1, -1, 1],
        ]
    )
    F6 = np.concatenate((np.zeros((2, 4)), F, np.zeros((1, 4))), axis=0)

    # The fixed offset from the chassis frame {b} to the base frame of the arm {0}
    Tb0 = np.array([[1, 0, 0, 0.1662], [0, 1, 0, 0], [0, 0, 1, 0.0026], [0, 0, 0, 1]])

    # The end-effector frame {e} at the zero configuration of the arm
    M0e = np.array([[1, 0, 0, 0.033], [0, 1, 0, 0], [0, 0, 1, 0.6546], [0, 0, 0, 1]])

    # Finding transformation matrix of the end effector in the chassis frame at zero configuration
    T0e = mr.FKinBody(M0e, Blist, thetalist)
    Tbe = np.matmul(Tb0, T0e)
    Teb = mr.TransInv(Tbe)

    # Finding the mobile manipulator jacobian Je
    J_base = np.matmul(mr.Adjoint(Teb), F6)
    J_arm = mr.JacobianBody(Blist, thetalist)
    Je = np.concatenate((J_base, J_arm), axis=1)

    psInv = np.linalg.pinv(Je, 1e-3)  # For calling into the wrapper script
    # psInv = np.linalg.pinv(Je) # For testing the code

    # Calculating end effector configuration error Xerr and Xerr_integral
    Xerr_bracket = mr.MatrixLog6(np.matmul(mr.TransInv(X), Xd))
    Xerr = mr.se3ToVec(Xerr_bracket)
    Xerr_integral = Xerr_integral + timestep * Xerr

    # Finding the desired body twist Vd
    Vd_bracket = (1 / timestep) * mr.MatrixLog6(np.matmul(np.linalg.inv(Xd), Xdnext))
    Vd = mr.se3ToVec(Vd_bracket)

    # Evaluating the commanded end-effector twist V expressed in the end-effector frame {e}
    Adj = mr.Adjoint(np.matmul(np.linalg.inv(X), Xd))
    Feedforward = np.matmul(Adj, Vd)
    V = Feedforward + np.matmul(Kp, Xerr) + np.matmul(Ki, Xerr_integral)

    # Hence the wheel and joint velocities are:
    u_theta_dot = np.matmul(psInv, V)
    return V, u_theta_dot, Xerr, Xerr_integral
    # return V_b, cmd , X_err, X_err_intergral


def state2config(curr_state):
    odemetry = curr_state[0:3]
    arm_joint_angles = curr_state[3:8]
    M_0e = np.array([[1, 0, 0, 0.033], [0, 1, 0, 0], [0, 0, 1, 0.6546], [0, 0, 0, 1]])

    # Jacobian matrices
    Blist = np.array(
        [
            [0, 0, 1, 0, 0.033, 0],
            [0, -1, 0, -0.5076, 0, 0],
            [0, -1, 0, -0.3526, 0, 0],
            [0, -1, 0, -0.2176, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]
    ).T

    theta_list = arm_joint_angles  # Extract joint angles
    J_arm = mr.JacobianBody(Blist, theta_list)
    r = 0.0475
    l = 0.235
    w = 0.15
    T0_e = mr.FKinBody(M_0e, Blist, theta_list)

    Tb_0 = np.array([[1, 0, 0, 0.1662], [0, 1, 0, 0], [0, 0, 1, 0.0026], [0, 0, 0, 1]])
    x = odemetry[1]
    y = odemetry[2]
    phi = odemetry[0]
    T_sb = np.array(
        [
            [np.cos(phi), -np.sin(phi), 0, x],
            [np.sin(phi), np.cos(phi), 0, y],
            [0, 0, 1, 0.0963],
            [0, 0, 0, 1],
        ]
    )
    T_se = np.dot(T_sb, Tb_0) @ T0_e

    return T_se


def base_state2config(curr_state):
    odemetry = curr_state[0:3]
    arm_joint_angles = curr_state[3:8]
    x = odemetry[1]
    y = odemetry[2]
    phi = odemetry[0]
    T_sb = np.array(
        [
            [np.cos(phi), -np.sin(phi), 0, x],
            [np.sin(phi), np.cos(phi), 0, y],
            [0, 0, 1, 0.0963],
            [0, 0, 0, 1],
        ]
    )

    return T_sb


def controller_0(X_d, X_d_next, curr_state, delta_t, K_p, K_i, X_err_prev_intergral):
    """
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
    """

    # Robot configuration
    r = 0.0475
    l = 0.235
    w = 0.15

    X = state2config(curr_state)

    # Transform matrices
    T_b0 = np.array([[1, 0, 0, 0.1662], [0, 1, 0, 0], [0, 0, 1, 0.0026], [0, 0, 0, 1]])

    M_0e = np.array([[1, 0, 0, 0.033], [0, 1, 0, 0], [0, 0, 1, 0.6546], [0, 0, 0, 1]])

    # Compute feedforward velocity and error twist
    V_d = 1.0 / delta_t * mr.MatrixLog6(np.linalg.inv(X_d) @ X_d_next)
    V_d = mr.se3ToVec(V_d)
    Ad_V_d = mr.Adjoint(np.linalg.inv(X) @ X_d) @ V_d

    X_err = mr.MatrixLog6(np.linalg.inv(X) @ X_d)
    X_err = mr.se3ToVec(X_err)

    X_err_intergral = X_err_prev_intergral + X_err * delta_t
    # control
    V_b = Ad_V_d + K_p @ X_err + K_i @ X_err_intergral

    # Jacobian matrices
    Blist = np.array(
        [
            [0, 0, 1, 0, 0.033, 0],
            [0, -1, 0, -0.5076, 0, 0],
            [0, -1, 0, -0.3526, 0, 0],
            [0, -1, 0, -0.2176, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]
    ).T

    theta_list = curr_state[3:8]  # Extract joint angles
    J_arm = mr.JacobianBody(Blist, theta_list)

    F = (
        r
        / 4
        * np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1 / (l + w), 1 / (l + w), 1 / (l + w), -1 / (l + w)],
                [1, 1, 1, 1],
                [-1, 1, -1, 1],
                [0, 0, 0, 0],
            ]
        )
    )
    T_0e = mr.FKinBody(M_0e, Blist, theta_list)
    J_base = mr.Adjoint(np.linalg.inv(T_0e) @ np.linalg.inv(T_b0)) @ F

    Je = np.hstack((J_base, J_arm))

    # Compute the pseudoinverse with a tolerance to mitigate singularity issues
    psInv = np.linalg.pinv(Je, 1e-3)
    # Now compute the pseudoinverse with correct dimension handling
    Je_inv = psInv

    # Compute the commanded velocities
    cmd = Je_inv @ V_b
    print("control input", cmd)
    print("V_b_", V_b)
    print("Je", Je)

    return V_b, cmd, X_err, X_err_intergral


def controller_1(X_d, X_d_next, curr_state, delta_t, K_p, K_i, X_err_prev_intergral):
    """
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
    """
    # Robot configuration
    r = 0.0475
    l = 0.235
    w = 0.15

    X = state2config(curr_state)
    T_sb = base_state2config(curr_state)
    # Transformation matrices
    T_b0 = np.array([[1, 0, 0, 0.1662], [0, 1, 0, 0], [0, 0, 1, 0.0026], [0, 0, 0, 1]])

    M_0e = np.array([[1, 0, 0, 0.033], [0, 1, 0, 0], [0, 0, 1, 0.6546], [0, 0, 0, 1]])

    # Compute the feedforward velocity and the error twist
    V_d = 1.0 / delta_t * mr.MatrixLog6(np.linalg.inv(X_d) @ X_d_next)
    V_d = mr.se3ToVec(V_d)
    Ad_V_d = mr.Adjoint(np.linalg.inv(X) @ X_d) @ V_d

    X_err = mr.MatrixLog6(np.linalg.inv(X) @ X_d)
    X_err = mr.se3ToVec(X_err)

    X_err_intergral = X_err_prev_intergral + X_err * delta_t
    # feedback control
    V_b = K_p @ X_err + K_i @ X_err_intergral

    # Jacobian for the arm and base
    Blist = np.array(
        [
            [0, 0, 1, 0, 0.033, 0],
            [0, -1, 0, -0.5076, 0, 0],
            [0, -1, 0, -0.3526, 0, 0],
            [0, -1, 0, -0.2176, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]
    ).T
    theta_list = curr_state[3:8]  # Extract the joint angles
    J_arm = mr.JacobianBody(Blist, theta_list)

    F = (
        r
        / 4
        * np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1 / (l + w), 1 / (l + w), 1 / (l + w), -1 / (l + w)],
                [1, 1, 1, 1],
                [-1, 1, -1, 1],
                [0, 0, 0, 0],
            ]
        )
    )
    T_0e = mr.FKinBody(M_0e, Blist, theta_list)
    J_base = mr.Adjoint(np.linalg.inv(T_0e) @ np.linalg.inv(T_b0)) @ F

    Je = np.hstack((J_base, J_arm))

    # Selectively Weighted Pseudoinverse Calculation

    # Define weighting matrices
    W_navigate = np.diag([5] * 4 + [1,1,1,0.1,0.1])  # Higher weight for base movement
    W_pick = np.diag([1] * 4 + [1] * 5)  # Higher weight for arm movement
    W_fine_tune = np.diag([0.1] * 4 + [3] * 5)  # Higher weight for arm movement

    # Adjust weighting based on your task preference or dynamic conditions
    # check the mode
    if judge_mode(X,T_sb,Tsc_initial_) == "navigate":
        W = W_navigate
    elif judge_mode(X,T_sb, Tsc_initial_) == "pick":
        W = W_pick
    elif judge_mode(X,T_sb, Tsc_initial_) == "fine_tune":
        W = W_fine_tune
    print("mode", judge_mode(X,T_sb, Tsc_initial_))
    Je_weighted = Je @ W

    Je_inv_weighted = np.linalg.pinv(Je_weighted, 1e-3)

    cmd = Je_inv_weighted @ V_b

    # print("control input", cmd)
    # print("V_b_", V_b)
    # print("Je_weighted", Je_weighted)
    # print("Je", Je)

    return V_b, cmd, X_err, X_err_intergral

def controller_2(X_d, X_d_next, curr_state, delta_t, K_p, K_i, X_err_prev_intergral,flag):
    """
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
    """
    # Robot configuration
    r = 0.0475
    l = 0.235
    w = 0.15

    X = state2config(curr_state)
    T_sb = base_state2config(curr_state)
    # Transformation matrices
    T_b0 = np.array([[1, 0, 0, 0.1662], [0, 1, 0, 0], [0, 0, 1, 0.0026], [0, 0, 0, 1]])

    M_0e = np.array([[1, 0, 0, 0.033], [0, 1, 0, 0], [0, 0, 1, 0.6546], [0, 0, 0, 1]])

    # Compute the feedforward velocity and the error twist
    V_d = 1.0 / delta_t * mr.MatrixLog6(np.linalg.inv(X_d) @ X_d_next)
    V_d = mr.se3ToVec(V_d)
    Ad_V_d = mr.Adjoint(np.linalg.inv(X) @ X_d) @ V_d

    X_err = mr.MatrixLog6(np.linalg.inv(X) @ X_d)
    X_err = mr.se3ToVec(X_err)

    X_err_intergral = X_err_prev_intergral + X_err * delta_t
    # feedback control
    V_b = K_p @ X_err + K_i @ X_err_intergral

    # Define weighting matrices
    W_navigate = np.diag([3] * 4 + [1,1,1,0.1,0.1])  # Higher weight for base movement
    W_pick = np.diag([0.8] * 4 + [2] * 5)  # Higher weight for arm movement
    W_fine_tune = np.diag([0.1] * 4 + [1,0.8,3,2,1])  # Higher weight for arm movement


    # Adjust weighting based on your task preference or dynamic conditions
    # check the mode
    # if flag == 0:
    X_base_err = mr.MatrixLog6(np.linalg.inv(T_sb) @ Tsc_initial_)
    X_base_err = mr.se3ToVec(X_base_err)
    mode = judge_mode(X,T_sb,Tsc_initial_)
    # else:
    #     X_base_err = mr.MatrixLog6(np.linalg.inv(T_sb) @ Tsc_final_)
    #     X_base_err = mr.se3ToVec(X_base_err)
    #     mode = judge_mode(X,T_sb,Tsc_final_)

    print(flag)

    if mode == "navigate":
        W = W_navigate
    elif mode == "pick":
        W = W_pick
    elif mode == "fine_tune":
        W = W_fine_tune
        flag = 1

    #base control
    K_p_base = 7
    V_base = K_p @ X_base_err

    # Jacobian for the arm and base
    Blist = np.array(
        [
            [0, 0, 1, 0, 0.033, 0],
            [0, -1, 0, -0.5076, 0, 0],
            [0, -1, 0, -0.3526, 0, 0],
            [0, -1, 0, -0.2176, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]
    ).T
    theta_list = curr_state[3:8]  # Extract the joint angles
    J_arm = mr.JacobianBody(Blist, theta_list)

    F = (
        r
        / 4
        * np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-1 / (l + w), 1 / (l + w), 1 / (l + w), -1 / (l + w)],
                [1, 1, 1, 1],
                [-1, 1, -1, 1],
                [0, 0, 0, 0],
            ]
        )
    )
    T_0e = mr.FKinBody(M_0e, Blist, theta_list)
    J_base = mr.Adjoint(np.linalg.inv(T_0e) @ np.linalg.inv(T_b0)) @ F

    J_base_base = F

    Je = np.hstack((J_base, J_arm))


    # Selectively Weighted Pseudoinverse Calculation



    print("mode", judge_mode(X,T_sb, Tsc_initial_))
    Je_weighted = Je @ W

    Je_inv_weighted = np.linalg.pinv(Je_weighted, 1e-3)

    cmd = Je_inv_weighted @ V_b

    # print("control input", cmd)
    # print("V_b_", V_b)
    # print("Je_weighted", Je_weighted)
    # print("Je", Je)

    if judge_mode(X,T_sb,Tsc_initial_) == "navigate":
        cmd[0:4] = np.linalg.pinv(J_base, 1e-3)@V_base
        cmd[4:9] = np.zeros(5)

    return V_b, cmd, X_err, X_err_intergral, flag
def judge_mode(T_current_ee,T_chassis, T_sc):
    """
    Judge the mode based on the current end-effector pose and the desired end-effector pose for grasping

    Input:
    T_current_ee: current end-effector pose
    T_sc: desired end-effector pose for grasping
    Output:
    mode: 'grasp' if the current pose is close to the desired pose, 'move' otherwise
    """
    # Define a threshold for determining if the poses are close
    threshold_2d = 0.1
    threshold_3d = 0.1  
    # caculate the difference bwteen x y z position
    diff_3d = np.linalg.norm(T_current_ee[0:3, 3] - T_sc[0:3, 3])
    diff_2d = np.linalg.norm(T_current_ee[0:2, 3] - T_sc[0:2, 3])

    diff_chassis = np.linalg.norm(T_chassis[0:2, 3] - T_sc[0:2, 3])
    print("diff_chassis", diff_chassis)

    if diff_chassis > 0.64:
        mode = "navigate"
        return mode
    elif diff_chassis > 0.62:
        mode = "pick"
        if diff_3d < threshold_3d:
            mode = "fine_tune"
        return mode
    # else:
    #     mode = "fine_tune"
    #     return mode
    # Check if the difference is below the threshold
    if diff_2d > threshold_2d:
        mode = "navigate"
    elif diff_3d > threshold_3d:
        mode = "pick"
    elif diff_3d <= threshold_3d:
        mode = "fine_tune"

    return mode


def main():
    X_d = np.array([[0, 0, 1, 0.5], [0, 1, 0, 0], [-1, 0, 0, 0.5], [0, 0, 0, 1]])
    X_d_next = np.array([[0, 0, 1, 0.6], [0, 1, 0, 0], [-1, 0, 0, 0.3], [0, 0, 0, 1]])
    X = np.array(
        [
            [0.170, 0, 0.985, 0.387],
            [0, 1, 0, 0],
            [-0.985, 0, 0.170, 0.570],
            [0, 0, 0, 1],
        ]
    )  # calculated from config

    curren_state = np.array([0, 0, 0, 0, 0, 0.2, -1.6, 0, 0, 0, 0, 0, 0])
    # X_est = state2config(curren_state)
    # print(X_est)

    delta_t = 0.01
    K_p = np.zeros((6, 6))
    K_i = np.zeros((6, 6))
    V_b, cmd = controller(X_d, X_d_next, curren_state, delta_t, K_p, K_i)
    # print(V_b, cmd)


if __name__ == "__main__":
    main()
