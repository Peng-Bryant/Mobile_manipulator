"""
For this component, you will write a function called NextState that uses the kinematics of the youBot 
(see MR Exercise 13.33), your knowledge of velocity kinematics, and your knowledge of the Euler method
 to predict how the robot will move in a small timestep given its current configuration and velocity. 
 Thus, your function NextState should take as inputs: 
 Inputs 
• The current state of the robot (13 variables: 3 for chassis, 5 for arm, 4 for wheel angles, one for gripper state) 
• The joint and wheel velocities (10 variables: 5 for arm  ̇ θ, 4 for wheels u, 1 for gripper state  ̇ η) 
• The timestep size ∆t (1 parameter) 
• The maximum joint and wheel velocity magnitude (1 parameter)
Outputs NextState should also produce the following outputs that describe the configuration of the robot 
one timestep (∆t) later: 
• The next state (configuration) of the robot (13 variables)

Approach The function NextState is based on a simple first-order Euler step: 
• new arm joint angles = (old arm joint angles) + (joint speeds)∆t 
• new wheel angles = (old wheel angles) + (wheel speeds)∆t 
• new chassis configuration is obtained from odometry, as described in Chapter 13.4
"""
import csv
from scipy.linalg import expm

import numpy as np
import modern_robotics as mr
def NextState(Current_state,jw_vels,timestep,max_jw_vels,asd):
    '''
    This function uses the kinematics of the youBot and Euler method to predict how the
    robot will move in a small timestep given its current configuration and velocity.

    Inputs:
    • The current state of the robot (12 variables: 3 for chassis, 5 for arm, 4 for wheel angles)
    • The joint and wheel velocities (9 variables: 5 for arm ˙θ, 4 for wheels u)
    • The timestep size ∆t (1 parameter)
    • The maximum joint and wheel velocity magnitude (1 parameter)

    Output:
    It produces the following outputs that describe the configuration of the robot one
    timestep (∆t) later:
    • The next state (configuration) of the robot (12 variables)
    array = [chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5, W1, W2, W3, W4]
    '''
    # Listing the dimensions of the chassis
    r = 0.0475
    l = 0.47 / 2
    w = 0.3 / 2
    z = 0.0963

    # Sorting input data
    cur_chassis_angle = Current_state[0]
    cur_chassis_pos = Current_state[1:3]
    cur_joint_angles = Current_state[3:8]
    cur_wheel_angles = Current_state[8:12]

    # Limiting joint velocities
    for i, jw_vel in enumerate(jw_vels):
        if abs(jw_vel) > max_jw_vels:
            jw_vels[i]= jw_vel/abs(jw_vel)*max_jw_vels # Preserving the signs
    vel_arm = np.array(jw_vels[4:-1])    
    vel_wheels = np.array(jw_vels[:4])  
    
    # print(vel_arm)
    # Finding new joints and wheel angles
    new_joint_angles = cur_joint_angles + vel_arm * timestep
    new_wheel_angles = cur_wheel_angles + vel_wheels * timestep
    # Hence we got J1, J2, J3, J4, J5, W1, W2, W3, W4 of the next state

    # Finding new robot pose for finding (chassis angle, chassis x, chassis y)

    # Getting current transformation matrix of the body frame
    cur_pose = np.array([ [np.cos(cur_chassis_angle), - np.sin(cur_chassis_angle),0,cur_chassis_pos[0]],
                            [np.sin(cur_chassis_angle), np.cos(cur_chassis_angle),0,cur_chassis_pos[1]],
                            [0,0,1,z],
                            [0,0,0,1] ]) 
    
    # Finding change in wheel angles
    delta_thetas = new_wheel_angles - cur_wheel_angles
 
    # Finding Chassis body twist
    F = (r / 4 ) * np.array([ [-1 / (l+w), 1 / (l+w), 1 / (l+w), -1 / (l+w)],
                                [1,1,1,1],
                                [-1,1,-1,1]])

    V_b = np.matmul(F,delta_thetas)
    # Hence we got yaw and x,y components of the body twist
    # Converting this to vector of R^6
    V_b6 = np.concatenate((np.array([0,0]), V_b, np.array([0])))

    # Converting the body twist vector to se3
    rel_pose = expm(mr.VecTose3(V_b6.T))

    # Finding new pose
    new_pose = np.matmul(cur_pose,rel_pose)

    # Determining new chassis angle and position 
    new_chassis_angle = np.arccos(new_pose[0,0])
    new_chassis_pos = new_pose[:2,3]
    # Hence we got (chassis angle, chassis x, chassis y)
    
    
    # Finding the new state by combining all the results  
    New_state = np.hstack((new_chassis_angle,new_chassis_pos, new_joint_angles, new_wheel_angles,np.array([jw_vels[-1]])))

    return New_state

def next_state(
    current_state, joint_and_wheel_velocities, delta_t, max_arm,max_wheel
):
    # exatract the current state
    odemetry_curr = current_state[0:3]
    arm_joint_angles_curr = current_state[3:8]
    wheel_angles_curr = current_state[8:12]
    # extract the joint and wheel velocities
    arm_joint_velocities = joint_and_wheel_velocities[4:9]
    wheel_velocities = joint_and_wheel_velocities[0:4]

    gripper_state = joint_and_wheel_velocities[9]
    #need to check the maximum joint and wheel velocity
    #if any of the joint or wheel velocity is greater than the maximum joint and wheel velocity, then clip this entry to be maximum joint and wheel velocity
    arm_joint_velocities = np.clip(arm_joint_velocities, -max_arm, max_arm)
    wheel_velocities = np.clip(wheel_velocities, -max_wheel, max_wheel)

    # specify the configuration of the chassis
    l = 0.47 / 2
    w = 0.3 / 2
    r = 0.0475

    # define the pseudoinverse of H for four-wheel mecanum drive
    F = (
        np.array(
            [
                [-1 / (l + w), 1 / (l + w), 1 / (l + w), -1 / (l + w)],
                [1, 1, 1, 1],
                [-1, 1, -1, 1],
            ]
        )
        * r
        / 4
    )
    V_chassis = np.dot(F, wheel_velocities) * delta_t
    # calculate the new odemetry
    wbz = V_chassis[0]
    vbx = V_chassis[1]
    vby = V_chassis[2]

    if wbz == 0:
        odemetry_new = odemetry_curr + np.array([0, vbx, vby])
    else:
        odemetry_new = odemetry_curr + np.array(
            [
                wbz,
                (vbx * np.sin(wbz) + vby * (np.cos(wbz) - 1)) / wbz,
                (vby * np.sin(wbz) + vbx * (1 - np.cos(wbz))) / wbz,
            ]
        )

    # calculate the new state

    # calculate the new arm joint angles
    arm_joint_angles_new = arm_joint_angles_curr + arm_joint_velocities * delta_t
    # calculate the new wheel angles
    wheel_angles_new = wheel_angles_curr + wheel_velocities * delta_t
    # calculate the new gripper state
    gripper_state_new = [gripper_state]

    # concatenate the new state
    new_state = np.concatenate((odemetry_new, arm_joint_angles_new, wheel_angles_new , gripper_state_new))

    return new_state

def main():
    #test the next_state function
    delta_t = 0.01 
    N = 1000
    max_joint_and_wheel_velocity = 5
    initial_state = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])

    current_state = initial_state
    control_input_0 = np.array([1,0,0,0,0,0,0,0,0,0])
    control_input_1 = np.array([0,1,0,0,0,0,0,0,0,0])
    control_input_2 = np.array([0,0,1,0,0,0,0,0,0,0])
    control_input_3 = np.array([0,0,0,1,0,0,0,0,0,0])
    control_input_4 = np.array([0,0,0,0,1,0,0,0,0,0])
    control_input_5 = np.array([0,0,0,0,0,1,0,0,0,0])
    control_input_6 = np.array([0,0,0,0,0,0,1,0,0,0])
    control_input_7 = np.array([0,0,0,0,0,0,0,1,0,0])
    control_input_8 = np.array([0,0,0,0,0,0,0,0,1,0])
    #append the control input into a list
    control_input = []
    control_input.append(control_input_0)
    control_input.append(control_input_1)
    control_input.append(control_input_2)
    control_input.append(control_input_3)
    control_input.append(control_input_4)
    control_input.append(control_input_5)
    control_input.append(control_input_6)
    control_input.append(control_input_7)
    control_input.append(control_input_8)

    state_trajectory = []
    state_trajectory.append(current_state)

    for i in range(N):
        j = int(i/100) % 9
        new_state = next_state(current_state, control_input[j], delta_t, max_joint_and_wheel_velocity)
        state_trajectory.append(new_state)
        # print(new_state)
        current_state = new_state
    
    print(state_trajectory[0:10])
    #writing csv files in Python
    with open("../../data/state_trajectory_1.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for config in state_trajectory:
            writer.writerow(config)

if __name__ == "__main__":
    main()
