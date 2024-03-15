from module.trajectory_generatator import TrajectoryGenerator
from module.motion_model import next_state
from module.feedback_control import controller
from robot import Robot
import numpy as np
import matplotlib.pyplot as plt
import csv
from module.trajectory_generatator import *
def main():
    p = 11
    i = 0.002
    K_p = p*np.eye(6)
    K_i = i*np.eye(6)
    robot = Robot(K_p, K_i)

    traj = robot.TrajectoryGenerator()
    initial_config = np.array([0.5, -0.2, 0.1, 0, 0, 0.2, -1.6, 0, 0, 0, 0, 0, 0])
    current_state = initial_config
    state_hitsory = []
    state_hitsory.append(current_state)
    X_error_history = []
    control_frequency = 1
    N = len(traj)-1
    N = 1300
    for i in range(N):

        X_d = rearrange_back(traj[i])
        X_d1 = rearrange_back(traj[i+1])
        # if i == 0:
        #     X_d = rearrange_back(traj[i])
        #     X_d1 = rearrange_back(traj[200])

        # else:
        #     X_d = rearrange_back(traj[200])
        #     X_d1 = rearrange_back(traj[200])

        for j in range(control_frequency):
            V_b, cmd, X_err = robot.controller_2(X_d,X_d1, current_state= current_state)
            cmd = cmd.tolist()+[traj[i][-1]]
            current_state = robot.next_state(current_state, cmd, 0.01, 10, 10)

        X_error_history.append(X_err)
        state_hitsory.append(current_state)
    #plot the error
    fig = plt.figure()
    plt.plot(X_error_history)
    plt.show()
    #save the plot with folder name error.png
    folder = '../result/'
    fig.savefig(folder + f'error_{p}_{i}.png')

    # Write the reference configurations to a .csv file
    with open("1.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for state in state_hitsory:
            writer.writerow(state)
    with open(folder+f"{p}_{i}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for state in state_hitsory:
            writer.writerow(state)

if __name__ == '__main__':
    main()