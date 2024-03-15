from module.trajectory_generatator import TrajectoryGenerator
from module.motion_model import next_state
from module.feedback_control import controller
from robot import Robot
import numpy as np
import matplotlib.pyplot as plt
import csv
from module.trajectory_generatator import *
def main():
    K_p = 2.5*np.eye(6)
    K_i = 0.01*np.eye(6)
    robot = Robot(K_p, K_i)

    traj = robot.TrajectoryGenerator()
    initial_config = np.array([0.5, -0.2, 0.1, 0, 0, 0.2, -1.6, 0, 0, 0, 0, 0, 0])
    current_state = initial_config
    state_hitsory = []
    state_hitsory.append(current_state)
    X_error_history = []

    for i in range(len(traj)-1):

        X_d = rearrange_back(traj[i])
        X_d1 = rearrange_back(traj[i+1])

        V_b, cmd, X_err = robot.controller(X_d,X_d1, current_state= current_state)

        cmd = cmd.tolist()+[traj[i][-1]]
        current_state = robot.next_state(current_state, cmd, 0.01, 10, 10)
        X_error_history.append(X_err)
        state_hitsory.append(current_state)
    plt.plot(X_error_history)
    plt.show()

    # Write the reference configurations to a .csv file
    with open("test1.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for state in state_hitsory:
            writer.writerow(state)


if __name__ == '__main__':
    main()