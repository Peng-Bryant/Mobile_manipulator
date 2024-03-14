from module.trajectory_generatator import TrajectoryGenerator
from module.motion_model import next_state
from module.feedback_control import controller
from robot import Robot
import numpy as np
import matplotlib.pyplot as plt
import csv
def main():
    K_p = 0.1*np.eye(6)
    K_i = 10*np.zeros(6)
    robot = Robot(K_p, K_i)

    traj = robot.TrajectoryGenerator()
    current_state = [0]*13
    state_hitsory = []
    state_hitsory.append(current_state)
    X_error_history = []
    for i in range(len(traj)-1):
        X_d = traj[i]
        V_b, cmd, X_err = robot.controller(X_d=traj[i], X_d_next=traj[i+1], current_state= current_state)

        cmd = cmd[0].tolist()+[traj[i][-1]]
        current_state = robot.next_state(current_state, cmd, 0.01, 1)
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
