import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_trajectory(prior_seqs, policy_seqs):
    ax = plt.axes(projection='3d')

    # plot actions
    x = prior_seqs[:,0]
    y = prior_seqs[:,1]
    z = prior_seqs[:,2]
    ax.plot3D(x, y, z, 'gray')
    ax.scatter3D(x, y, z)

    # plot actions
    x = policy_seqs[:,0]
    y = policy_seqs[:,1]
    z = policy_seqs[:,2]
    ax.plot3D(x, y, z, 'blue')
    ax.scatter3D(x, y, z)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([0.1, 0.2])
    ax.set_ylim([0.2, 0.3])
    ax.set_zlim([0.0, 0.15])

    plt.show()
    plt.savefig('fig/'+'traj.png')

if __name__ == '__main__':    
    traj_df = pd.read_csv('~/Workspaces/rl_ws/spirl/trajectories2.csv', header=None)
    policy_traj_array = traj_df.iloc[:,0:3].values.astype(np.float32)#.reshape(-1,120)
    prior_traj_array = traj_df.iloc[:,3:6].values.astype(np.float32)#.reshape(-1,120)
    plot_trajectory(prior_traj_array, policy_traj_array)