import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


x = np.linspace(0,2*np.pi,100)
y = np.sin(x) + np.random.random(100) * 0.8

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

#plot(x, y,'o')
#plot(x, smooth(y,3), 'r-', lw=2)
#plot(x, smooth(y,19), 'g-', lw=2)

def plot_mean_and_CI(axis, mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(axis, ub, lb, color=color_shading, alpha=0.1)
    
    # plot the mean on top
    plt.plot(axis, mean, color_mean, alpha=0.8)

def plot_smooth_reward1(reward_list):
    sns.set(style="darkgrid", font_scale=1.3)
    x = np.linspace(0, 99, 100)
    colors = ['purple', 'green', 'orange']
    for i_list in range(len(reward_list)):
        if True:
            for i in range(3):
                reward_list[i_list][i] = smooth(reward_list[i_list][i], 5)
                
        # rewards mean and std                  
        mean = reward_list[i_list].mean(axis=0)[0:100]
        std = reward_list[i_list].std(axis=0)[0:100]
        plot_mean_and_CI(x, mean, mean-std, mean+std, colors[i_list], colors[i_list])

    plt.legend(labels=["offset 1cm","offset 3cm","offset 5cm"],loc='lower right')
    plt.xlabel('training episodes')
    plt.ylabel('reward')
    plt.savefig('reward1.png')

def plot_smooth_reward2(reward_list):
    sns.set(style="darkgrid", font_scale=1.3)
    x = np.linspace(0, 99, 100)
    colors = ['purple', 'blue', 'green', 'orange']
    for i_list in range(len(reward_list)):
        if True:
            for i in range(3):
                reward_list[i_list][i] = smooth(reward_list[i_list][i], 6)
                
        # rewards mean and std                  
        mean = reward_list[i_list].mean(axis=0)[0:100]
        std = reward_list[i_list].std(axis=0)[0:100]
        plot_mean_and_CI(x, mean, mean-std, mean+std, colors[i_list], colors[i_list])

    # plt.yticks([0, 2, 4, 6, 8, 10])
    plt.legend(labels=["square 5°","square 10°","triangle 5°", "triangle 10°"],loc='lower right')
    plt.xlabel('Environment episodes')
    plt.ylabel('reward')
    plt.title('Adapt circular peg-in-hole policy')
    plt.show()
    plt.savefig('reward2.png')


def plot_smooth_reward2_new(reward_list):
    sns.set(style="darkgrid", font_scale=1.3)
    x = np.linspace(0, 99, 100)
    colors = ['purple', 'green', 'orange']
    for i_list in range(len(reward_list)):
        if True:
            for i in range(3):
                reward_list[i_list][i] = smooth(reward_list[i_list][i], 6)
                
        # rewards mean and std                  
        mean = reward_list[i_list].mean(axis=0)[0:100]
        std = reward_list[i_list].std(axis=0)[0:100]
        plot_mean_and_CI(x, mean, mean-std, mean+std, colors[i_list], colors[i_list])

    plt.yticks([0, 2, 4, 6, 8, 10])
    plt.legend(labels=["circular", "triangular 5°", "triangular 10°"],loc='lower right')
    plt.xlabel('Environment episodes')
    plt.ylabel('reward')
    plt.title('Adapt square peg-in-hole policy')
    plt.show()
    plt.savefig('reward2.png')
    
def plot_smooth_reward3(reward_list):
    sns.set(style="darkgrid", font_scale=1.3)
    x = np.linspace(0, 99, 100)
    colors = ['purple', 'green', 'orange']
    for i_list in range(len(reward_list)):
        if True:
            for i in range(3):
                reward_list[i_list][i] = smooth(reward_list[i_list][i], 8)
                
        # rewards mean and std              
        mean = reward_list[i_list].mean(axis=0)[0:100]
        std = reward_list[i_list].std(axis=0)[0:100]
        plot_mean_and_CI(x, mean, mean-std, mean+std, colors[i_list], colors[i_list])

    plt.legend(labels=["circle","square","triangle"],loc='lower right')
    plt.xlabel('training episodes')
    plt.ylabel('reward')
    plt.show()
    plt.savefig('reward3.png')

def plot_action(seqs):
    for i_seq in range(len(seqs)):
        actions = seqs[i_seq]['actions']

        # plot actions
        ax = plt.axes(projection='3d')
        x = actions[:,0]
        y = actions[:,1]
        z = actions[:,2]
        ax.plot3D(x, y, z, 'gray')
        ax.scatter3D(x, y, z)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([0.1, 0.3])
        ax.set_ylim([0.1, 0.4])
        ax.set_zlim([0.1, 0.3])
        plt.show()
        plt.savefig('fig/'+str(i_seq)+'.png')


def plot_reward(rewards):
    sns.lineplot(data=rewards)
    plt.show()
    plt.savefig('val_rewards.png')


def plot_wrench(wrench):
    wrench_x = wrench[:,0]
    wrench_y = wrench[:,1]
    wrench_z = wrench[:,2]
    sns.set(style="darkgrid", font_scale=1.0)

    sns.lineplot(data=wrench_x)
    sns.lineplot(data=wrench_y)
    sns.lineplot(data=wrench_z)
    plt.xlabel('steps')
    plt.ylabel('wrench (N)')
    plt.legend(labels=["wrench_x","wrench_y", "wrench_z"])
    #plt.show()
    plt.savefig('wrench.png')
    
def plot_variable_stiffness(stiffness):
    sns.set(style="darkgrid", font_scale=1.0)
    f, (ax1, ax2, ax3, ax4)=plt.subplots(4,sharex=True, sharey=True)
    ax1.plot(stiffness[:,0],'purple',label='x')
    ax1.legend(loc='lower right')
    ax1.set_ylabel('N/m')
    #plt.subplot(4,1,2)
    ax2.plot(stiffness[:,1],'green',label='y')
    ax2.legend(loc='lower right')
    ax2.set_ylabel('N/m')
    #plt.subplot(4,1,3)
    ax3.plot(stiffness[:,2],'orange',label='z')
    ax3.legend(loc='lower right')
    ax3.set_ylabel('N/m')
    #plt.subplot(4,1,4)
    ax4.plot(stiffness[:,3],'red',label='theta_z')
    ax4.legend(loc='lower right')   
    ax4.set_ylabel('Nm/rad')
    ax4.set_xlabel('Steps')
    #plt.xlabel('steps')
    #plt.ylabel('Variable stiffness (N/m or Nm/rad)')
    #plt.legend(labels=["x","y", "z", "theta_z"],loc='lower right')
    #plt.legend(loc='lower right')
    #plt.show()
    plt.savefig('stiffness.png')

def plot_variable_position(position):
    sns.set(style="darkgrid", font_scale=1.0)
    f, (ax1, ax2, ax3)=plt.subplots(3,sharex=True, sharey=True)
    ax1.plot(position[:,0],'purple',label='x')
    ax1.legend(loc='lower right')
    ax1.set_ylabel('N/m')
    # ax1.set_ylim(0.34, 0.36)
    #plt.subplot(4,1,2)
    ax2.plot(position[:,1],'green',label='y')
    ax2.legend(loc='lower right')
    ax2.set_ylabel('N/m')
    # ax2.set_ylim(0.28, 0.34)
    #plt.subplot(4,1,3)
    ax3.plot(position[:,2],'orange',label='z')
    ax3.legend(loc='lower right')
    ax3.set_ylabel('N/m')
    ax3.set_ylim(0.028, 0.06)
    #plt.subplot(4,1,4)
    # ax4.plot(stiffness[:,3],'red',label='theta_z')
    # ax4.legend(loc='lower right')   
    # ax4.set_ylabel('Nm/rad')
    # ax4.set_xlabel('Steps')
    #plt.xlabel('steps')
    #plt.ylabel('Variable stiffness (N/m or Nm/rad)')
    #plt.legend(labels=["x","y", "z", "theta_z"],loc='lower right')
    #plt.legend(loc='lower right')
    plt.show()
    # plt.savefig('stiffness.png')



if __name__ == '__main__':
    # wrench = reward_df.iloc[:,18:21].values.astype(np.float32)
    # plot_wrench(wrench)
    # reward_df1 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/circle/same_position/train_episode_reward1.csv', header=None)
    # stiffness_df = pd.read_csv('~/Workspaces/rl_ws/spirl/variable_stiffness1.csv', header=None)
    # stiffness_array = stiffness_df.iloc[:,0:4].values.astype(np.float32)#.reshape(-1,120)
    # plot_variable_stiffness(stiffness_array)

    # circle same position
    if False:#reward1
        # circle same position
        reward_df1 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/circle/same_position/train_episode_reward1.csv', header=None)
        reward_df2 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/circle/same_position/train_episode_reward2.csv', header=None)
        reward_df3 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/circle/same_position/train_episode_reward3.csv', header=None)    
        reward_array1 = reward_df1.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array2 = reward_df2.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array3 = reward_df3.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
    
        # circle close position
        reward_df1_close = pd.read_csv('~/Workspaces/rl_ws/spirl/results/circle/close_position/train_episode_reward1.csv', header=None)
        reward_df2_close = pd.read_csv('~/Workspaces/rl_ws/spirl/results/circle/close_position/train_episode_reward2.csv', header=None)
        reward_df3_close = pd.read_csv('~/Workspaces/rl_ws/spirl/results/circle/close_position/train_episode_reward3.csv', header=None)    
        reward_array1_close = reward_df1_close.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array2_close = reward_df2_close.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array3_close = reward_df3_close.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        
        # circle far position
        reward_df1_far = pd.read_csv('~/Workspaces/rl_ws/spirl/results/circle/far_position/train_episode_reward1.csv', header=None)
        reward_df2_far = pd.read_csv('~/Workspaces/rl_ws/spirl/results/circle/far_position/train_episode_reward2.csv', header=None)
        reward_df3_far = pd.read_csv('~/Workspaces/rl_ws/spirl/results/circle/far_position/train_episode_reward3.csv', header=None)    
        reward_array1_far = reward_df1_far.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array2_far = reward_df2_far.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array3_far = reward_df3_far.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        
        
        reward_list = [np.concatenate((reward_array1_far, reward_array2_far, reward_array3_far)),
                       np.concatenate((reward_array1, reward_array2, reward_array3)),
                       np.concatenate((reward_array1_close, reward_array2_close, reward_array3_close))
                       ]
    
        plot_smooth_reward1(reward_list)
        
    elif True:#reward3
        if True:
            # square small
            reward_df1 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/square/small_angle/train_episode_reward1.csv', header=None)
            reward_df2 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/square/small_angle/train_episode_reward2.csv', header=None)
            reward_df3 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/square/small_angle/train_episode_reward3.csv', header=None)    
            reward_array1 = reward_df1.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
            reward_array2 = reward_df2.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
            reward_array3 = reward_df3.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        
            # square large
            reward_df1_close = pd.read_csv('~/Workspaces/rl_ws/spirl/results/square/large_angle/train_episode_reward1.csv', header=None)
            reward_df2_close = pd.read_csv('~/Workspaces/rl_ws/spirl/results/square/large_angle/train_episode_reward2.csv', header=None)
            reward_df3_close = pd.read_csv('~/Workspaces/rl_ws/spirl/results/square/large_angle/train_episode_reward3.csv', header=None)    
            reward_array1_close = reward_df1_close.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
            reward_array2_close = reward_df2_close.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
            reward_array3_close = reward_df3_close.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
            
            # triangle small
            reward_df1_far = pd.read_csv('~/Workspaces/rl_ws/spirl/results/triangle/small_angle/train_episode_reward1.csv', header=None)
            reward_df2_far = pd.read_csv('~/Workspaces/rl_ws/spirl/results/triangle/small_angle/train_episode_reward2.csv', header=None)
            reward_df3_far = pd.read_csv('~/Workspaces/rl_ws/spirl/results/triangle/small_angle/train_episode_reward3.csv', header=None)
            reward_array1_far = reward_df1_far.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
            reward_array2_far = reward_df2_far.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
            reward_array3_far = reward_df3_far.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
            
            # triangle large
            reward_df1_tri_large = pd.read_csv('~/Workspaces/rl_ws/spirl/results/triangle/large_angle/train_episode_reward1.csv', header=None)
            reward_df2_tri_large = pd.read_csv('~/Workspaces/rl_ws/spirl/results/triangle/large_angle/train_episode_reward2.csv', header=None)
            reward_df3_tri_large = pd.read_csv('~/Workspaces/rl_ws/spirl/results/triangle/large_angle/train_episode_reward3.csv', header=None)    
            reward_array1_tri_large = reward_df1_tri_large.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
            reward_array2_tri_large = reward_df2_tri_large.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
            reward_array3_tri_large = reward_df3_tri_large.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
            
            reward_list = [np.concatenate((reward_array1_far, reward_array2_far, reward_array3_far)),
                        np.concatenate((reward_array1, reward_array2, reward_array3)),
                        np.concatenate((reward_array1_close, reward_array2_close, reward_array3_close)),
                        np.concatenate((reward_array1_tri_large, reward_array2_tri_large, reward_array3_tri_large))
                        ]
        
            plot_smooth_reward2(reward_list)

        elif False:
            reward_df1 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square2other/square2circular/seed1.csv', header=None)
            reward_df2 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square2other/square2circular/seed2.csv', header=None)
            reward_df3 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square2other/square2circular/seed3.csv', header=None)    
            reward_array1 = reward_df1.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
            reward_array2 = reward_df2.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
            reward_array3 = reward_df3.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        
            reward_df1_close = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square2other/square2triangular5/seed1.csv', header=None)
            reward_df2_close = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square2other/square2triangular5/seed2.csv', header=None)
            reward_df3_close = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square2other/square2triangular5/seed3.csv', header=None)    
            reward_array1_close = reward_df1_close.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
            reward_array2_close = reward_df2_close.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
            reward_array3_close = reward_df3_close.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
            
            reward_df1_far = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square2other/square2triangular10/seed1.csv', header=None)
            reward_df2_far = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square2other/square2triangular10/seed2.csv', header=None)
            reward_df3_far = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square2other/square2triangular10/seed3.csv', header=None)    
            reward_array1_far = reward_df1_far.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
            reward_array2_far = reward_df2_far.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
            reward_array3_far = reward_df3_far.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
            
            reward_list = [np.concatenate((reward_array1, reward_array2, reward_array3)),
                        np.concatenate((reward_array1_close, reward_array2_close, reward_array3_close)),
                        np.concatenate((reward_array1_far, reward_array2_far, reward_array3_far)),
                        ]
        
            plot_smooth_reward2_new(reward_list)
        
    elif False:#reward2
        # circle
        reward_df1 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/circle_square_triangle/circle/train_episode_reward1.csv', header=None)
        reward_df2 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/circle_square_triangle/circle/train_episode_reward2.csv', header=None)
        reward_df3 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/circle_square_triangle/circle/train_episode_reward3.csv', header=None)    
        reward_array1 = reward_df1.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array2 = reward_df2.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array3 = reward_df3.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
    
        # square
        reward_df1_close = pd.read_csv('~/Workspaces/rl_ws/spirl/results/circle_square_triangle/square/train_episode_reward1.csv', header=None)
        reward_df2_close = pd.read_csv('~/Workspaces/rl_ws/spirl/results/circle_square_triangle/square/train_episode_reward2.csv', header=None)
        reward_df3_close = pd.read_csv('~/Workspaces/rl_ws/spirl/results/circle_square_triangle/square/train_episode_reward3.csv', header=None)    
        reward_array1_close = reward_df1_close.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array2_close = reward_df2_close.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array3_close = reward_df3_close.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        
        # triangle
        reward_df1_far = pd.read_csv('~/Workspaces/rl_ws/spirl/results/circle_square_triangle/triangle/train_episode_reward1.csv', header=None)
        reward_df2_far = pd.read_csv('~/Workspaces/rl_ws/spirl/results/circle_square_triangle/triangle/train_episode_reward2.csv', header=None)
        reward_df3_far = pd.read_csv('~/Workspaces/rl_ws/spirl/results/circle_square_triangle/triangle/train_episode_reward3.csv', header=None)    
        reward_array1_far = reward_df1_far.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array2_far = reward_df2_far.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array3_far = reward_df3_far.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        
        
        reward_list = [np.concatenate((reward_array1, reward_array2, reward_array3)),
                       np.concatenate((reward_array1_far, reward_array2_far, reward_array3_far)),
                       np.concatenate((reward_array1_close, reward_array2_close, reward_array3_close))
                       ]
    
        plot_smooth_reward3(reward_list)
    
    
    
    