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
    colors = ['purple', 'green', 'orange', 'red']
    for i_list in range(len(reward_list)):
        if True:
            for i in range(3):
                reward_list[i_list][i] = smooth(reward_list[i_list][i], 6)
                
        # rewards mean and std                  
        mean = reward_list[i_list].mean(axis=0)[0:100]
        std = reward_list[i_list].std(axis=0)[0:100]
        plot_mean_and_CI(x, mean, mean-std, mean+std, colors[i_list], colors[i_list])

    plt.legend(labels=["square 5°","square 10°","triangle 5°", "triangle 10°"],loc='lower right')
    plt.xlabel('training episodes')
    plt.ylabel('reward')
    plt.savefig('reward2.png')
    
def plot_smooth_reward3(reward_list):
    sns.set(style="darkgrid", font_scale=1.3)
    x = np.linspace(0, 99, 100)
    colors = ['purple', 'green', 'orange', 'blue', "deepskyblue"]
    for i_list in range(len(reward_list)):
        if True:
            for i in range(3):
                reward_list[i_list][i] = smooth(reward_list[i_list][i], 5)
                
        # rewards mean and std                  
        mean = reward_list[i_list].mean(axis=0)[0:100]
        std = reward_list[i_list].std(axis=0)[0:100]
        plot_mean_and_CI(x, mean, mean-std, mean+std, colors[i_list], colors[i_list])

    plt.yticks([0, 2, 4, 6, 8, 10])
    plt.legend(labels=["VIA-SPiRL", "SPiRL", "SAC", "BC+SAC", "PILCO"],loc='lower right')
    plt.xlabel('Environment episodes')
    plt.ylabel('reward')
    plt.title('Triangular Insertion')
    plt.show()
    plt.savefig('reward3.png')

def plot_alpha(alpha_list):
    sns.set(style="darkgrid", font_scale=1.3)
    # sns.set_style("whitegrid", {'grid.linestyle': '--'})
    plt.figure(figsize=(8,6))
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    x = np.linspace(0, 99, 100)

    colors = ['red', 'green', 'blue']
    # for i in range(len(alpha_list)):
        # alpha_list[i] = smooth(alpha_list[i], 5)
    plt.plot(x, alpha_list[0], colors[0], linewidth=4.0)
    plt.plot(x, alpha_list[1], colors[1], alpha=0.7, linewidth=3.0)
    plt.plot(x, alpha_list[2], colors[2], alpha=0.7, linewidth=3.0)

    font_scale = 16
    plt.legend(labels=["Circular", "Square", "Triangular"], loc='upper right', prop={'size': 16})
    plt.xlabel('Environment episodes', fontsize=font_scale)
    # plt.ylabel('Temperature weight', fontsize=font_scale)
    plt.gca().set_ylabel(r'Entropy weight $\alpha$', fontsize=font_scale)
    plt.show()
    # plt.title('Maze Navigation', fontsize=20)
    plt.savefig('alpha.png')

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

if __name__ == '__main__':
    #wrench = reward_df.iloc[:,18:21].values.astype(np.float32)
    #plot_wrench(wrench)
    
    # stiffness_df = pd.read_csv('~/Workspaces/rl_ws/spirl/variable_stiffness.csv', header=None)
    # stiffness_array = stiffness_df.iloc[:,0:4].values.astype(np.float32)#.reshape(-1,120)
    # plot_variable_stiffness(stiffness_array)

    
    # circle
    if False:
        # via-spirl
        reward_df1 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/circle/via-spirl/seed1.csv', header=None)
        reward_df2 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/circle/via-spirl/seed2.csv', header=None)
        reward_df3 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/circle/via-spirl/seed3.csv', header=None)    
        reward_array1 = reward_df1.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array2 = reward_df2.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array3 = reward_df3.iloc[:120,0].values.astype(np.float32).reshape(-1,120)

        # spirl
        reward_df1_spirl = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/circle/spirl/seed1.csv', header=None)
        reward_df2_spirl = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/circle/spirl/seed2.csv', header=None)
        reward_df3_spirl = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/circle/spirl/seed3.csv', header=None)    
        reward_array1_spirl = reward_df1_spirl.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array2_spirl = reward_df2_spirl.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array3_spirl = reward_df3_spirl.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
    
        # sac
        reward_df1_sac = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/circle/sac/seed1.csv', header=None)
        reward_df2_sac = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/circle/sac/seed2.csv', header=None)
        reward_df3_sac = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/circle/sac/seed3.csv', header=None)    
        reward_array1_sac = reward_df1_sac.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array2_sac = reward_df2_sac.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array3_sac = reward_df3_sac.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        
        # bc+sac
        reward_df1_bc = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/circle/bc/seed1.csv', header=None)
        reward_df2_bc = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/circle/bc/seed2.csv', header=None)
        reward_df3_bc = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/circle/bc/seed3.csv', header=None)    
        reward_array1_bc = reward_df1_bc.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array2_bc = reward_df2_bc.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array3_bc = reward_df3_bc.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        
        reward_list = [np.concatenate((reward_array1, reward_array2, reward_array3)),
                       np.concatenate((reward_array1_spirl, reward_array2_spirl, reward_array3_spirl)),
                       np.concatenate((reward_array1_sac, reward_array2_sac, reward_array3_sac)),
                       np.concatenate((reward_array1_bc, reward_array2_bc, reward_array3_bc))]

        # pilco
        reward_df1_pilco = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/circle/pilco/seed1.csv', header=None)
        reward_df2_pilco = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/circle/pilco/seed2.csv', header=None)
        reward_df3_pilco = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/circle/pilco/seed3.csv', header=None)    
        reward_array1_pilco = reward_df1_pilco.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array2_pilco = reward_df2_pilco.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array3_pilco = reward_df3_pilco.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        
        reward_list = [np.concatenate((reward_array1, reward_array2, reward_array3)),
                       np.concatenate((reward_array1_spirl, reward_array2_spirl, reward_array3_spirl)),
                       np.concatenate((reward_array1_sac, reward_array2_sac, reward_array3_sac)),
                       np.concatenate((reward_array1_bc, reward_array2_bc, reward_array3_bc)),
                       np.concatenate((reward_array1_pilco, reward_array2_pilco, reward_array3_pilco))]

        plot_smooth_reward3(reward_list)
        
    # square
    if False:
        # via-spirl
        reward_df1 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square/via-spirl/seed1.csv', header=None)
        reward_df2 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square/via-spirl/seed2.csv', header=None)
        reward_df3 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square/via-spirl/seed3.csv', header=None)    
        reward_array1 = reward_df1.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array2 = reward_df2.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array3 = reward_df3.iloc[:120,0].values.astype(np.float32).reshape(-1,120)

        # spirl
        reward_df1_spirl = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square/spirl/seed1.csv', header=None)
        reward_df2_spirl = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square/spirl/seed2.csv', header=None)
        reward_df3_spirl = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square/spirl/seed3.csv', header=None)    
        reward_array1_spirl = reward_df1_spirl.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array2_spirl = reward_df2_spirl.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array3_spirl = reward_df3_spirl.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
    
        # sac
        reward_df1_sac = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square/sac/seed1.csv', header=None)
        reward_df2_sac = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square/sac/seed2.csv', header=None)
        reward_df3_sac = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square/sac/seed3.csv', header=None)    
        reward_array1_sac = reward_df1_sac.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array2_sac = reward_df2_sac.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array3_sac = reward_df3_sac.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        
        # bc+sac
        reward_df1_bc = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square/bc/seed1.csv', header=None)
        reward_df2_bc = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square/bc/seed2.csv', header=None)
        reward_df3_bc = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square/bc/seed3.csv', header=None)    
        reward_array1_bc = reward_df1_bc.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array2_bc = reward_df2_bc.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array3_bc = reward_df3_bc.iloc[:120,0].values.astype(np.float32).reshape(-1,120)

        # pilco
        reward_df1_pilco = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square/pilco/seed1.csv', header=None)
        reward_df2_pilco = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square/pilco/seed2.csv', header=None)
        reward_df3_pilco = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square/pilco/seed3.csv', header=None)    
        reward_array1_pilco = reward_df1_pilco.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array2_pilco = reward_df2_pilco.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array3_pilco = reward_df3_pilco.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        
        
        reward_list = [np.concatenate((reward_array1, reward_array2, reward_array3)),
                       np.concatenate((reward_array1_spirl, reward_array2_spirl, reward_array3_spirl)),
                       np.concatenate((reward_array1_sac, reward_array2_sac, reward_array3_sac)),
                       np.concatenate((reward_array1_bc, reward_array2_bc, reward_array3_bc)),
                       np.concatenate((reward_array1_pilco, reward_array2_pilco, reward_array3_pilco))]
    
        plot_smooth_reward3(reward_list)

    # triangle  
    elif False:
        # via-spirl
        reward_df1 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/triangle/via-spirl/seed1.csv', header=None)
        reward_df2 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/triangle/via-spirl/seed2.csv', header=None)
        reward_df3 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/triangle/via-spirl/seed3.csv', header=None)    
        reward_array1 = reward_df1.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array2 = reward_df2.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array3 = reward_df3.iloc[:120,0].values.astype(np.float32).reshape(-1,120)

        # spirl
        reward_df1_spirl = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/triangle/spirl/seed1.csv', header=None)
        reward_df2_spirl = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/triangle/spirl/seed2.csv', header=None)
        reward_df3_spirl = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/triangle/spirl/seed3.csv', header=None)    
        reward_array1_spirl = reward_df1_spirl.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array2_spirl = reward_df2_spirl.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array3_spirl = reward_df3_spirl.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
    
        # sac
        reward_df1_sac = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/triangle/sac/seed1.csv', header=None)
        reward_df2_sac = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/triangle/sac/seed2.csv', header=None)
        reward_df3_sac = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/triangle/sac/seed3.csv', header=None)    
        reward_array1_sac = reward_df1_sac.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array2_sac = reward_df2_sac.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array3_sac = reward_df3_sac.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        
        # bc+sac
        reward_df1_bc = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/triangle/bc/seed1.csv', header=None)
        reward_df2_bc = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/triangle/bc/seed2.csv', header=None)
        reward_df3_bc = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/triangle/bc/seed3.csv', header=None)    
        reward_array1_bc = reward_df1_bc.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array2_bc = reward_df2_bc.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array3_bc = reward_df3_bc.iloc[:120,0].values.astype(np.float32).reshape(-1,120)

        # pilco
        reward_df1_pilco = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/triangle/pilco/seed1.csv', header=None)
        reward_df2_pilco = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/triangle/pilco/seed2.csv', header=None)
        reward_df3_pilco = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/triangle/pilco/seed3.csv', header=None)    
        reward_array1_pilco = reward_df1_pilco.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array2_pilco = reward_df2_pilco.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        reward_array3_pilco = reward_df3_pilco.iloc[:120,0].values.astype(np.float32).reshape(-1,120)
        
        
        reward_list = [np.concatenate((reward_array1, reward_array2, reward_array3)),
                       np.concatenate((reward_array1_spirl, reward_array2_spirl, reward_array3_spirl)),
                       np.concatenate((reward_array1_sac, reward_array2_sac, reward_array3_sac)),
                       np.concatenate((reward_array1_bc, reward_array2_bc, reward_array3_bc)),
                       np.concatenate((reward_array1_pilco, reward_array2_pilco, reward_array3_pilco))]

    
        plot_smooth_reward3(reward_list)

    # barchart: position offset(circular)
    elif False:
        bar_df = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/offset.csv')

        sns.set(style="darkgrid", font_scale=1.3)
        sns.barplot(x = "Circular position offset", y = "Evaluation score", hue = "method", ci = "sd", capsize = 0.1, data = bar_df, saturation=1.0)
        plt.yticks([0, 2, 4, 6, 8, 10, 12])
        plt.legend(loc='upper right')
        plt.show()

    # barchart: initial angles(square) 
    elif False:
        bar_df = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/square_rotation.csv')

        sns.set(style="darkgrid", font_scale=1.3)
        sns.barplot(x = "Square initial angles", y = "Evaluation score", hue = "method", ci = "sd", capsize = 0.1, data = bar_df, saturation=1.0)
        plt.yticks([0, 2, 4, 6, 8, 10, 12])
        plt.legend(loc='upper right')
        plt.show()

    # barchart: initial angles(triangular)
    elif True:
        bar_df = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/triangle_rotation.csv')

        sns.set(style="darkgrid", font_scale=1.3)
        sns.barplot(x = "Triangular initial angles", y = "Evaluation score", hue = "method", ci = "sd", capsize = 0.1, data = bar_df, saturation=1.0)
        plt.yticks([0, 2, 4, 6, 8, 10, 12])
        plt.legend(loc='upper right')
        plt.show()

    elif False:
        alpha_df1 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/alpha/alpha1.csv', header=None)
        alpha_df2 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/alpha/alpha2.csv', header=None)
        alpha_df3 = pd.read_csv('~/Workspaces/rl_ws/spirl/results/ral/alpha/alpha3.csv', header=None)
        alpha_array1 = alpha_df1.iloc[:].values.astype(np.float32).reshape(100,)
        alpha_array2 = alpha_df2.iloc[:].values.astype(np.float32).reshape(100,)
        alpha_array3 = alpha_df3.iloc[:].values.astype(np.float32).reshape(100,)

        alpha_list = [alpha_array1, alpha_array2, alpha_array3]
        plot_alpha(alpha_list)

        # x = np.linspace(0, 100, 100)
        # y1 = np.exp(-0.11*x)
        # y2 = np.exp(-0.07*x)
        # y3 = np.exp(-0.04*x)
        # np.savetxt('/home/quantao/Workspaces/rl_ws/spirl/results/ral/alpha/alpha1.csv', y1, delimiter='\n')
        # np.savetxt('/home/quantao/Workspaces/rl_ws/spirl/results/ral/alpha/alpha2.csv', y2, delimiter='\n') 
        # np.savetxt('/home/quantao/Workspaces/rl_ws/spirl/results/ral/alpha/alpha3.csv', y3, delimiter='\n') 

        # plt.figure()
        # plt.plot(x, y1)
        # plt.plot(x, y2)
        # plt.plot(x, y3)
        # plt.show()



    
    
    
    