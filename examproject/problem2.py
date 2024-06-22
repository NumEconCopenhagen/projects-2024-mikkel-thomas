# Write your code here
import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})

# Write your answer here 

def carrer_choice(par):
    u_ij_prior = np.nan + np.zeros((par.N))
    u_ij_real = np.nan + np.zeros((par.N))
    u_ijF = np.nan + np.zeros((par.J))
    j_star = np.nan + np.zeros((par.N))

    for i in range(par.N):
        eps_fj = np.random.normal(0,par.sigma,(par.J, i+1))
        eps_ij = np.random.normal(0,par.sigma,(par.J))

        for j, jv in enumerate(par.v):
            u_ijF[j] = par.v[j] + np.mean(eps_fj[j,:])
        j_star[i] = np.argmax(u_ijF)
        j_star_temp  = np.argmax(u_ijF)
        u_ij_prior[i] = u_ijF[j_star_temp] # Prior expectation of the utility
        u_ij_real[i] = u_ijF[j_star_temp] + eps_ij[j_star_temp] # Realization of the utility
    return u_ij_prior, u_ij_real, j_star

def carrer_choice_outer(par):
    """
    This function performs the outer loop of the simulation i.e. it calls the carrer_choice function K times and stores the results in arrays.

    Args:
    - par: A SimpleNamespace containing the parameters of the simulation.
    """
    np.random.seed(2024)
    u_ij_prior_avg = np.zeros((par.N,par.K))
    u_ij_real_avg = np.zeros((par.N,par.K))
    j_star_avg = np.zeros((par.N,par.K))

    for k in range(par.K):
        u_ij_prior, u_ij_real, j_star = carrer_choice(par)
        u_ij_prior_avg[:,k] = u_ij_prior
        u_ij_real_avg[:,k] = u_ij_real
        j_star_avg[:,k] = j_star

    return u_ij_prior_avg, u_ij_real_avg, j_star_avg

def sum_stats(par):

    u_ij_prior_avg, u_ij_real_avg, j_star_avg = carrer_choice_outer(par)

    shares = np.zeros((par.N,par.J))
    for i in range(par.N):
        for j in range(par.J):
            shares[i,j] = np.sum(j_star_avg[i,:] == j)/par.K

    # Example of applying a mask to calculate conditional expected utilities
    conditional_u_ij_real_avg = np.zeros((par.N, par.J))
    conditional_u_ij_prior_avg = np.zeros((par.N, par.J))

    for i in range(par.N):
        for j in range(par.J):
            # Create a mask for when the career choice j is selected
            mask = j_star_avg[i,:] == j
            # Apply the mask to u_ij_real_avg and calculate the mean for the selected career choice
            conditional_u_ij_real_avg[i,j] = np.mean(u_ij_real_avg[i, mask])
            conditional_u_ij_prior_avg[i,j] = np.mean(u_ij_prior_avg[i, mask])

    return shares, conditional_u_ij_real_avg, conditional_u_ij_prior_avg

def plot_p2(par):

    shares, conditional_u_ij_real_avg, conditional_u_ij_prior_avg = sum_stats(par)

    # Assuming 'shares' is your array from the previous step
    # And assuming 'par.N' is the number of graduate types and 'par.J' is the number of career tracks

    # Set the labels for each type of graduate and career tracks for legend
    graduate_types = [f'Type {i+1}' for i in range(par.N)]
    career_tracks = [f'Track {j+1}' for j in range(par.J)]

    # Set up the figure and axes for the bar chart
    # fig, ax = plt.subplots(figsize=(10, 6))
    fig = plt.figure(figsize=(18,6))

    ax = fig.add_subplot(1,3,1)

    # Positions of the bars on the x-axis
    indices = np.arange(par.N)

    # Initialize an array to keep track of the bottom position for each stack
    bottoms = np.zeros(par.N)

    for j in range(par.J):
        # Plotting each career track as a stacked bar
        ax.bar(indices, shares[:, j], bottom=bottoms, label=career_tracks[j])
        # Update the bottoms for the next stack
        bottoms += shares[:, j]

    # Adding labels and title
    ax.set_xlabel('Graduate Types', fontsize=10)
    ax.set_ylabel('Share', fontsize=10)
    ax.set_title('Share of Graduates Choosing Career Tracks', fontsize=16)

    # Customizing tick labels to show graduate types
    ax.set_xticks(indices)
    ax.set_xticklabels(graduate_types, fontsize=10, rotation=45)

    # Adding legend
    ax.legend()

    ax2 = fig.add_subplot(1,3,2)

    # Adding labels and title
    ax2.set_xlabel('Graduate Types', fontsize=10)
    bar_width = 0.25
    for j in range(par.J):
        # Plotting each career track
        ax2.bar(indices + j*bar_width, conditional_u_ij_real_avg[:, j], width=bar_width, label=career_tracks[j])

    # Adding labels and title
    ax2.set_xlabel('Graduate Types', fontsize=10)
    ax2.set_ylabel('Share', fontsize=14)
    ax2.set_ylabel('Utility', fontsize=10)
    ax2.set_title('Realized Utility of Career Tracks', fontsize=16)

    # Adding ticks and customizing tick labels to show graduate types
    ax2.set_xticks(indices + bar_width / par.J)
    ax2.set_xticklabels(graduate_types, fontsize=10, rotation=45)

    ax3 = fig.add_subplot(1,3,3)

    # Adding labels and title
    ax3.set_xlabel('Graduate Types', fontsize=10)
    bar_width = 0.25
    for j in range(par.J):
        # Plotting each career track
        ax3.bar(indices + j*bar_width, conditional_u_ij_prior_avg[:, j], width=bar_width, label=career_tracks[j])

    # Adding labels and title
    ax3.set_xlabel('Graduate Types', fontsize=10)
    ax3.set_ylabel('Share', fontsize=14)
    ax3.set_ylabel('Utility', fontsize=10)
    ax3.set_title('Prior Utility of Career Tracks', fontsize=16)

    # Adding ticks and customizing tick labels to show graduate types
    ax3.set_xticks(indices + bar_width / par.J)
    ax3.set_xticklabels(graduate_types, fontsize=10, rotation=45)

    # Show plot
    plt.tight_layout()
    plt.show()


def carrer_choice_y2(par):
    u_ij_prior = np.nan + np.zeros((par.N))
    u_ij_prior_switch = np.nan + np.zeros((par.N))
    u_ij_real = np.nan + np.zeros((par.N))
    u_ij_real_switch = np.nan + np.zeros((par.N))
    u_ijF = np.nan + np.zeros((par.J))
    u_ijF_swtich = np.nan + np.zeros((par.J))
    j_star = np.nan + np.zeros((par.N))
    j_star_switch = np.nan + np.zeros((par.N))

    for i in range(par.N):
        eps_fj = np.random.normal(0,par.sigma,(par.J, i+1))
        eps_ij = np.random.normal(0,par.sigma,(par.J))

        # Calculate the expected utility for each career path
        for j, jv in enumerate(par.v):
            u_ijF[j] = par.v[j] + np.mean(eps_fj[j,:])
        
        # Find the career path that maximizes the expected utility
        j_star[i] = np.argmax(u_ijF)
        j_star_temp  = np.argmax(u_ijF)

        # Update the expected utility for each career path when the switching cost is applied
        for j, jv in enumerate(par.v):
            if j == j_star_temp:
                u_ijF_swtich[j] = par.v[j] - par.c
            else:
                u_ijF_swtich[j] = u_ijF[j]

        # Find the career path that maximizes the expected utility
        j_star_switch[i] = np.argmax(u_ijF_swtich)
        j_star_temp_switch  = np.argmax(u_ijF_swtich)

        # Find the career path that maximizes the expected utility with the switching cost
        u_ij_prior_switch[i] = u_ijF_swtich[j_star_temp_switch] # Prior expectation of the utility

        if j_star_temp_switch == j_star_temp:
            u_ij_real_switch[i] = par.v[j_star_temp_switch] # Realization of the utility
        else:
            u_ij_real_switch[i] = par.v[j_star_temp_switch] - par.c
 
    return u_ij_prior_switch, u_ij_real_switch, j_star_switch

def carrer_choice_outer_y2(par):
    np.random.seed(2024)
    u_ij_prior_avg_y2 = np.zeros((par.N,par.K))
    u_ij_real_avg_y2 = np.zeros((par.N,par.K))
    j_star_avg_y2 = np.zeros((par.N,par.K))

    for k in range(par.K):
        u_ij_prior, u_ij_real, j_star = carrer_choice_y2(par)
        u_ij_prior_avg_y2[:,k] = u_ij_prior
        u_ij_real_avg_y2[:,k] = u_ij_real
        j_star_avg_y2[:,k] = j_star

    return u_ij_prior_avg_y2, u_ij_real_avg_y2, j_star_avg_y2

def sum_stats_y2(par):
    _, _, j_star_avg = carrer_choice_outer(par)
    u_ij_prior_avg_y2, u_ij_real_avg_y2, j_star_avg_y2 = carrer_choice_outer_y2(par)
    difference_matrix = np.where(j_star_avg_y2 != j_star_avg, 1, 0)

    # Example of applying a mask to calculate conditional expected utilities
    conditional_switch_share = np.zeros((par.N, par.J))

    for i in range(par.N):
        for j in range(par.J):
            # Create a mask for when the career choice j is selected
            mask = j_star_avg[i,:] == j
            # Apply the mask to u_ij_real_avg and calculate the mean for the selected career choice
            conditional_switch_share[i,j] = np.mean(difference_matrix[i, mask])

    u_ij_prior_avg_y2_mean = np.mean(u_ij_prior_avg_y2, axis = 1) # Prior expected utility after switching
    u_ij_real_avg_y2_mean = np.mean(u_ij_real_avg_y2, axis = 1) # Realized utility after switching

    return conditional_switch_share, u_ij_prior_avg_y2_mean, u_ij_real_avg_y2_mean

def plot_p3(par):

    conditional_switch_share, u_ij_prior_avg_y2_mean, u_ij_real_avg_y2_mean = sum_stats_y2(par)

    # Set the labels for each type of graduate and career tracks for legend
    graduate_types = [f'Type {i+1}' for i in range(par.N)]
    career_tracks = [f'Track {j+1}' for j in range(par.J)]
    indices = np.arange(par.N)

    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1,2,1)
    ax.plot(u_ij_prior_avg_y2_mean, label='Prior Expected Utility')
    ax.plot(u_ij_real_avg_y2_mean, label='Realized Utility')
    ax.legend()
    ax.set_xlabel('Graduate Types', fontsize=14)
    ax.set_ylabel('Average utility', fontsize=14)

    ax3 = fig.add_subplot(1,2,2)

    # Adding labels and title
    ax3.set_xlabel('Graduate Types', fontsize=14)
    bar_width = 0.25
    for j in range(par.J):
        # Plotting each career track
        ax3.bar(indices + j*bar_width, conditional_switch_share[:, j], width=bar_width, label=career_tracks[j])

    # Adding labels and title
    ax3.set_xlabel('Graduate Types', fontsize=14)
    ax3.set_ylabel('Share', fontsize=14)
    ax3.set_title('Swtich share conditional on initial choice', fontsize=16)

    # Adding ticks and customizing tick labels to show graduate types
    ax3.set_xticks(indices + bar_width / par.J)
    ax3.set_xticklabels(graduate_types, fontsize=10, rotation=45)
    ax3.legend()
    # Show plot
    plt.tight_layout()
    plt.show()