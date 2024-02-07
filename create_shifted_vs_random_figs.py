import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt


gloss_names = {'aligned': 'fixed angle', 'random': 'random angle'}
all_eps_options = np.array([1,2,4,8,16])
for min_time_between in [0,25,50,100]:
    eps_list = all_eps_options[(all_eps_options*min_time_between)<500]
    for start_str in ['aligned', 'random']:
        # Create a figure for each start_str
        fig, axs = plt.subplots(len(eps_list), 2, figsize=(12, 4 * 5))  # Adjust the size as needed
        fig.suptitle(f'Start: {gloss_names[start_str]}, Min Time: {min_time_between}',fontsize=16) 
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        
        stat_start_str = 'shifted' if start_str =='random' else start_str

        for idx, number_of_events_per_second in enumerate(eps_list):
            random_percentile_list, shifted_percentile_list = [], []
            for par in range(300):
                par_str = f'par-{par}_trials-100_cycles-3_lencycle-4_stdcycle-(0.75,)_eps-{number_of_events_per_second}_mtb-{min_time_between}'
                out_str = f'simulated_biologically_plausible_participants_{start_str}_stats/{stat_start_str}_{par_str}.npz'
                data = np.load(out_str)
                random_null_dist = data['random_null']
                shifted_null_dist = data['shifted_null']
                real = data['real_mi']
                random_percentile = stats.percentileofscore(random_null_dist, real)
                random_percentile_list.append(random_percentile)
                shifted_percentile = stats.percentileofscore(shifted_null_dist, real)
                shifted_percentile_list.append(shifted_percentile)

            # Plot histograms in the respective rows
            axs[idx, 0].hist(random_percentile_list)
            axs[idx, 0].set_title(f'Random Permutations Mean: {round(np.mean(random_percentile_list), 2)}')

            axs[idx, 1].hist(shifted_percentile_list)
            axs[idx, 1].set_title(f'Shifted Permutations Mean: {round(np.mean(shifted_percentile_list), 2)}')

            # Set title for each row
            axs[idx, 0].set_ylabel(f'EPS: {number_of_events_per_second}')

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        fig_save_str = f'figures_for_shift_vs_random/start-{start_str}_min_time-{min_time_between}.png'
        plt.savefig(fig_save_str)
        plt.show()
