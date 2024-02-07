import sys
print(sys.version)
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

sys.path.append('../')
from circular_shift_helper import shift_perm_distribution
import MiStatHelpers as mi_helper


n_bins = 6
n_iterations = 400
col_to_perm = 'Angle'
event_col = 'Saccade_Indicator'
group_col = 'trial'
bins_col='Angle_bins'
min_time_between=int(sys.argv[1])
n_jobs = 8

print(min_time_between,n_jobs)
for number_of_events_per_second in [1,2,4,8,16]:#[1,2,4,8,16]:
    if number_of_events_per_second * min_time_between < 500:
        print('running:',min_time_between,number_of_events_per_second)
        for par in range(300):
            for start_str in ['aligned','shifted']:
                par_str = f'par-{par}_trials-100_cycles-3_lencycle-4_stdcycle-(0.75,)_eps-{number_of_events_per_second}_mtb-{min_time_between}.csv'
                full_path = f'simulated_participants/{start_str}_trial_starts/{start_str}_{par_str}'
                out_str = f'simulated_participants/{start_str}_trial_starts_mi_stats_batch/{start_str}_{par_str[:-4]}.npz'

                aligned_start_df = pd.read_csv(full_path, usecols=[col_to_perm,event_col,group_col])
                aligned_start_df[bins_col]=mi_helper.create_bins_for_col(
                    aligned_start_df,
                    'Angle',
                    n_bins=n_bins
                )
                trial_df = aligned_start_df
                event_df = trial_df[trial_df[event_col]==1]
                prob_dist_real = (event_df[bins_col].value_counts())/len(event_df)
                par_mod_index = mi_helper.calc_modulation_index(prob_dist_real,6)
                random_null = mi_helper.create_permute_null_distribution(
                    aligned_start_df,
                    col_to_perm,
                    event_col,
                    n_bins,
                    n_iterations=n_iterations,
                    tqdm_disabled=True
                )
                try:
                    shifted_null = Parallel(n_jobs=n_jobs)(delayed(shift_perm_distribution)(aligned_start_df, 
                                                                                 group_col, col_to_perm, 
                                                                                 event_col, n_bins) for i in range(n_iterations))


                    np.savez(out_str,
                             real_mi = par_mod_index,
                             shifted_null=shifted_null,
                             random_null=random_null
                            )
                except:
                    print('except')
                    out_str = f'simulated_participants/{start_str}_trial_starts_mi_stats/shift_error_{start_str}_{par_str[:-4]}.npz'
                    np.savez(out_str,
                             real_mi = par_mod_index,
                             shifted_null=[],
                             random_null=random_null
                            )
                        
                