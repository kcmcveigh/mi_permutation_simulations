import simulation_helpers as sim_helpers
import circular_shift_helper as shift_helper

n_trials = 100
num_cycles=3
cycle_mean_sec=4
cycle_std_sec=0.75,

for min_time_between in [0,25,50,100]:#[0,25,50,100,75]:
    for number_of_events_per_second in [1,2,4,8,16]:
        if number_of_events_per_second * min_time_between < 500:
            for par in range(0,300):
                #first we'll generate 
                df = sim_helpers.create_simulated_trials_df(
                    n_trials,
                    num_cycles,
                    min_time_between_events = min_time_between,
                    desired_events_per_1000 = number_of_events_per_second,
                    mean_cycle_time_sec=cycle_mean_sec,
                    std_dev_cycle_time_sec=cycle_std_sec,
                ) 
                #df.to_csv(f'simulated_participants/aligned_trial_starts/aligned_par-{par}_trials-{n_trials}_cycles-{num_cycles}_lencycle-{cycle_mean_sec}_stdcycle-{cycle_std_sec}_eps-{number_of_events_per_second}_mtb-{min_time_between}.csv')
                df['Angle']= df.groupby('trial')['Angle'].transform(shift_helper.circular_shift)
                #df.to_csv(f'simulated_participants/shifted_trial_starts/shifted_par-{par}_trials-{n_trials}_cycles-{num_cycles}_lencycle-{cycle_mean_sec}_stdcycle-{cycle_std_sec}_eps-{number_of_events_per_second}_mtb-{min_time_between}.csv')
                if par % 50 ==0:
                    print(par,min_time_between,number_of_events_per_second)