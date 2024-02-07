import numpy as np
import pandas as pd
# Helper functions
def generate_cycle_lengths(num_cycles=200, mean=4, std_dev=0.75):
    """Generate cycle lengths from a normal distribution."""
    return np.random.normal(mean, std_dev, num_cycles)

def assign_milliseconds(cycle_lengths):
    """Convert cycle lengths to milliseconds."""
    return (cycle_lengths * 1000).astype(int)

def get_millisecond_ranges(cycle_lengths_ms):
    """Get ranges of milliseconds for each cycle."""
    last_end = 0
    ranges = []
    for length in cycle_lengths_ms:
        start = last_end
        end = start + length
        last_end = end
        ranges.append((start, end))
    return ranges
# TODO clarify seconds vs hz
def generate_angles_for_cycle(start, end):
    """Generate angle values evenly distributed within the cycle."""
    cycle_length = end - start
    return np.linspace(0, 360, cycle_length, endpoint=False)

def generate_saccades_indicator_for_cycle(start, end, frequency=3):
    """Generate an indicator array for saccades within a cycle using Poisson distribution."""
    cycle_length = end - start
    # Poisson distribution for the number of saccades
    num_saccades = np.random.poisson(cycle_length * frequency / 1000)
    saccade_positions = np.random.choice(cycle_length, num_saccades, replace=False) + start
    indicators = np.zeros(cycle_length)
    for pos in saccade_positions:
        indicators[pos - start] = 1
    return indicators

# Main simulation function
def simulate_cycle_data(
    num_cycles = 200,
    mean=4, 
    std_dev=0.75
):
    # Generate cycle lengths and ranges
    cycle_lengths = generate_cycle_lengths(
        num_cycles=num_cycles
    )
    cycle_lengths_ms = assign_milliseconds(cycle_lengths)
    millisecond_ranges = get_millisecond_ranges(cycle_lengths_ms)

    # Prepare lists to hold time series, saccade indicators, and angles
    time_series = []
    cycle_start = []
    angles = []

    # Populate the lists
    for start, end in millisecond_ranges:
        n_samples = end-start
        cycle_range = range(start, end)
        time_series.extend(cycle_range)
        cycle_start.extend(np.ones(n_samples)*start)
        
        angles.extend(generate_angles_for_cycle(start, end))
       # saccade_indicators.extend(generate_saccades_indicator_for_cycle(start, end))

    # Construct the DataFrame
    df = pd.DataFrame({
        'Time_MS': time_series,
         'Cycle_Start':cycle_start,
        #'Saccade_Indicator': saccade_indicators,
        'Angle': angles
    })
    n = round(len(time_series)/1000*3)
    df['Saccade_Indicator']=0
    df.loc[df.sample(n=n).index,'Saccade_Indicator'] = 1

    return df

def generate_biologically_plausible_events(
    cycle_angles,
    initial_probability,
    increment_probability,
    min_samples_between_events = 50
):
    '''
    function to generate events under some biologically plausible constraints because the biology of many events
    constrains events such true randomness i.e. two independent events in adjacent time samples is biologically
    impossible
    len_time_series (param): int length of series of events you want to generate
    intial_probaility (param): float how likely is an event to occur at a baseline rate
    increment_probability (param): float how much with each passing time sample should an event become more probably
    min_samples_between_events (param): int how many samples are there at minimum between events
    '''
    # Generating events based on a biologically possible processes
    event_series_dynamic = np.zeros(len(cycle_angles))
    p_current = initial_probability
    event_i=0
    for i,angle in enumerate(cycle_angles):
        if (i-event_i)>min_samples_between_events:#assume no events occur within 50 ms of each other
            event_occurs = np.random.rand() < p_current
            event_series_dynamic[i] = event_occurs

            # Update probability based on whether an event occurred or not
            if event_occurs:
                event_i = i
                p_current = 0  # Reset probability to zero after an event
            else:
                p_current = min(initial_probability, p_current + increment_probability)
    return event_series_dynamic

def generate_biologically_plausible_events_no_ramp(
    cycle_angles,
    initial_probability,
    min_samples_between_events = 50
):
    '''
    function to generate events under some biologically plausible constraints because the biology of many events
    constrains events such true randomness i.e. two independent events in adjacent time samples is biologically
    impossible
    len_time_series (param): int length of series of events you want to generate
    intial_probaility (param): float how likely is an event to occur at a baseline rate
    increment_probability (param): float how much with each passing time sample should an event become more probably
    min_samples_between_events (param): int how many samples are there at minimum between events
    '''
    # Generating events based on a biologically possible processes
    event_series_dynamic = np.zeros(len(cycle_angles))
    p_current = initial_probability
    event_i=0
    for i,angle in enumerate(cycle_angles):
        if (i-event_i) > min_samples_between_events:#assume no events occur within 50 ms of each other
            event_occurs = np.random.rand() < p_current
            if event_occurs:
                event_series_dynamic[i] = event_occurs
                event_i = i
    return event_series_dynamic

def create_simulated_trials_df(
    num_trials,
    num_cycles,
    desired_events_per_1000=3.5,
    min_time_between_events = 50,
    mean_cycle_time_sec=4, 
    std_dev_cycle_time_sec=0.75,
):
    expected_dead_time_per_second = min_time_between_events*desired_events_per_1000
    initial_probability = desired_events_per_1000 / (1000-expected_dead_time_per_second)
    df_list =[]
    for trial_idx in range(num_trials):
        
        df = simulate_cycle_data(
            num_cycles=num_cycles,
            mean=mean_cycle_time_sec, 
            std_dev=std_dev_cycle_time_sec
        )
        #df['Angle'] = shift_helper.circular_shift(df['Angle'])
        df['trial'] = trial_idx
        
        event_series_dynamic = generate_biologically_plausible_events_no_ramp(
            df.Angle,
            initial_probability,
            min_samples_between_events=min_time_between_events
        )
        
        df['Saccade_Indicator']=event_series_dynamic
        df_list.append(df)
        
    return pd.concat(df_list)