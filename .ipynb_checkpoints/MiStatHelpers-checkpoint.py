import numpy as np
import pandas as pd
from tqdm import tqdm

def process_cycle_data(input_df: pd.DataFrame, 
                       angle_col: str,
                       cycle_start_angle_max: float,
                       cycle_end_angle_min: float,
                       diff_threshold: float = 2.0) -> pd.DataFrame:
    """
    Process a DataFrame to compute cycle start and end angles based on the given angle column.

    Parameters:
    - input_df (pd.DataFrame): Input DataFrame containing the angle data.
    - angle_col (str): Column name containing the angle data.
    - diff_threshold (float): Threshold for considering an increase as a big increase.

    Returns:
    - pd.DataFrame: Processed DataFrame with added columns for cycle start and end angles.
    """
    
    # Ensure that the angle column exists in the input DataFrame
    if angle_col not in input_df:
        raise ValueError(f"The column '{angle_col}' does not exist in the input DataFrame.")

    df = input_df.reset_index()
    
    # Calculate the differences between consecutive rows
    diffs = df[angle_col].diff()

    # Identify decreases and big increases
    decreases = diffs < 0
    big_increases = diffs > diff_threshold

    # Combine the two conditions
    all_big_diffs = decreases | big_increases

    cycle_start_col, cycle_end_vals = np.empty_like(df[angle_col]), np.empty_like(df[angle_col])
    cycle_start_col[:] = np.nan
    cycle_end_vals[:] = np.nan

    idx_of_cycle_starts = df[all_big_diffs].index
    cycle_start_col[idx_of_cycle_starts] = df.iloc[idx_of_cycle_starts][angle_col]
    idx_cycle_ends = idx_of_cycle_starts - 1
    cycle_end_vals[idx_cycle_ends] = df.iloc[idx_cycle_ends][angle_col]

    df['cycle_start_angles'] = cycle_start_col
    df['cycle_start_angles'].ffill(inplace=True)
    df['cycle_end_angles'] = cycle_end_vals
    df['cycle_end_angles'].bfill(inplace=True)
    
    #filter incomplete cycles
    df = df.loc[
        (df['cycle_end_angles']>cycle_end_angle_min) &
        (df['cycle_start_angles']<cycle_start_angle_max)
    ]

    return df

def calc_and_save_mi_dist(    
    bins_col,
    event_col,
    angles_col,
    modality,
    df,
    save_string,
    par,
    n_iterations=1000,
    n_bins = 6
):
    #create bins and enforce uniformity
    df[bins_col] = create_bins_for_col(df,angles_col,n_bins=n_bins)
    min_count = df[bins_col].value_counts().min()
    balanced_df = df.groupby(bins_col).apply(lambda x: x.sample(min_count)).reset_index(drop=True)
    df = balanced_df
    
    #get section of data we're interested in
    trial_df = df.copy()
    event_df = trial_df[trial_df[event_col]==1]#this the event of interest i.e. - saccade
    
    #calc boot strapped mi
    real_mod_index_list = create_bootstrapped_test_stat(event_df,bins_col,n_bins=n_bins)

    null_mi_dist = create_permute_null_distribution(
        trial_df,#create new bins based on all the trials - but this doesn't quite seem right
        angles_col,
        event_col,
        n_bins=n_bins,
        n_iterations=n_iterations
    )
    
    null_save = save_string.format(
        modality,
        modality,
        event_col,
        'null',
        par
    )
    np.savetxt(
        null_save,
        null_mi_dist
    )
    
    real_save = save_string.format(
        modality,
        modality,
        event_col,
        'real',
        par
    )
    np.savetxt(
        real_save,
        real_mod_index_list
    )

def calc_modulation_index(dist_probs,k):
    '''
    calculates modulation index from tort et al. 2010
    dist_probs:probability of event falling in a certain bin
    k:number of bins
    '''
    neg_entropy = np.sum(dist_probs*np.log(dist_probs))
    log_k = np.log(k)
    return (log_k + neg_entropy)/log_k
def create_bins_for_col(df_to_bin, col, n_bins=10, min_val=0, max_val=360):
    '''
    add bins for column
    df: dataframe with col
    col(string): string name of column
    n_bins(optional:int):how many bins you want to make
    min(optional:int): minimum for cuts to make bins default 0
    max(optional:int):maximum for cuts to make bins default 360
    '''
    bin_labels = list(range(n_bins))
    bins = np.linspace(min_val,max_val,n_bins+1)
    return pd.cut(df_to_bin[col].values, bins, labels=bin_labels)

def create_permuted_data(perm_df,col_to_perm,n_bins):
    '''
    create a new permutted column
    '''
    # why am I doing this instead of permutting the bins?
    # for some reason this seems more thorough (underlying signal)
    # permutes where the angles are in reference to events - but overall bin distribution is maintained
    perm_df.loc[:,'perm_angles']=np.random.permutation(perm_df[col_to_perm])
    perm_df.loc[:,'perm_bins'] = create_bins_for_col(perm_df,
                                               'perm_angles',
                                               n_bins=n_bins)
    return perm_df
def create_bootstrapped_test_stat(
        event_df,
        bins_col,
        boot_strap_frac = .8,
        n_boots = 1000,
        n_bins=10,
):
    '''
    bootstrap modulation index
    '''
    real_mod_index_list = []
    for bootstrap_idx in range(n_boots):
        bootstrap_df = event_df.sample(frac=boot_strap_frac)
        prob_dist_real = (bootstrap_df[bins_col].value_counts())/len(bootstrap_df)
        real_mod_index_list.append(calc_modulation_index(prob_dist_real,n_bins))
    return real_mod_index_list


def create_permute_null_distribution(
    data_df,
    col_to_perm,
    event_col,
    n_bins,
    n_iterations=1000,
    tqdm_disabled=False
):
    '''

    '''
    mi_dist_perm = []
    for i in tqdm(range(n_iterations),disable=tqdm_disabled):
        permed_df=create_permuted_data(data_df,col_to_perm,n_bins)
        permed_event_df = permed_df[permed_df[event_col]==1]
        prob_dist_perm = permed_event_df['perm_bins'].value_counts()/len(permed_event_df)
        mi_dist_perm.append(calc_modulation_index(prob_dist_perm,n_bins))
    return mi_dist_perm