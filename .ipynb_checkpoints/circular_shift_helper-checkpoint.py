import random
import copy
import MiStatHelpers as mi_helper
def circular_shift(angles):
    """
    Shifts the given list of angles by a random value between 0 and 360 degrees.
    The result is wrapped around the 360-degree boundary.
    
    Args:
        angles (list of float): List of angular values in degrees.
        
    Returns:
        list of float: List of shifted angular values in degrees.
    """
    shift_value = random.uniform(0, 359.99)
    shifted_angles = [(angle + shift_value) % 360 for angle in angles]
    return shifted_angles

def apply_shift_to_dataframe(df, group_col, angle_col):
    """
    Applies circular shift to the angle values within each group in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with at least two columns.
        group_col (str): Name of the column to use for grouping.
        angle_col (str): Name of the column containing angular values to shift.
        
    Returns:
        pd.DataFrame: DataFrame with shifted angle values in the specified column.
    """
    shifted_values = df.groupby(group_col)[angle_col].transform(circular_shift)
    return shifted_values

def shift_perm_distribution(
    data_df,
    group_col,
    col_to_perm,
    event_col, 
    n_bins
):
    """
    data_df (pd.DataFrame):data frame with group_col,col_to_perm,and event_col
    group_col (str): column name for column to group shifted sections
    col_to_perm (str): column to shift
    event_col (str): column name of events to calc modulation index events==1
    n_bins (int): number of bins to create for col_to_perm
    """
    permed_col = col_to_perm + '_shifted'
    data_df[permed_col] = apply_shift_to_dataframe(
        data_df, 
        group_col,
        col_to_perm
    )
    data_df['perm_bins'] = mi_helper.create_bins_for_col(
        data_df,
        permed_col,
        n_bins=6
    )
    permed_event_df = data_df[data_df[event_col]==1]
    prob_dist_perm = permed_event_df['perm_bins'].value_counts() / len(permed_event_df)
    return mi_helper.calc_modulation_index(prob_dist_perm, n_bins)

def test_apply_shift_to_dataframe(df):
    # Apply the function
    df = copy.deepcopy(df)
    shifted_df = apply_shift_to_dataframe(df, 'group', 'angle')
    
    # Test that the output DataFrame has the same length as the input
    assert len(shifted_df) == len(df), f"Expected length {len(df)} but got {len(shifted_df)}"
    
    # Test that the 'other_col' remains unchanged
    assert all(shifted_df['other_col'] == df['other_col']), "Values in 'other_col' changed."
    
    # Test that the shifted values in 'angle' column are between 0 and 360
    for angle in shifted_df['angle']:
        assert 0 <= angle < 360, f"Angle {angle} is out of range [0, 360)"
    return shifted_df