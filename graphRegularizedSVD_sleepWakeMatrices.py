"""
@author: Mindy Ross
python version 3.7.4
"""

# FUNCTIONS TO GET SLEEP/WAKE LABELS BY HOUR FROM BIAFFECT KEYPRESS FILE

import re
import glob
import itertools
import warnings
import numpy as np
import pandas as pd
import numpy as np
from scipy.linalg import svd
from numpy.linalg import cholesky
import scipy.linalg
from scipy.linalg import solve_triangular
from scipy.linalg import inv
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import scale
# from sksparse import cholmod # Cannot get this to work for now
from scipy.sparse import csgraph
from skimage import segmentation
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.autonotebook import tqdm

# sort files numerically
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return(parts)

# calculate typing speed (median IKD)
def median_aaikd(df: pd.DataFrame):
    grpAA = df.loc[((df['keypress_type'] == 'alphanum') &
                    (df['previousKeyType'] == 'alphanum'))]
    # get median IKD
    medAAIKD = np.nanmedian(grpAA['IKD']) if len(grpAA) >= 20 else float('NaN')

    df['']

    return(medAAIKD)

# calculate typing speed (median IKD)
def medianAAIKD(dataframe):
    grpAA = dataframe.loc[((dataframe['keypress_type'] == 'alphanum') &
                                (dataframe['previousKeyType'] == 'alphanum'))]
    # get median IKD
    medAAIKD = np.nanmedian(grpAA['IKD']) if len(grpAA) >= 20 else float('NaN')
    return(medAAIKD)

# Expects hierarchical column index
def _add_missing_hours(df: pd.DataFrame):
    """
    Add columns of zeros for the hours not present in the columns of `df`. Expects a MultiIndex column index.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with days (potentially as a part of a hierarchical index with subject) in the row index, 
        and a hierarchical index of variable (level 0) and hour (level 1) as the column index.

    Returns
    -------
    df: pd.DataFrame
        Same as input but with columns of zeros inserted for all hours in the [0, 23] interval not previously present in the column index.
    """

    present_hours = df.columns.get_level_values('hour').unique()

    # Hours range from 0 up to and including 23
    missing_hours = [h for h in range(24) if h not in present_hours]

    # Variable level (e.g., upright or active)
    var_lvls = df.columns.unique(level=0)

    # Insert 0 for all missing hours
    for h in missing_hours:
        for lvl in var_lvls:
            df[lvl, h] = [0] * len(df.index)

    return df.sort_index(axis='columns')

def pivot_split(df: pd.DataFrame, variables: list[str]):
    """
    Create matrices out of variable columns, split based on subject.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with variable columns that need to be turned into matrices. 
        Requires the columns `subject`, `dayNumber`, `hour`, and those specified by the `variables` parameter.
    variables: list[str]
        List of variable names.

    Returns
    -------
    dats_dict: dict[str, dict[str, pd.DataFrame]]
        Two-level dictionary. First level is indexed by the subject ID, second level by the variable name.
        Contains the day-by-hour matrices as DataFrames.
    """

    # Creates hierarchical column index (variable, hour)
    dat_hour_us = df.unstack(level='hour', fill_value=0) \
        .sort_index() # Unstacking seems to mess with index sorting

    dat_hour_complete = _add_missing_hours(dat_hour_us)

    # Split into a dict of {participant: DataFrame}
    dat_hour_gb = dat_hour_complete.groupby(level='subject')
    dats_hour_us = {g: dat_hour_gb.get_group(g) for g in dat_hour_gb.groups}

    # Split into a dict of {participant: {variable: DataFrame}}
    dats_dict = {id: {var: df[var].droplevel('subject') for var in variables} for id, df in dats_hour_us.items()}

    return dats_dict

def _rank_date(df: pd.DataFrame):
    """
    Rank the date column `date` and put the result in `dayNumber`, allowing for gaps between dates.

    Parameters
    ----------
    df: pd.DataFrame
        A DataFrame with a `date` column containing date objects.

    Returns
    -------
    df: pd.DataFrame
        Same as input but with a `dayNumber` column added that gives the number of days since the earliest date.
    """

    dates = df['date']

    date_diffs = dates - dates.min()

    df_c = df.copy() # Let's not mutate the original
    df_c['dayNumber'] = date_diffs.map(lambda d: d.days)

    return df_c

def rank_dates(df: pd.DataFrame, group_by: str = 'subject'):
    """
    Rank the date column `date` by the column indicated by `group_by` and put the result in `dayNumber`, allowing for gaps between dates.

    Parameters
    ----------
    df: pd.DataFrame
        A DataFrame with a `date` column containing date objects.
    group_by: str
        A string that specifies the column by which to group the ranking.

    Returns
    -------
    df: pd.DataFrame
        Same as input but with a `dayNumber` column added that gives the number of days since the earliest date, for every group. 
    """

    # Don't modify the original DataFrame
    df_c = df.copy()

    df_min_date = df_c.groupby(group_by, group_keys=False)['date'].transform('min')

    df_c['dayNumber'] = (df_c['date'] - df_min_date).dt.days

    return df_c

def _calculate_aaikd(df: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    """
    Calculate the median inter-key delay of alphanumeric-alphanumeric key transitions.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame of BiAffect key presses.
    group_by:
        List of columns to group by.

    Returns
    -------
    df_aaikd: pd.DataFrame
        Long data frame with the group_by columns and the median AAIKD in the `IKD` column.
    """

    df_g = df.loc[(df['keypress_type'] == 'alphanum') & (df['previousKeyType'] == 'alphanum')] \
        .groupby(group_by)

    df_s = df_g.size()

    df_aaikd = df_g \
        .median(numeric_only=True) \
        .loc[df_s >= 20, ['IKD']]
    
    return df_aaikd

def _pre_filter(df: pd.DataFrame, participant_mask: pd.Series | None = None):
    """
    Filter out participants if they do not have enough typing data.

    If `participant_mask` is not provided, a mask will be constructed from the following requirements:
    Participants should have, on average, 1) >= 20% of their daily hours filled with some data,
    2) >= 50 key presses per hour, and 3) >= 9 days of data 
    (the first and last days will be dropped, resulting in at least a week of data).

    Parameters
    ----------
    df: pd.DataFrame
        Long DataFrame with a hierarchical index of subject, day number and hour.
    participant_mask: pd.Series | None
        Optional. If missing, a mask will be constructed using the requirements outlined above.
        Otherwise, it will be applied as-is. Its index should match the unique values of the subject index in `df`.

    Returns
    -------
    dat_masked: pd.DataFrame
        Same as `df`, but with nonsatisfactory participants removed.
    participant_mask: pd.Series
        Given or constructed mask.
    """

    if participant_mask is None:
        # Unstack, add missing hours, and re-stack. Very fast.
        dat_hour_us = df.unstack(level='hour', fill_value=0) \
            .sort_index() # Unstacking seems to mess with index sorting

        dat_hour_complete = _add_missing_hours(dat_hour_us)

        dat_complete = dat_hour_complete.stack().sort_index()

        dat_binary = dat_complete > 0

        dat_kp_daily_activity = dat_binary['n_presses'].groupby(['subject', 'dayNumber']).mean()
        dat_kp_daily_amount = df['n_presses'].groupby(['subject', 'dayNumber']).median()

        avg_daily_activity = dat_kp_daily_activity.groupby('subject').mean()
        avg_daily_amount = dat_kp_daily_amount.groupby('subject').median()

        n_days = df.groupby(['subject', 'dayNumber']).first() \
            .groupby('subject').size()

        participant_mask = (avg_daily_activity >= 0.2) \
            & (avg_daily_amount >= 50) \
            & (n_days >= 9) # First and last day will be dropped

        participant_mask.name = 'mask'

    # Propagate mask values to all entries of every participant
    dat_masked = df.join(participant_mask, on='subject')
    # Subset data based on mask
    dat_masked = dat_masked.loc[dat_masked['mask']]
    # Drop the mask column from the DataFrame
    dat_masked = dat_masked.drop('mask', axis='columns')

    return dat_masked, participant_mask

def _get_days_to_remove(n_presses_df: pd.DataFrame):
    # Trim off the first and last day
    n_presses_df_trimmed = n_presses_df.iloc[1:-1]

    # remove 7-day segments of data if not enough TODO: Allow for gaps between days
    slidingWindowListForRemoval = sliding_window(np.array(n_presses_df_trimmed), window_size=7, gap=1)
    daysToRemove = []
    c = 0
    for w in slidingWindowListForRemoval:
        if len(w) < 7:
            break
        Wbinary = np.where(w > 0, 1, 0)
        avgActivity = Wbinary.mean(axis=1).mean()
        Wkp = np.where(w > 0, w, np.nan)
        avgAmount = np.nanmedian(np.nanmedian(Wkp, axis=1))
        if (avgActivity < 0.2) | (avgAmount < 50):
            daysToRemove.extend(list(range(c, c + 7)))
        c += 1
    
    last_idx = len(n_presses_df.index) - 1

    daysToRemove = np.array([*set(daysToRemove)]) + 1
    daysToRemove = np.concatenate(([0], daysToRemove, [last_idx]))

    return daysToRemove

def _get_days_to_remove2(n_presses_df: pd.DataFrame):
    """
    Return an array of indices of days for which less than 33% of the hours
    (i.e., fewer than eight hours) contains data.

    Parameters
    ----------
    n_presses_df: pd.DataFrame
        Participant-specific data frame in long format containing
        the number of key presses.

    Returns
    -------
    out: np.ndarray
        Array of indices for the days that are to be excluded.
    """
    
    n_presses_binary = n_presses_df > 0
    # n_presses_nan = n_presses_df.where(n_presses_df > 0)
    
    perc_hours_active = n_presses_binary.mean(axis=1)
    # median_n_presses = n_presses_nan.median(axis=1)

    # Percentage is hard-coded for now
    day_mask: pd.Series = perc_hours_active >= 0.33
    
    # dayNumber index is turned into a column
    day_mask_df = day_mask.reset_index(name='included')

    excluded_df = day_mask_df.loc[~day_mask_df['included']]

    return excluded_df.index.values

def _remove_days(participant_dict: dict[str, pd.DataFrame]):
    """
    For all data frames of a specific participant, 
    remove days that don't have enough data.

    Parameters
    ----------
    participant_dict: dict[str, pd.DataFrame]
        Dictionary of modalities (keys) and the corresponding
        day-by-hour data frames.

    Returns
    -------
    cleaned_dict: dict[str, pd.DataFrame]
        Same dictionary as input but with unsatisfactory days removed.
    """
    
    cleaned_dict = {}

    df = participant_dict['n_presses']
    
    # Create contiguous index from 0 to nrow(df), 
    # then check which indices should go and invert
    days_to_remove = _get_days_to_remove2(df)
    day_mask = ~np.isin(np.arange(len(df.index)), days_to_remove)

    cleaned_dict = {var: df.iloc[day_mask, :] for var, df in participant_dict.items()}

    # cleaned_dict = {var: df[~df.index.isin(days_to_remove, level='dayNumber')] for var, df in participant_dict.items()}

    return cleaned_dict

def _post_filter(dats_dict: dict[str, dict[str, pd.DataFrame]]):
    """
    For every participant, remove days that don't have enough data.

    Parameters
    ----------
    dats_dict: dict[str, dict[str, pd.DataFrame]]
        Dictionary of participants (keys) and their modality dictionaries (values),
        which contain the modality strings (keys) and the corresponding day-by-hour
        data frames (values).

    Returns
    -------
    out: dict[str, dict[str, pd.DataFrame]]
        Same dictionary but with unsatisfactory days stripped from the data frames.
    """
    
    return {part: _remove_days(dic) for part, dic in dats_dict.items()}

def get_typing_matrices(dat_kp: pd.DataFrame, dat_ses: pd.DataFrame):
    """
    Set up Biaffect typing matrices for graph
    regularized SVD.

    Parameters
    ----------
    dat_kp: pd.DataFrame
        DataFrame containing the key press level BiAffect data for all participants.
    dat_ses: pd.DataFrame
        DataFrame containing the session level BiAffect data for all participants.

    Returns
    -------
    matrices: dict[str, dict[str, pd.DataFrame]]
        Two-level dictionary. First level is indexed by the subject ID, second level by the variable name.
        Contains the day-by-hour matrices as DataFrames.
    """

    group_by = ['subject', 'dayNumber', 'hour']



    kp_variables = ['IKD', 'n_presses']

    dat_kp_ranked = rank_dates(dat_kp)
    dates = dat_kp_ranked[['subject', 'date', 'dayNumber']].drop_duplicates()

    # Number of key presses, including backspace and autocorrect
    dat_kp_n_presses = dat_kp_ranked.groupby(group_by).size()
    dat_kp_n_presses.name = 'n_presses'

    # _calculate_aaikd filters out some hours, so joining with dat_kp_n_presses will result in NaNs
    dat_kp_vars = _calculate_aaikd(dat_kp_ranked, group_by)
    dat_kp_vars = dat_kp_vars.join(dat_kp_n_presses, how='outer').fillna(0)

    dat_kp_masked, participant_mask = _pre_filter(dat_kp_vars)


    ses_variables = ['active', 'upright']

    dat_ses_ranked = rank_dates(dat_ses)

    dat_ses_ranked_g = dat_ses_ranked.groupby(by=group_by)

    # Some accelerometer measurements are NaNs
    dat_ses_vars = dat_ses_ranked_g[ses_variables].mean().fillna(0)

    dat_ses_masked, _ = _pre_filter(dat_ses_vars, participant_mask)


    dat_masked = dat_kp_masked.join(dat_ses_masked, how='outer').fillna(0)
    dats_dict = pivot_split(dat_masked, kp_variables + ses_variables)


    dats_dict = _post_filter(dats_dict)


    return dats_dict, dates


def get_typingMatrices(df: pd.DataFrame):
    """
    Set up Biaffect typing activity and typing speed matrices for graph
    regularized SVD.

    Parameters
    ----------
    df : pandas dataframe
         preprocessed BiAffect keypress file.

    Returns
    -------
    activityM : pandas dataframe
                BiAffect typing activity by hour of shape (days x hours).
    speedM : pandas dataframe
             BiAffect typing speed by hour of shape (days x hours).
    """

    # get matrix of typing activity by day and hour
    df['hour'] = pd.to_datetime(df['keypressTimestampLocal']).dt.hour
    df['dayNumber'] = df['dayNumber'].astype(int)
    M = df.groupby(['dayNumber','hour'],as_index = False).size().pivot(index='dayNumber', columns='hour').fillna(0)

    # insert hours with no activity across all days
    missingHours = [h for h in range(24) if h not in list(M['size'].columns)]
    M.columns = M.columns.droplevel(0)
    for h in missingHours:
        M.insert(h,h,[0]*M.shape[0])
    M = M.sort_index(ascending=True)

    # Filter users with not enough data
    # find avg number of hours of activity/day
    Mbinary = np.where(M > 0, 1, 0)
    avgActivityPerDay = Mbinary.mean(axis=1).mean()
    # of the days with kp, find median amount
    Mkp = np.where(M > 0, M, np.nan)
    avgAmountPerDay = np.nanmedian(np.nanmedian(Mkp, axis=1))
    # if not enough typing activity for user, skip user
    if (avgActivityPerDay < 0.2) | (avgAmountPerDay < 50):
        return pd.DataFrame(), pd.DataFrame()

    # remove first and last days
    activityM = M.iloc[1:-1]

    # if less than 7 days, skip subject
    if activityM.shape[0] < 7:
        return pd.DataFrame(), pd.DataFrame()

    # remove 7-day segments of data if not enough
    slidingWindowListForRemoval = sliding_window(np.array(activityM), window_size=7, gap=1)
    daysToRemove = []
    c = 1 # LK: Changed this from 0 to allow for correct indexing
    for w in slidingWindowListForRemoval:
        if len(w) < 7:
            break
        Wbinary = np.where(w > 0, 1, 0)
        avgActivity = Wbinary.mean(axis=1).mean()
        Wkp = np.where(w > 0, w, np.nan)
        avgAmount = np.nanmedian(np.nanmedian(Wkp, axis=1))
        if (avgActivity < 0.2) | (avgAmount < 50):
            daysToRemove.extend(list(range(c, c + 7)))
        c += 1
    # remove rows corresponding to indices in daysToRemove
    activityM = activityM[~activityM.index.isin([*set(daysToRemove)])] # Should this be loc instead?

    # if less than 7 days, skip subject
    if activityM.shape[0] < 7:
        return pd.DataFrame(), pd.DataFrame()

    # get matrix of typing speed by hour
    speedM=df.groupby(['dayNumber','hour'],as_index = False).apply(lambda x: medianAAIKD(x)).pivot(index='dayNumber', columns='hour')
    speedM.columns = speedM.columns.droplevel(0)
    for h in missingHours:
        speedM.insert(h,h,[np.nan]*speedM.shape[0])
    speedM = speedM.sort_index(ascending=True)

    # remove first and last days
    speedM = speedM.iloc[1:-1]
    # remove rows corresponding to indices in daysToRemove
    speedM = speedM[~speedM.index.isin([*set(daysToRemove)])]
    speedM = speedM.replace(np.nan, 0)

    return activityM, speedM

def cosine_similarity(a,b):
    import numpy as np
    from numpy.linalg import norm
    cosine = np.dot(a,b)/(norm(a)*norm(b))
    return cosine

# adjacency matrix weight between consecutive days
def day_weight(dAround):
    return np.nanmedian(dAround)

# adjacency matrix weight between consecutive hours
def hour_weight(h1,h2):
    return (h1+h2)/2

# calculate weighted adjacency matrix for graph regulated SVD
def weighted_adjacency_matrix(mat):
    # days = rows
    # hours = columns
    W = np.zeros((mat.size, mat.size))
    for i in range(mat.size):
        for j in range(mat.size):
            # iterate across hours of each day then across days
            # d1h1, d1h2, d1h3, d1h4...d2h1, d2h2, d3h3...
            i_Mi = i//mat.shape[1]
            i_Mj = i%mat.shape[1]
            j_Mi = j//mat.shape[1]
            j_Mj = j%mat.shape[1]
            # diagonals
            if i == j:
                W[i,j] = 0
            # if abs(subtraction of col indices) == 1 & subtraction of row indices == 0:
            elif (abs(j_Mj-i_Mj) == 1) & ((j_Mi-i_Mi) == 0):
                W[i,j] = hour_weight(mat[i_Mi,i_Mj],mat[j_Mi,j_Mj])
            # if abs(subtraction of row indices) == 1 & subtraction of col indices == 0:
            elif (abs(j_Mi-i_Mi) == 1) & ((j_Mj-i_Mj) == 0):
                if i_Mi <= 3:
                    W[i,j] = day_weight(mat[0:7,i_Mj])
                elif i_Mi >= mat.shape[0]-4:
                    W[i,j] = day_weight(mat[-8:-1,i_Mj])
                else:
                    W[i,j] = day_weight(mat[i_Mi-3:i_Mi+4,i_Mj])
            # connect 23hr with 00hr
            elif (i_Mj == mat.shape[1]-1) & ((j_Mi-i_Mi) == 1) & (j_Mj == 0):
                W[i,j] = hour_weight(mat[i_Mi,i_Mj],mat[i_Mi+1,0])
            else:
                W[i,j] = 0
    return W

# Helper function for construct_w_binary
def _set_weight(W: np.ndarray, idx1: int, idx2: int):
    W[idx1, idx2] = 1

def construct_w_binary(dat_df: pd.DataFrame):
    """
    Constructs a binary adjacency matrix based on the row and column indices of the given DataFrame.

    Every hour of every day is regarded as a node in a network. Every hour is connected to the previous and next one, 
    potentially crossing the day boundary. Hours are also connected to their corresponding hours in the previous and next day.

    Parameters
    ----------
    dat_df: pd.DataFrame
        DataFrame that should match the original data matrix in terms of labels and size. 
        Its index will be used to extract the days, its columns to extract the hours.

    Returns
    -------
    W_binary: np.ndarray
        A binary, symmetric adjacency matrix indicating connectedness between nodes by 1. It is 0 everywhere else.
    """

    # Flat numpy arrays
    days = dat_df.index.values
    hours = dat_df.columns.values

    W_binary = np.zeros((dat_df.size, dat_df.size))
    node1_idx = 0

    days_hours = list(itertools.product(days, hours))

    for node1_day, node1_hour in days_hours:
        # Adjacency matrix is symmetric, we only need to consider upper triangle
        node2_idx = node1_idx + 1

        for node2_day, node2_hour in days_hours[node2_idx:]:
            day_diff = node2_day - node1_day
            hour_diff = node2_hour - node1_hour

            # Connect hours within same day
            if day_diff == 0 and abs(hour_diff) == 1:
                _set_weight(W_binary, node1_idx, node2_idx)

            # Connect final and first hour of this and next day, respectively
            # If day_diff is positive, hour_diff should be negative and vice versa
            if abs(day_diff) == 1 and abs(hour_diff) == 23 and np.sign(day_diff) != np.sign(hour_diff):
                _set_weight(W_binary, node1_idx, node2_idx)

            # Connect hour to the same hours of the previous and next day
            if abs(day_diff) == 1 and hour_diff == 0:
                _set_weight(W_binary, node1_idx, node2_idx)

            node2_idx += 1

        node1_idx += 1

    W_binary = W_binary + W_binary.T

    return W_binary

def regularized_svd(X, B, rank, alpha, as_sparse=False):
    """
    Perform graph regularized SVD as defined in
    Vidar & Alvindia (2013).

    Parameters
    ----------
    X : numpy array
        m x n data matrix.
    B : numpy array
        n x n graph Laplacian of nearest neighborhood graph of data.
    W : numpy array
        n x n weighted adjacency matrix.
    rank : int
        Rank of matrix to approximate.
    alpha : float
        Scaling factor.
    as_sparse : bool
        If True, use sparse matrix operations. Default is False.
    
    Returns
    -------
    H_star : numpy array
        m x r matrix (Eq 15).

    W_star : numpy array
        r x n matrix (Eq 15).
    """

    I = np.eye(B.shape[0])
    C = I + (alpha * B)
    D = cholesky(C)
    E, S, Fh = svd(X @ inv(D.T), full_matrices=False)
    E_tilde = E[:, :rank]  # rank-r approximation; H_star = E_tilde (Eq 15)
    H_star = E_tilde  # Eq 15
    S_tilde = S[:rank]
    Fh_tilde = Fh[:rank,:]
    W_star = np.diag(S_tilde) @ Fh_tilde @ inv(D) #E_tilde.T @ X @ inv(C)  # Eq 15
    return H_star, W_star

def regularized_svd_chol(
        X: np.ndarray, 
        B: np.ndarray, 
        rank: int, 
        alpha: float
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform graph regularized SVD as defined in
    Vidar & Alvindia (2013), with correction, using the Cholesky decomposition.

    Parameters
    ----------
    X : numpy array
        m x n data matrix, where m is the dimensionality of a single data point and n the number of data points.
    B : numpy array
        n x n graph Laplacian of nearest neighborhood graph of data.
    rank : int
        Rank of matrix to approximate.
    alpha : float
        Scaling factor.
    
    Returns
    -------
    H_hat : numpy array
        m x r matrix (Eq 15).

    W_hat : numpy array
        r x n matrix (Eq 15).
    """

    # Eq 11
    I = np.eye(B.shape[0])
    C = I + (alpha * B)
    L = scipy.linalg.cholesky(C, lower=True)
    Y = solve_triangular(L, X.T, lower=True).T

    U, sigma, Vh = svd(Y, full_matrices=False) # Singular values returned in descending order
    U_tilde = U[:, :rank]  # rank-r approximation
    sigma = sigma[:rank] # Vector
    V_tilde = Vh[:rank, :].T

    H_hat = U_tilde 
    W_hat = solve_triangular(L.T, V_tilde @ np.diag(sigma)).T

    return H_hat, W_hat

def regularized_svd_chol_old( 
        X: np.ndarray, 
        B: np.ndarray, 
        rank: int, 
        alpha: float
    ) -> tuple[np.ndarray, np.ndarray]:

    # Eq 11
    I = np.eye(B.shape[0])
    C = I + (alpha * B)
    L = scipy.linalg.cholesky(C, lower=True)
    Y = solve_triangular(L, X.T, lower=True).T

    U, _, _ = svd(Y, full_matrices=False) # Singular values returned in descending order
    U_tilde = U[:, :rank]  # rank-r approximation

    H_hat = U_tilde 
    W_hat = solve_triangular(L.T, Y.T @ U_tilde).T 

    return H_hat, W_hat

def regularized_svd_test(
        X: np.ndarray, 
        B: np.ndarray,
        alpha: float,
        U_tilde: np.ndarray
    ) -> np.ndarray:
    """
    Apply H_hat = U_tilde from a previous graph-regularised SVD to a new data and
    Laplacian matrix.

    Parameters
    ----------
    X : numpy array
        m x n test data matrix, where m is the dimensionality of a single data point and n the number of data points.
        m should match the number of rows in the training data matrix.
    B : numpy array
        n x n graph Laplacian of nearest neighborhood graph of data.
    alpha : float
        Regularisation parameter. Higher values mean more regularisation.
    U_tilde : numpy array
        m x r matrix of left-singular vectors of the original graph-regularised SVD,
        where r is the imposed rank of the decomposition. Equivalent to H_hat.
    
    Returns
    -------
    W_hat : numpy array
        r x n matrix constructed by rotating the data to a new basis using U_tilde.
    """

    I = np.eye(B.shape[0])
    C = I + (alpha * B)
    L = scipy.linalg.cholesky(C, lower=True)

    # Same space as Sigma_tilde @ Vh_tilde
    Y = scipy.linalg.solve_triangular(L, X.T @ U_tilde, lower=True).T

    W_hat = scipy.linalg.solve_triangular(L.T, Y.T).T

    return W_hat
    

def calculate_svd(
        typing_dfs: dict[str, dict[str, pd.DataFrame]], 
        modalities: str | list[str] = 'all',
        rank: int = 1,
        alpha: float = 1,
        train_ratio: float = 1
    ) -> tuple[dict[str, np.ndarray] | dict[str, dict[str, np.ndarray]], dict[str, int]]:
    """
    Calculates a graph-regularised SVD for different participants based on their hourly typing characteristics.
    
    Parameters
    ----------
    typing_dfs: dict[str, dict[str, pd.DataFrame]]
        Two-level dictionary with participant ID at the first level and modality name at the second level.
        Values should be day-by-hour DataFrames.
    modalities: str | list[str]
        If a list of strings, will be used as a (non-strict) subset of the available data modalities in `typing_dfs` 
        which will be used for the SVD. If 'all', will be used to select all of the available modalities.
    rank: int
        Rank of approximation.
    alpha: float
        Regularisation parameter. Higher values mean more regularisation.
    train_ratio: float
        Proportion of data (in the interval [0, 1]) that should be allocated to the training set. The remainder
        will go to the test set. If no testing is to be performed, set to 1.
        
    Returns
    -------
    svd_mats: dict[str, np.ndarray] | dict[str, dict[str, np.ndarray]]
        The first component of the graph-regularised SVD, indexed by participant IDs (first level)
        and potentially the 'train' and 'test' keys (second level).
    split_indices: dict[str, int]
        Day index where the train-test split occurred. Keys are the participant IDs.

    Raises
    ------
    ValueError
        If there is no data in the training set.
    """

    svd_mats = dict()
    split_indices = dict()

    if modalities == 'all':
        first_sub = next(iter(typing_dfs))
        modalities = list(typing_dfs[first_sub].keys())

    train_ratio = max(0, min(1, train_ratio))

    for subject, dfs in tqdm(typing_dfs.items()):
        # Subset the DataFrames to use, convert to 1D ndarray
        dfs_svd = {modality: dfs[modality] for modality in modalities}
        dats_svd = {modality: df.to_numpy().flatten() for modality, df in dfs_svd.items()}

        # Transform to constrain range
        if 'n_presses' in dats_svd:
            dats_svd['n_presses'] = np.log1p(dats_svd['n_presses'])

        # W is constructed by looking at the index values, 
        # so actual matrix values are irrelevant; just use the first one
        # TODO: Optimize construction by splitting for test and train data
        W_binary = construct_w_binary(dfs_svd[modalities[0]])

        dat_svd = np.vstack(list(dats_svd.values()))

        # Split data into training and testing set
        split_idx = 24 * int(dat_svd.shape[1] / 24 * train_ratio)
        dat_svd_train = dat_svd[:, :split_idx]
        dat_svd_test = dat_svd[:, split_idx:] # Empty if train_ratio is too high

        if dat_svd_train.shape[1] == 0:
            # raise ValueError(f"No data in training set for participant {subject}")
            warnings.warn(f"No data in training set for participant {subject}")
            continue

        # Scale expects a matrix of (n_samples, n_features)
        dat_svd_train = scale(dat_svd_train.T, with_mean=False).T

        # Calculate SVD
        lapl = csgraph.laplacian(W_binary[:split_idx, :split_idx])
        U_tilde, W_hat = regularized_svd_chol(dat_svd_train, lapl, rank, alpha)
        # Get SVD matrix
        svd_mat_train = W_hat.reshape((-1, 24))

        # Sometimes the matrix comes out all negative
        if svd_mat_train.max() <= 0.00000001:
            svd_mat_train = svd_mat_train * -1

        # Check whether test set is not empty
        if dat_svd_test.shape[1] > 0:
            dat_svd_test = scale(dat_svd_test.T, with_mean=False).T

            # Calculate SVD for test cases
            lapl = csgraph.laplacian(W_binary[split_idx:, split_idx:])

            W_hat_test = regularized_svd_test(dat_svd_test, lapl, alpha, U_tilde)
            svd_mat_test = W_hat_test.reshape((-1, 24))

            if svd_mat_test.max() <= 0.00000001:
                svd_mat_test = svd_mat_test * -1

            svd_mats[subject] = {
                'train': svd_mat_train,
                'test': svd_mat_test
            }
        else:
            svd_mats[subject] = svd_mat_train

        split_indices[subject] = split_idx // 24

    return svd_mats, split_indices

def get_SVD(activityM, speedM):
    """
    Apply graph regularized SVD as defined in
    Vidar & Alvindia (2013) to typing data.
    
    Parameters
    ----------
    activityM : pandas dataframe
                BiAffect typing activity by hour of shape (days x hours).
    speedM : pandas dataframe
             BiAffect typing speed by hour of shape (days x hours).
    
    Returns
    -------
    svdM : numpy array
           graph regularized SVD of typing features matrix of shape (days x hours).
    """

    # normalize nKP matrix
    activityM = activityM / activityM.sum().sum()
    # SVD
    # get adjacency matrix for SVD
    W = weighted_adjacency_matrix(np.array(activityM))
    # normalize keypress values
    normKP = np.array(activityM).flatten()
    ikd_vals = np.array(speedM).flatten()
    data = np.vstack((normKP,ikd_vals))
    # get graph laplacian
    B = csgraph.laplacian(W)
    # get graph normalized SVD
    H_star, W_star = regularized_svd(data, B, rank=1, alpha=100, as_sparse=False)
    # get SVD matrix
    svdM = W_star.reshape(activityM.shape)
    if svdM.max() <= 0.00000001:
        svdM = svdM * -1
    return svdM

# create list of indices for each sliding window
def sliding_window(elements, window_size, gap):
    """
    Create a list of the elements (not indices) returned by a sliding window.

    Parameters
    ----------
    elements: list[Any] | np.ndarray
        Something indexable from which the sliding window will return values. 
        If a 2D numpy array, the sliding window is applied to the first axis (the rows). 
        In other words, all columns will be returned for every slice.
    window_size: int
        Size of the window sliding across (the first axis of) elements.
    gap: int
        Gap with which the window jumps while sliding.

    Returns
    -------
    windows: list[list[Any]] | list[np.ndarray]
        A list of lists/arrays, one for each window. The window lists/arrays are of size `window_size`
        and contain the values from `elements` corresponding to the specific window.
        Note that the lists/arrays can be nested or >1D.
    """

    if len(elements) <= window_size:
       return elements
    windows = []
    ls = np.arange(0, len(elements), gap)
    for i in ls:
        windows.append(elements[i:i+window_size])
    return windows

def get_sleepWakeLabels(svd_mat):
    """
    Get sleep/wake labels per hour of BiAffect typing data.
   
    Parameters
    ----------
    df : numpy array
         graph regularized SVD of typing features matrix of shape (days x hours).
    
    Returns
    -------
    sleepWakeMatrix : numpy array
                      sleep/wake labels per hour of BiAffect typing data 
                      of shape (days x hours).
    """

    # Binarize SVD
    binarizedSVD = np.where(svd_mat <= 0.00000001, 0,1)
    # sleep/wake labels from binarized SVD matrix
    sleep_label = 0
    wake_label = 1
    # flood fill main sleep component
    # initiate matrix filled with value 2 (meaning no label yet)
    floodFillM = np.full(shape=svd_mat.shape, fill_value=2, dtype='int')
    for r in range(svd_mat.shape[0]):
        # get row
        row = binarizedSVD[r]
        # if entire row is sleep, continue
        if ((row == np.array([sleep_label]*len(row))).all()) == True:
            floodFillM[r] = [sleep_label]*len(row)
            continue
        # if wake labels only during hours 2-15, get min from 0-24 hr range
        if ((binarizedSVD[r, 2:15] == np.array([wake_label]*13)).all()) == True:
            idx_rowMin = np.argmin(svd_mat[r])
        else: # else limit min hour to between hr 2-15
            idx_rowMin = np.argmin(svd_mat[r, 2:15]) + 2
        # if min value not equal to sleep_label, then no sleep that day. continue
        if binarizedSVD[r,idx_rowMin] != sleep_label:
            floodFillM[r] = [wake_label]*len(row)
            continue
        # flood fill 
        sleep_flood = segmentation.flood(binarizedSVD, (r,idx_rowMin))#NEED DIAG CONNECTIVITY #connectivity=1)
        # replace output matrix row with flood fill values
        floodFillM[r] = np.invert(sleep_flood[r])

        # add sleep label before midnight if exists
        # if iteration at last row
        if r == svd_mat.shape[0]-1:
            # if last cell is sleep label
            if (row[-1] == sleep_label):
                i = 0
                # find earliest index of sleep label for that row prior to midnight
                while row[23-i] == sleep_label:
                    end_idx = i
                    i += 1
                # replace identified ending cells with sleep label
                floodFillM[r,(23-end_idx):] = sleep_label
        # if interation not at last row
        else:
            # if last cell is sleep label and first cell of next row is sleep label
            if (row[-1] == sleep_label) & (binarizedSVD[r+1,0] == sleep_label):
                i = 0
                # find earliest index of sleep label for that row prior to midnight
                while row[23-i] == sleep_label:
                    end_idx = i
                    i += 1
                # replace identified ending cells with sleep label
                floodFillM[r,(23-end_idx):] = sleep_label

    # fill in gaps in sleep component
    # get list of sliding windows to remove gaps in sleep
    window_size = 8
    hr_space = 2
    floodFillFlatten = floodFillM.flatten()
    slidingWindowList = sliding_window(floodFillFlatten, window_size, hr_space)

    # iterate through windows and fill gap between first and last sleep index in window
    windowCount = 0
    for window in slidingWindowList:
        try:
            # get index of first sleep label in window
            firstSleep_idx = next(i for i,v in enumerate(window) if v == sleep_label)
        # if no sleep label within window, continue
        except StopIteration:
            windowCount += 1
            continue
        # get index of last sleep label in window
        lastSleep_idx = len(window) - next(i for i, val in enumerate(reversed(window), 1) if val != wake_label)
        # get index of first label in window from array of all labels
        grpIdx0 = windowCount * hr_space
        # replace window values with sleep labels within identified indices
        floodFillFlatten[(grpIdx0+firstSleep_idx):(grpIdx0+lastSleep_idx)] = [sleep_label]*(lastSleep_idx-firstSleep_idx)
        windowCount += 1
    # reshape array to be matrix of sleep/wake activity
    sleepWakeMatrix = floodFillFlatten.reshape(binarizedSVD.shape)
    return sleepWakeMatrix

def plot_heatmaps(activityM, speedM, svdM, sleepWakeMatrix):
    """
    Get heatmaps of steps in process to label BiAffect typing data as sleep/wake.
    
    Parameters
    ----------
    activityM : numpy array
                BiAffect typing activity by hour of shape (days x hours).
    speedM : numpy array
             BiAffect typing speed by hour of shape (days x hours).
    svdM : numpy array
           graph regularized SVD of typing features matrix of shape (days x hours).
    sleepWakeMatrix : numpy array
                      sleep/wake labels per hour of BiAffect typing data 
                      of shape (days x hours).
    
    Returns
    -------
    f : 2 x 2 matplotlib figure
        heatmaps of steps to label BiAffect typing data
    """

    # Visualize heatmap of steps
    f, ax = plt.subplots(nrows=2,ncols=2, sharex=False, sharey=True,
                        figsize=(10,10), facecolor='w')
    # PLOT 1
    sns.heatmap(activityM, cmap='viridis', ax=ax[0,0], vmin=0, vmax=500,
                cbar_kws={'label': '# Keypresses', 'fraction': 0.043})
    # PLOT 2
    sns.heatmap(speedM, cmap='viridis', ax=ax[0,1], vmin=0, vmax=0.3,
                cbar_kws={'label': 'Median IKD', 'fraction': 0.043})
    # PLOT 3
    sns.heatmap(svdM, cmap='viridis', ax=ax[1,0], vmin=0,vmax=0.25,
                cbar_kws={'fraction': 0.043})
    # PLOT 4
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom',
        colors=['#de8f05', '#0173b2'], N=2)
    sns.heatmap(sleepWakeMatrix, ax=ax[1,1], cmap=cmap,
                cbar_kws={'fraction': 0.043})
    colorbar = ax[1,1].collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels(['0', '1'])
    colorbar.set_label('Cluster')
    ax[0,0].set(title='Input Typing Activity', xlabel='Hour', ylabel='Day')
    ax[0,1].set(title='Input Typing Speed', xlabel='Hour', ylabel='Day')
    ax[1,0].set(title='Graph Regularized SVD', xlabel='Hour', ylabel='Day')    
    ax[1,1].set(title='Sleep/Wake Labels', xlabel='Hour', ylabel='Day')
    f.tight_layout()
    return f

def plot_input_heatmaps(activityM: pd.DataFrame | np.ndarray, speedM: pd.DataFrame | np.ndarray):
    """
    Get heatmaps of steps in process to label BiAffect typing data as sleep/wake.

    Parameters
    ----------
    activityM : numpy array | pandas DataFrame
                BiAffect typing activity by hour of shape (days x hours).
    speedM : numpy array | pandas DataFrame
             BiAffect typing speed by hour of shape (days x hours).
    
    Returns
    -------
    f : 2 x 2 matplotlib figure
        heatmaps of steps to label BiAffect typing data
    """
    
    plt.rcParams.update({'font.size': 32})
    # Visualize heatmap of steps
    fig, ax = plt.subplots(nrows=2,ncols=1, sharex=False, sharey=False,
                        figsize=(11,20), facecolor='w')
    # PLOT 1
    sns.heatmap(activityM, cmap='viridis', ax=ax[0], vmin=0, vmax=500,
                cbar_kws={'label': '# Keypresses', 'fraction': 0.043})
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    # PLOT 2
    sns.heatmap(speedM, cmap='viridis', ax=ax[1], vmin=0, vmax=0.3,
                cbar_kws={'label': 'Median IKD (s)', 'fraction': 0.043})
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    ax[0].set(title='Input Typing Activity', xlabel='Hour', ylabel='Day')
    ax[1].set(title='Input Typing Speed', xlabel='Hour', ylabel='Day')
    fig.tight_layout()
    return fig

def plot_svdHeatmap(svdM):
    """
    Get heatmaps of steps in process to label BiAffect typing data as sleep/wake.

    Parameters
    ----------
    svdM : numpy array
           graph regularized SVD of typing features matrix of shape (days x hours).
    
    Returns
    -------
    f : matplotlib figure
        heatmap of SVD matrix
    """
    
    # Visualize heatmap of steps
    plt.rcParams.update({'font.size': 32})
    f, ax = plt.subplots(nrows=1,ncols=1,figsize=(11,10), facecolor='w')
    # PLOT 3
    sns.heatmap(svdM, cmap='viridis', vmin=0,vmax=0.25,
                cbar_kws={'fraction': 0.043})
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    ax.set(title='Graph Regularized SVD', xlabel='Hour', ylabel='Day')    
    f.tight_layout()
    return f

def plot_sleepWakeHeatmap(sleepWakeMatrix):
    """
    Get heatmaps of steps in process to label BiAffect typing data as sleep/wake.

    Parameters
    ----------
    sleepWakeMatrix : numpy array
                      sleep/wake labels per hour of BiAffect typing data 
                      of shape (days x hours).
    
    Returns
    -------
    f : matplotlib figure
        heatmap of sleep/wake matrix
    """

    # Visualize heatmap of steps
    plt.rcParams.update({'font.size': 32})
    f, ax = plt.subplots(nrows=1,ncols=1,figsize=(11,10), facecolor='w')
    # PLOT 4
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom',
                colors=['#de8f05', '#0173b2'], N=2)
    sns.heatmap(sleepWakeMatrix, cmap=cmap,cbar_kws={'fraction': 0.043})
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels(['Sleep', 'Wake'])
    colorbar.ax.tick_params(rotation=90)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    ax.set(title='Sleep/Wake Labels', xlabel='Hour', ylabel='Day')
    f.tight_layout()
    return f

################################################################################################

# Only run if the code is not imported as a module
if __name__ == '__main__':
    # file path of BiAffect keypress files
    pathIn = '/' # insert file path of keypress files
    pathFig = '/'

    from pyarrow import parquet

    # get list of keypress files in file path
    all_files = sorted(glob.glob(pathIn + "*.csv"), key = numericalSort)
    file_type = 'csv'
    if len(all_files) == 0:
        file_type = 'parquet'
        all_files = sorted(glob.glob(pathIn + "*.parquet"), key = numericalSort)

    # loop through keypress files
    for file in all_files:
        # read in keypress file
        if file_type == 'csv':
            dfKP = pd.read_csv(file, index_col=0)
        else:
            dfKP = pd.read_parquet(file, engine='pyarrow')
        # get userID
        user = int(dfKP['userID'].unique())
        print('user: {}'.format(user))

        dfKP['date'] = pd.to_datetime(dfKP['keypressTimestampLocal']) \
            .map(lambda x: x.date())
        dfKP['dayNumber'] = dfKP['date'].rank(method='dense')

        ################################################################
        # FIND SLEEP/WAKE LABELS FROM BIAFFECT KEYPRESS DATA FILE
        ################################################################
        # STEP 1
        # get input matrices of shape days x hours for typing activity (nKP) and speed (median IKD)
        ## matrices may have missing days
        ## check index here to identify day number since first date of typing data
        Mactivity, Mspeed = get_typingMatrices(dfKP)
        # if not enough data in keypress file, skip to next subject
        if len(Mactivity) == 0:
            continue
        
        # STEP 2
        # get graph regularized SVD
        svdMatrix = get_SVD(Mactivity, Mspeed)
        
        # STEP 3
        # get sleep/wake labels by hour
        sleepMatrix = get_sleepWakeLabels(svdMatrix)
        
        # # Plot steps if desired
        f=plot_heatmaps(Mactivity, Mspeed, svdMatrix, sleepMatrix)
        # f.savefig(pathFig + 'user_{}.png'.format(user))

        # Plot input matrices
        inputs = plot_input_heatmaps(Mactivity,Mspeed)
        # inputs.savefig(pathFig + 'user_{}_SVD_inputMatrices-vertical.png'.format(user))

        # # Plot SVD matrix
        svdPlot = plot_svdHeatmap(svdMatrix)
        # svdPlot.savefig(pathFig + 'user_{}_SVD_SVDmatrix.png'.format(user))

        # # Plot sleep/wake matrix
        sleepPlot = plot_sleepWakeHeatmap(sleepMatrix)
        # sleepPlot.savefig(pathFig + 'user_{}_SVD_sleepLabels.png'.format(user))

    print('finish')
