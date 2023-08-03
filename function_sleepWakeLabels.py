#%%
# FUNCTIONS TO GET SLEEP/WAKE LABELS BY HOUR FROM BIAFFECT KEYPRESS FILE

import re
import glob
import numpy as np
import pandas as pd
import numpy as np
from scipy.linalg import svd
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular
from scipy.linalg import inv
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd
# from sksparse import cholmod # Cannot get this to work for now
from scipy.sparse import csgraph
from skimage import segmentation
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# sort files numerically
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return(parts)

# calculate typing speed (median IKD)
def medianAAIKD(dataframe):
    grpAA = dataframe.loc[((dataframe['keypress_type'] == 'alphanum') &
                                (dataframe['previousKeyType'] == 'alphanum'))]
    # get median IKD
    medAAIKD = np.nanmedian(grpAA['IKD']) if len(grpAA) >= 20 else float('NaN')
    return(medAAIKD)

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
    slidingWindowListForRemoval = sliding_window(activityM, window_size=7, gap=1)
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

# adjacency matrix weight between consecutive days
def day_weight(d1,d2):
    return (d1+d2)

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
                W[i,j] = day_weight(mat[i_Mi,i_Mj],mat[j_Mi,j_Mj])
            # connect 23hr with 00hr
            elif (i_Mj == mat.shape[1]-1) & ((j_Mi-i_Mi) == 1) & (j_Mj == 0):
                W[i,j] = hour_weight(mat[i_Mi,i_Mj],mat[i_Mi+1,0])
            else:
                W[i,j] = 0
    return W

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

    if as_sparse:
        # Use sparse matrix operations to reduce memory
        I = sp.lil_matrix(B.shape)
        I.setdiag(1)
        C = I + (alpha * B)
        print('Computing Cholesky decomposition')

        # Currently cannot get scikit-sparse to work. TODO: Fix scikit-sparse
        # factor = cholmod.cholesky(C) 
        # D = factor.L()
        D = cholesky(C)
        
        print('Computing inverse of D.T')
        invDt = sp.linalg.inv(D.T)
        # Eq 11
        print('Computing randomized SVD')
        E, S, Fh = randomized_svd(X @ invDt,
                                  n_components=rank,
                                  random_state=123)
        E_tilde = E[:, :rank]  # rank-r approximation; H_star = E_tilde (Eq 15)
        H_star = E_tilde  # Eq 15
        W_star = E_tilde.T @ X @ sp.linalg.inv(C)  # Eq 15
    else:
        # Eq 11
        I = np.eye(B.shape[0])
        C = I + (alpha * B)
        D = cholesky(C)
        Y = solve_triangular(D, X.T, lower=True).T
        E, s, Fh = svd(Y) # Singular values returned in descending order
        E_tilde = E[:, :rank]  # rank-r approximation; H_star = E_tilde (Eq 15)
        H_star = E_tilde  # Eq 15
        # print("C: {}, D: {}, X: {}, Y: {}, E: {}, E_tilde: {}".format(C.shape, D.shape, X.shape, Y.shape, E.shape, E_tilde.shape))
        W_star = solve_triangular(D.T, Y.T @ E_tilde)  # Eq 15
    return H_star, W_star

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
    activityM = np.log(activityM+1)
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
    H_star, W_star = regularized_svd(data, B, rank=1, alpha=1, as_sparse=False)
    # get SVD matrix
    svdM = W_star.reshape(activityM.shape)
    if svdM.max() <= 0:
        svdM = svdM * -1
    return svdM

# create list of indices for each sliding window
def sliding_window(elements, window_size, gap):
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
    binarizedSVD = np.where(svd_mat == 0, 0,1)
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
    window_size = 6
    hr_space = 1
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
    sns.heatmap(activityM, cmap='viridis', ax=ax[0,0], vmin=0, vmax=10,
                cbar_kws={'label': '# keypresses', 'fraction': 0.043})
    # PLOT 2
    sns.heatmap(speedM, cmap='viridis', ax=ax[0,1], vmin=0, vmax=0.3,
                cbar_kws={'label': '# keypresses', 'fraction': 0.043})
    # PLOT 3
    sns.heatmap(svdM, cmap='viridis', ax=ax[1,0],
                cbar_kws={'label': '# keypresses', 'fraction': 0.043})
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
    ax[0,0].set(title='Original Typing Activity', xlabel='Hour', ylabel='Day')
    ax[0,1].set(title='Original Typing Speed', xlabel='Hour', ylabel='Day')
    ax[1,0].set(title='Graph Regularized SVD', xlabel='Hour', ylabel='Day')    
    ax[1,1].set(title='Sleep/Wake Labels', xlabel='Hour', ylabel='Day')
    f.tight_layout()
    return f

#############################################################################################################

# Only run if the code is not imported as a module
if __name__ == '__main__':
    # file path of BiAffect keypress files
    pathIn = ''

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
            dfKP = pd.read_csv(file, index_col=False)
        else:
            dfKP = pd.read_parquet(file, engine='pyarrow')
        user = int(dfKP['userID'].unique())
        print('user: {}'.format(user))

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
        svd = get_SVD(Mactivity, Mspeed)
        
        # STEP 3
        # get sleep/wake labels by hour
        sleepMatrix = get_sleepWakeLabels(svd)
        
        # Plot steps if desired
        plot_heatmaps(Mactivity, Mspeed, svd, sleepMatrix)

        ################################################################

        # break    

    print('finish')


# %%
