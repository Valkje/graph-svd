import function_sleepWakeLabels as sleep
import pandas as pd
import glob
import re
from os.path import join
from tqdm import tqdm
import pickle

dat_dir = '/project_cephfs/3022017.02/projects/lorkno/data'

all_files = sorted(glob.glob("sub-*/preproc/*dat-kp.csv", root_dir=dat_dir, recursive=True))

pat = re.compile(r"sub-(\d+)")
subs = [re.match(pat, f).group(1) for f in all_files]

gsvd_results = {}

# Using zip does not allow tqdm to know the max number of iterations
n_subs = len(subs)
for i in tqdm(range(n_subs)):
    file = all_files[i]
    sub = subs[i]

    # read in keypress file
    dfKP = pd.read_csv(join(dat_dir, file), index_col=0)
    dfKP['date'] = pd.to_datetime(dfKP['keypressTimestampLocal']) \
        .map(lambda x: x.date())
    dfKP['dayNumber'] = dfKP['date'].rank(method='dense')

    # Necessary for joining sleep self-report to the key press data
    dates = dfKP[['date', 'dayNumber']].drop_duplicates()

    ################################################################
    # FIND SLEEP/WAKE LABELS FROM BIAFFECT KEYPRESS DATA FILE
    ################################################################
    # STEP 1
    # get input matrices of shape days x hours for typing activity (nKP) and speed (median IKD)
    ## matrices may have missing days
    ## check index here to identify day number since first date of typing data
    Mactivity, Mspeed = sleep.get_typingMatrices(dfKP)

    if Mactivity.empty:
        print("Not enough data, skipping subject {}".format(sub))
        continue

    # STEP 2
    # get graph regularized SVD
    svd = sleep.get_SVD(Mactivity, Mspeed)

    # STEP 3
    # get sleep/wake labels by hour
    sleepMatrix = sleep.get_sleepWakeLabels(svd)

    gsvd_results[sub] = {
        'dates': dates,
        'Mactivity': Mactivity,
        'Mspeed': Mspeed,
        'svd': svd,
        'sleepMatrix': sleepMatrix
    }

with open(join(dat_dir, 'gsvd_results.pkl'), 'wb') as handle:
    pickle.dump(gsvd_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
