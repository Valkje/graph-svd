Code to merge, plot, and analyse key press and sleep data.

## Getting started

I have included an environment.yml file that can be used to create a conda environment called `graph_svd`. (This file was created on Almalinux.) To create the environment, run:

```console
conda env create -f environment.yml
```

Alternatively, you can install all Python packages manually with pip.

In addition, the code expects a `.env` file to set the path to the proper data directory. This has to be created manually, and should look like this:

```
DAT_DIR=/path/to/data/dir
```

## Code run-down

- Most useful library functions are in `gsvd.py`. An example of how to use the functions can be found in `analyze_alex.ipynb`.
- `gsvd.py` mostly consists of functions for data preparation. My entry point there is `get_typing_matrices`, which constructs day-by-hour matrices for every relevant BiAffect variable, for every participant. It does this by taking in keypress- and session-level data frames, ranking the dates within participants, filtering, and then constructing a long-format data frame that looks like this, with a hierarchical row index of (subject, dayNumber, hour):
```
                             IKD  n_presses    active  upright
subject dayNumber hour                                        
alex    0         19    0.182817        771  0.066667      1.0
                  20    0.233297        392  0.095238      1.0
                  21    0.176844         49  0.000000      1.0
                  22    0.000000         34  0.000000      1.0
                  23    0.199749        213  0.400000      1.0
...                          ...        ...       ...      ...
        344       18    0.182697       1363  0.235294      1.0
                  19    0.199217        446  0.166667      1.0
                  20    0.000000         15  0.500000      1.0
                  21    0.198719        154  0.000000      1.0
                  22    0.182370        158  0.500000      1.0
```
- Within `get_typing_matrices`, `pivot_split` then takes this data frame and pivots every variable column into a day-by-hour matrix, resulting in a hierarchical column index of (variable, hour):
```
                        IKD                                                    
hour                     0         1         2         3         4         5    
subject dayNumber                                                               
alex    0          0.000000  0.000000  0.000000  0.000000  0.000000  0.000000   
        1          0.182950  0.000000  0.216362  0.000000  0.183777  0.241262   
        2          0.215384  0.000000  0.200042  0.167262  0.200689  0.000000   
        3          0.182904  0.199669  0.000000  0.000000  0.000000  0.000000   
        4          0.196758  0.216712  0.184235  0.000000  0.182593  0.000000   
...                     ...       ...       ...       ...       ...       ...   ...
        340        0.000000  0.184188  0.000000  0.000000  0.000000  0.000000   
        341        0.140836  0.000000  0.000000  0.000000  0.000000  0.000000   
        342        0.000000  0.191480  0.000000  0.233269  0.000000  0.000000   
        343        0.000000  0.199515  0.000000  0.000000  0.000000  0.000000   
        344        0.000000  0.174891  0.000000  0.000000  0.000000  0.000000   
```
- After this, the entire data frame is split into separate matrices based on subject and variable, resulting in a two-level dictionary with participant as the first index and variable as the second index:
```
{'alex': {'IKD': 
hour             0         1         2         3         4         5       6   
dayNumber                                                                       
0          0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.0000   
1          0.182950  0.000000  0.216362  0.000000  0.183777  0.241262  0.2836   
2          0.215384  0.000000  0.200042  0.167262  0.200689  0.000000  0.0000   
3          0.182904  0.199669  0.000000  0.000000  0.000000  0.000000  0.0000   
4          0.196758  0.216712  0.184235  0.000000  0.182593  0.000000  0.0000   
...             ...       ...       ...       ...       ...       ...     ...   ...
340        0.000000  0.184188  0.000000  0.000000  0.000000  0.000000  0.0000   
341        0.140836  0.000000  0.000000  0.000000  0.000000  0.000000  0.0000   
342        0.000000  0.191480  0.000000  0.233269  0.000000  0.000000  0.0000   
343        0.000000  0.199515  0.000000  0.000000  0.000000  0.000000  0.0000   
344        0.000000  0.174891  0.000000  0.000000  0.000000  0.000000  0.0000      ,
...
```
- After data preparation, the two-level dictionary from `get_typing_matrices` can be passed to `calculate_svd`, which loops over all participants, constructs a time connectivity graph based on the row and column indices of their matrices, and calculates the graph-regularised SVD. It supports train/test splits, but by default uses all data for training (the difference has never been huge because we only used four data modalities for BiAffect, which results in a small number of parameters). With train/test splits, the output is a two-level dictionary (one-level without train/test splits) of numpy arrays, with the first index being the participant, and the second being `'train'` or `'test'`:
```
{'alex': {'train': array([[1.36293734, 1.63534636, 1.70035293, 1.00559802, 0.71495363,
          1.00167044, 0.3900861 , 0.18152092, 0.12258938, 0.13548369,
          0.22333073, 0.47443882, 1.19062331, 0.992356  , 1.47716155,
          2.58896073, 3.03002062, 2.73025944, 2.21940934, 1.54958712,
          1.41246628, 1.22464226, 1.52615389, 2.48503361],
...
```
- Although `get_typing_matrices` heavily relies on the assumption that we want to aggregate our data to hours, `calculate_svd` is relatively time bin agnostic, in that it primarily looks at the data matrix shapes and does not care if the data has been aggregated into hours or, say, half hours. The most important thing is that the rows and columns are marked with integer indices, which is used during the construction of the connectivity graph to determine adjacency.

## Regression

The current Rmarkdown files do not contain the most up-to-date regressions.
