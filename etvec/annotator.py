
"""
Functions for annotating a gaze-frame from a reader object with
- a custom label
- a set of token-identifiers (word form and index)
Supports grouping and multiclass labels.
Returns a pandas dataframe with index from original reader
and an optional label column.
"""

# provide x-y resolution per stimulus of words
# using the same unit as reader-coordinates
# (i.e.: dundee:wordID,Line; tobii:x_pxls,y_pxls)

import pandas as pd
import numpy as np


def annotate_coords(df, coord_df, fixcount=True, dundee=False):
    """
    Expects a dataframe (from a reader-function)
    + a dataframe for looking-up (X,Y)-coordinates
    based on stim-column.
    Returns the reader-dataframe with aoi-based columns.
    Label should be a df or nested dictionary with
    [stimulus, tokenID] as keys and label as value.
    Fixcount annotates how many times the aoi has been fixated incl. current.
    """

    curr_stim = None
    df['aoi_id'] = np.NaN
    df['aoi'] = np.NaN

    for i, row in coord_df.iterrows():

        if row['stim'] != curr_stim:
            curr_stim = row['stim']
            if i % 10 == 0:
                print(curr_stim)

        if dundee:
            upper, lower = row['coordY'], row['coordY']
        else:
            upper, lower = row['bottom'] + 100, row['top'] - 100

        left, right = row['left'], row['right']

        target = df[((left <= df.coordX) &
                     (df.coordX <= right) &
                     (lower <= df.coordY) &
                     (df.coordY <= upper) &
                     (df.stim == row['stim']))]

        df.loc[target.index, ['aoi_id', 'aoi']] = row['id'], row['text']

    # consider keeping the coordY for knowing when lines change...
    df = df.drop(['coordX', 'coordY'], axis=1)

    if fixcount:
        df['fixcount'] = df.groupby(['stim', 'subj']).aoi_id.apply(lambda x:
                                                                   fnummer(x))

    return df


def fnummer(fix_seq):
    """
    helper function for annotating fixation counts.
    """
    fcount = []
    for i, x in enumerate(fix_seq.values):
        fcount.append(sum([1 for v in fix_seq.values[:i + 1] if v == x]))

    # debugging:
    if fcount[0] > 1:
        raise Exception('First fixID is not recorded as such:\n'
                        'input fixation sequence: {}, \n'
                        'recorded fixation counts: {}'.format(fix_seq, fcount))

    return pd.Series(fcount, index=fix_seq.index)


def labeler(df, labels, keys=[], to_quantile=False, inplace=False):
    """
    function for annotating df with labels.
    Keys should identify unique labels and exist
    both as column(s) in df and as (multi-)index in labels.
    If to_quantile = n > 1, first transform label-column to n
    groups by splitting into quantiles of 1/n size
    """
    if not inplace:
        df_cp = df.copy()
    else:
        df_cp = df

    l_cp = labels.copy()
    if to_quantile > 1:
        l_cp = categorize(pd.DataFrame(l_cp), [l_cp.name], to_quantile,
                          print_bins=True)

    if keys:
        for key, group in df_cp.groupby(keys):
            df_cp.loc[group.index, 'label'] = l_cp.loc[key]

    elif df_cp.index.shape[0] == labels.index.shape[0]:
        df_cp['label'] = l_cp

    return df_cp


def relative_dur(df, inplace=False):
    """
    Add column of durations relative to personal
    median first fixation duration
    """
    if not inplace:
        df_cp = df.copy()
    else:
        df_cp = df_cp

    subj_vals = df_cp[df_cp.fixcount == 1].groupby(
        ['subj']).dur.apply(lambda x: x.median())

    for subj, subdf in df_cp.groupby('subj'):
        df_cp.loc[subdf.index, 'rel_dur'] = subdf.dur - subj_vals[subj]

    return df_cp


def categorize(df, cols, n_quantiles=4, print_bins=False, inplace=False):
    """
    function for quantilizing columns in n_quantile bins.
    If print_bins, print the cut-offs used.
    TODO: implement accepting pre-defined categories.
    """
    if not inplace:
        cat_df = df.copy()
    else:
        cat_df = df

    for col in cols:
        limits = cat_df[col].quantile([i/n_quantiles for
                                      i in range(1, n_quantiles+1)])
        if print_bins:
            print('bins used for column:', col)
            print(limits)
            print()

        cat_df[col] = cat_df[col].apply(lambda x: [i[0] for i in limits.items()
                                        if x <= i[1]][0])

    return cat_df
