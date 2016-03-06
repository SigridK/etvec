
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
            print(curr_stim)

        if dundee:
            upper, lower = row['coordY'], row['coordY']
        else:
            upper, lower = row['bottom']+100, row['top']-100

        left, right = row['left'], row['right']

        target = df[((left <= df.coordX) &
                     (df.coordX <= right) &
                     (lower <= df.coordY) &
                     (df.coordY <= upper) &
                     (df.stim == row['stim']))]

        df.loc[target.index, ['aoi_id', 'aoi']] = row['id'], row['text']
        #df.loc[target.index, 'aoi'] = row['text']

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
    for i, x in fix_seq.items():
        fcount.append(len([v for v in fix_seq.values[:i+1] if v == x]))
    return pd.Series(fcount, index=fix_seq.index)


def labeler(df, labels, keys=[]):
    """
    function for annotating df with labels.
    Keys should identify unique labels and exist
    both as column(s) in df and as (multi-)index in labels.
    """

    for key, group in df.groupby(keys):
        df.loc[group.index, 'label'] = labels.loc[key]

    return df
