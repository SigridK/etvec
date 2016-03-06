
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


def annotate_coords(df, coord_df, label=None, fixcount=True):
    """
    Expects a dataframe (from a reader-function)
    + a dataframe for looking-up (X,Y)-coordinates
    based on stim-column.
    Returns the reader-dataframe with aoi-based columns.
    Label should be a df or nested dictionary with
    [stimulus, tokenID] as keys and label as value.
    Fixcount annotates how many times the aoi has been fixated incl. current.
    """

    df['aoi'], df['aoi_id'] = None, None

    # go through each stimulus with groupby:
    for stimulus, subdf in df.groupby('stim'):

        # fetch relevant coordinates
        coords = coord_df[coord_df.stim == stimulus]

        # map fixation and aoi coordinates
        aoi_id = subdf.apply(lambda fix:
                             coords[((coords.top < fix.coordY) &
                                     (coords.bottom > fix.coordY) &
                                     (coords.left < fix.coordX) &
                                     (coords.right > fix.coordX))].id.min(),
                             axis=1).T

        aoi = aoi_id.apply(lambda x: coords[coords.id == x].text.min())

        # insert the new columns in the original df:
        df.loc[subdf.index, ['aoi']] = aoi
        df.loc[subdf.index, ['aoi_id']] = aoi_id

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
        fcount.append(len([v for v in fix_seq.values[:i] if v == x]))
    return pd.Series(fcount, index=fix_seq.index)
