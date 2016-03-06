
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


def annotate_coords(df, coord_df, label=None):
    """
    Expects a dataframe (from a reader-function)
    + a dataframe for looking-up (X,Y)-coordinates
    based on stim-column.
    Returns the reader-dataframe with aoi-based columns.
    """

    df['aoi'], df['aoi_id'] = None, None

    # go through each stimulus with groupby:
    for stimulus, subdf in df.groupby('stim'):

        # fetch relevant coordinates
        coords = coord_df[coord_df.stim == stimulus]

        # aoi_condition = ((coords.top < fix.coordY) &
        #                  (coords.bottom > fix.coordY) &
        #                  (coords.left > fix.coordX) &
        #                  (coords.right < fix.coordX))

        aoi = subdf.apply(lambda fix:
                                  coords[((coords.top < fix.coordY) &
                                          (coords.bottom > fix.coordY) &
                                          (coords.left < fix.coordX) &
                                          (coords.right > fix.coordX))].text.min(),
                                  axis=1).T

        aoi_id = subdf.apply(lambda fix:
                                  coords[((coords.top < fix.coordY) &
                                          (coords.bottom > fix.coordY) &
                                          (coords.left < fix.coordX) &
                                          (coords.right > fix.coordX))].id.min(),
                                  axis=1).T

        # insert the new col in the original df:
        df.loc[subdf.index, ['aoi']] = aoi
        df.loc[subdf.index, ['aoi_id']] = aoi_id

    return df
