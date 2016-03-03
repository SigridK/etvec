
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


def labeler(df, coord_df):
    """
    Expects a dataframe (from a reader-function)
    + a dataframe for looking-up (X,Y)-coordinates
    based on stim-column.
    Returns the reader-dataframe with a 'label' column.
    """

    #df.groupby('stim') = 

    #result = df
    return #result
