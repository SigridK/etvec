
"""
Class for extracting a pandas dataframe of gaze snippets from a reader object.
Parameters include window size and gaze aspects.

FROM DURATION_SNIPPETS.IPYNB

Don't trust fixcount for now.
"""

import numpy as np
import pandas as pd

# -----------------
# helper functions
# -----------------


def saccade_len(subdf, fix_id, aoi_ids, relative=False):
    """
    return array recording distances as n aoi_ids from currently focused token
    or if relative:
    return array recording distances as n aoi_ids from last fixation,
    i.e. saccade length counted in aoi_ids
    """

    if relative:
        result = (aoi_ids-aoi_ids.shift(1)).values

        if relative == 'direction':
            result = np.sign(result)

    else:
        result = aoi_ids.add(-subdf[subdf.fixID ==
                             fix_id].aoi_id.values[0]).values

    return result


def fixation_dur(subdf, fix_id, aoi_ids=None, relative=False):
    """return array of fixations durations
    if relative, df should have a rel_dur column"""

    if relative == 'direction':
        result = np.sign(subdf.dur - subdf.dur.shift(1)).values

    elif not relative:
        result = subdf.dur.values

    else:
        try:
            result = subdf.rel_dur.values
        except:
            raise Exception('no relative duration column (rel_dur)')

    return result


def uniq_indexer(df):
    return df[['subj', 'stim', 'fixID']].apply(lambda x:
                                               '_'.join([str(val) for
                                                         val in x.values]),
                                               axis=1).values


def snipper(subdf, transform=saccade_len, relative=False):
    """
    1: assume a (participant X stimulus) frame of fixations,
    2: get the list of token-ids fixated from column 'aoi_id',
    3: for each fixation (row) get saccade distance sequence,
    4: return dict of fixation-ids and their saccade distance sequence
    """

    sequence_dict = {fix_id: transform(subdf, fix_id, subdf.aoi_id, relative)
                     for fix_id in subdf.fixID}

    return sequence_dict


# -----------------
# main functions
# -----------------


def raw_snips(df, transform, relative='direction', cut=0):
    """
    Assume dataframe formated via etvec reader+annotator.
    columns: ['subj', 'stim', 'fixID', 'dur',
              'aoi_id', 'aoi', 'fixcount', 'label']

    Transform is a function of [saccade_len, fixation_dur]

    Relative is a keyword for the transform function;
    can be bool or 'direction', which records change as +/-1 only.

    Cut can be sent to the transform function to limit extreme values.
    Default [0] does not cut any values.

    """

    # make sure fixID and aoi_id are 0-indexed ()
    if df[df.fixID == 0].shape[0] == 0:
        df.fixID = df.fixID - 1
    if df[df.aoi_id == 0].shape[0] == 0:
        df.aoi_id = df.aoi_id - 1

    # get the maximum fixation number:
    max_fixID = df.fixID.max()

    # generate empty snippet-dataframe of
    # (fixID X relative fixID distance)
    snips = pd.DataFrame(index=df.index,
                         columns=np.arange(-(max_fixID + 1), max_fixID + 1))
    snips['aoi'] = df.aoi
    snips['aoi_id'] = df.aoi_id
    snips['label'] = df.label
    snips['fixcount'] = df.fixcount
    snips['uniqID'] = uniq_indexer(df)
    snips.set_index('uniqID', append=False, inplace=True)

    # for debugging timing
    rest_subj = len(df.subj.unique())
    curr_subj = ''

    # apply snipping to populate snips-df
    for (subj, stim), subdf in df.groupby(['subj', 'stim'], as_index=False):

        # debugging timing
        if subj != curr_subj:
            curr_subj = subj
            rest_subj -= 1
            print(curr_subj, rest_subj)

        # get sequence-dict from snipper function
        seq_dict = snipper(subdf, transform, relative)

        for fix_id, gaze_snip in seq_dict.items():

            snip_cols = np.arange(-fix_id, (len(gaze_snip) - fix_id))
            snip_idx = uniq_indexer(subdf[subdf.fixID == fix_id])

            # handle cut-value:
            if cut:
                gaze_snip = [v if (np.sign(v)*v < cut) or (v is np.NaN)
                             else np.sign(v)*cut for v in gaze_snip]

            # # debugging insertion
            # if subj in ['sa', 'sj']:
            #     print(snip_cols[:5])
            #     print(fix_id, snip_idx)
            #     print(gaze_snip[:5])
            #     print()

            # put the sequence in the right spot of the big snip-df
            snips.loc[snip_idx, snip_cols] = gaze_snip

    return snips
