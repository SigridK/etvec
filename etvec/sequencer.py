
"""
Class for extracting a pandas dataframe of gaze snippets from a reader object.
Parameters include window size and gaze aspects.

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
        result = (aoi_ids - aoi_ids.shift(1)).values

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


def snipper(subdf):
    """
    1: assume a (participant X stimulus) frame of fixations,
    2: get the list of token-ids fixated from column 'aoi_id',
    3: for each fixation (row) get saccade distance sequence,
    4: return dict of fixation-ids and their saccade distance sequence
    """

    trans = {'sacc': (saccade_len, False), 'saccRel': (saccade_len, False),
             'saccAbsDirect': (saccade_len, 'direction'),
             'fixd': (fixation_dur, False), 'fixdRel': (fixation_dur, False),
             'fixdRelDirect': (fixation_dur, 'direction')}

    sequence_dict = {}
    for fix_id in subdf.fixID:
        sequence_dict[fix_id] = {name: param[0](subdf, fix_id,
                                                subdf.aoi_id, param[1])
                                 for (name, param) in trans.items()}

    return sequence_dict


# -----------------
# main functions
# -----------------


def raw_snips(df):
    """
    Assume dataframe formated via etvec reader+annotator.
    columns: ['subj', 'stim', 'fixID', 'dur',
              'aoi_id', 'aoi', 'fixcount', 'label']

    Transform is a list of functions: [saccade_len, fixation_dur]

    Relative is a keyword for the transform function;
    can be bool or 'direction', which records change as +/-1 only.

    (Cut should be implemented later)
    """

    # make sure fixID and aoi_id are 0-indexed ()
    if df[df.fixID == 0].shape[0] == 0:
        df.fixID = df.fixID - 1
    if df[df.aoi_id == 0].shape[0] == 0:
        df.aoi_id = df.aoi_id - 1

    # generate empty snippet-dataframe of
    # (fixID X relative fixID distance)
    snips = pd.DataFrame()

    snips['uniqID'] = uniq_indexer(df)
    snips.set_index('uniqID', append=False, inplace=True)
    trans = ['fixd', 'fixdRel', 'fixdRelDirect',
             'sacc', 'saccRel', 'saccAbsDirect']

    col_list = zip(trans, [list(range(int(-df.fixID.max()-1),
                                      int(df.fixID.max()+1)))
                           for _ in range(len(trans))])
    zeroes = len(str(int(df.fixID.max())))+2
    col_list = [[name+str(int(i)).zfill(zeroes) for i in cols]
                for name, cols in col_list]

    snips = snips.reindex(columns=[item for sublist in col_list
                                   for item in sublist])

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
        seq_dict = snipper(subdf)

        for fix_id, snip_dict in seq_dict.items():

            snip_idx = uniq_indexer(subdf[subdf.fixID == fix_id])
            seq_len = len(snip_dict['sacc'])
            snip_cols, gaze_snips = [], []
            raw_cols = np.arange(-fix_id, (seq_len - fix_id))
            for name, snip in snip_dict.items():
                snip_cols += [name +
                              str(int(n)).zfill(zeroes)
                              for n in raw_cols]
                gaze_snips += snip.tolist()

            # put the sequence in the right spot of the big snip-df
            snips.loc[snip_idx, snip_cols] = gaze_snips

    snips['aoi'] = df.aoi
    snips['aoi_id'] = df.aoi_id
    snips['label'] = df.label
    snips['fixcount'] = df.fixcount

    return snips


def vectors(df_list, fix_range=[-1, 3], fill=np.NaN):
    pass
