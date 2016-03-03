
"""
Functions for reading gaze data from different sources.
Produces a pandas dataframe object with one row per fixation and columns
['subj','stim',*'group1',*'group2',*...,'coordX','coordY','dur',*'label']
*:optional
"""

import pandas as pd


def tobii(filename, groups=[], sep='\t', label=[],
          subj='ParticipantName', stim='MediaName', fixID='FixationIndex',
          coordX='FixationPointX (MCSpx)', coordY='FixationPointY (MCSpx)',
          dur='GazeEventDuration'):
    """
    Read specified columns of Tobii-file (incl. groups and label)
    Return standardized dataframe of one fixation per row and
    re-named columns.
    """

    if type(label) != list:
        label = [label]

    if type(groups) != list:
        groups = [groups]

    df = pd.read_csv(filename, sep=sep, encoding='utf-8', header=0,
                     index_col=None,
                     usecols=[subj, stim] + groups +
                     [fixID, coordX, coordY, dur] + label)

    df.reset_index(drop=False)

    df.rename(columns={subj: 'subj', stim: 'stim', fixID: 'fixID',
                       coordX: 'coordX', coordY: 'coordY', dur: 'dur'},
              inplace=True)

    for i, c in enumerate(groups):
        df.rename(columns={c: 'group'+str(i+1)}, inplace=True)

    if label:
        df.rename(columns={label[0]: 'label'}, inplace=True)

    groupcols = ['subj', 'stim', 'fixID'] + ['group' + str(i+1) for (i, _)
                                             in enumerate(groups)]

    grouper = df.groupby(groupcols, as_index=False, sort=False)
    result = grouper.first()

    return result


def dundee(dundee_path, groups=[], label=[]):
    """
    Read all .dat files in specified directory (assuming raw dundee corpus).
    Return one standardized dataframe of one fixation per row and
    re-named columns.
    Include columns in "groups" as group1 ... groupN.
    Coordinates are in characters (X) and lines (Y)
    """
    import os

    if type(label) != list:
        label = [label]

    if type(groups) != list:
        groups = [groups]

    result = pd.DataFrame()

    fnames = (f for f in os.listdir(dundee_path) if f.endswith('ma1p.dat') and
              not f.startswith('tx'))

    for filename in fnames:
        full_path = os.path.join(dundee_path, filename)
        subj, stim_prefix = filename[:2], filename[2:4]

        subdf = pd.read_csv(full_path, encoding='latin1',
                            delim_whitespace=True, skipinitialspace=True)

        # remove blinks to get fixation count from new index
        subdf = subdf[subdf.WORD != '*Blink']
        subdf.reset_index(inplace=True, drop=True)

        # rename and process the columns of interest
        # to prepare for concatenation
        subdf.rename(columns={'LINE': 'coordY', 'XPOS': 'coordX',
                              'FDUR': 'dur'}, inplace=True)
        subdf['subj'] = subj
        subdf['stim'] = 'tx' + stim_prefix + 'img' + subdf.TEXT.astype(str)
        subdf['fixID'] = subdf.index

        for i, gr in enumerate(groups):
            subdf.rename(columns={gr: 'group'+str(i+1)}, inplace=True)

        if label:
            subdf.rename(columns={label[0]: 'label'}, inplace=True)

        out_cols = ['subj', 'stim'] + \
                   ['group'+str(i+1) for i, _ in enumerate(groups)] + \
                   ['fixID', 'dur', 'coordX', 'coordY'] + \
                   ['label' for _ in label]

        # print(out_cols)
        # subdf = subdf[out_cols]

        result = result.append(subdf[out_cols],
                               ignore_index=True, verify_integrity=True)

    return result


def coordinates(tsv_dir):
    """
    Function for reading coordinates from generated csv-files into dataframe
    (from stimulus_builder) with columns
    [stim, id, text, height, width, top, bottom, left, right]
    """
    import os

    fnames = [f for f in os.listdir(tsv_dir) if f.endswith('.tsv')]
    # raise exception if no files:
    if len(fnames) < 1:
        raise Exception('No files ending in .tsv in dir: {}'.format(tsv_dir))

    df = pd.DataFrame()

    for f in fnames:
        full_path = os.path.join(tsv_dir, f)
        stim = f.rstrip('.tsv')
        subdf = pd.read_csv(full_path, sep='\t', header=0, encoding='utf-8')
        if set(subdf.columns) == set(['id', 'text', 'height', 'width',
                                      'top', 'bottom', 'left', 'right']):
            subdf['stim'] = stim
            df = pd.concat([df, subdf], axis=0, ignore_index=True)

    return df
