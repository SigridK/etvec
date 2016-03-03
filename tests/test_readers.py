
from etvec import readers
import pytest


# *******************
# Tobii reader tests
# *******************

def test_tobii_bad_path():
    with pytest.raises(IOError):
        readers.tobii('data/nonexistent_dir/')


@pytest.fixture(params=[
                # dicts with (input file, reader params and expected output)
                {'filename': 'data/testTobii1.csv',
                 'read_params': {'subj': 'ParticipantName',
                                 'stim': 'MediaName',
                                 'groups': []},
                 'out_columns': ['subj', 'stim', 'fixID', 'dur',
                                 'coordX', 'coordY']},
                {'filename': 'data/testTobii1.csv',
                 'read_params': {'subj': 'ParticipantName',
                                 'stim': 'MediaName',
                                 'groups': '[Student]Value',
                                 'label': 'PupilRight'},
                 'out_columns': ['subj', 'stim', 'group1', 'fixID', 'dur',
                                 'coordX', 'coordY', 'label']},
                {'filename': 'data/testTobii1.csv',
                 'read_params': {'subj': 'ParticipantName',
                                 'stim': 'MediaName',
                                 'groups': ['RecordingName', '[Gender]Value']},
                 'out_columns': ['subj', 'stim', 'group1', 'group2', 'fixID',
                                 'dur', 'coordX', 'coordY']}
                ])
def test_tobiidata(request):
    """
    supply different raw tobii files,
    the dictionary of parameters to pass to the reader
    and the list of expected columns in the reader output.
    """
    return request.param


def test_tobii_reader(test_tobiidata):
    """
    Test that a raw tobii file is turned into a dataframe with
    - expected columns,
    - uniquely identifiable fixations from all grouping columns
    """

    args = test_tobiidata['read_params']

    df = readers.tobii(test_tobiidata['filename'],
                       **args)

    assert set(df.columns) == set(test_tobiidata['out_columns'])

    assert len(df.columns) == len(test_tobiidata['out_columns'])

    unique_keys = [c for c in df.columns if c not in
                   ['coordX', 'coordY', 'dur', 'label']]

    unique_rows = df.apply(lambda x:
                           '_'.join([str(x[c]) for c in unique_keys]),
                           axis=1)
    assert len(unique_rows.unique()) == len(df.index)


# *******************
# Dundee reader tests
# *******************

def test_dundee_bad_path():
    with pytest.raises(IOError):
        readers.dundee('data/nonexistent_dir/')


@pytest.fixture(params=[
                # dicts with (input dir, reader params and expected output)
                {'dundee_dir': 'data/testDundee',
                 'read_params': {},
                 'out_columns': ['subj', 'stim', 'fixID', 'dur',
                                 'coordX', 'coordY']},
                {'dundee_dir': 'data/testDundee',
                 'read_params': {'groups': 'LAUN', 'label': 'TXFR'},
                 'out_columns':  ['subj', 'stim', 'group1', 'fixID', 'dur',
                                  'coordX', 'coordY', 'label']},
                {'dundee_dir': 'data/testDundee',
                 'read_params': {'groups': ['WLEN', 'LAUN']},
                 'out_columns': ['subj', 'stim', 'group1', 'group2', 'fixID',
                                 'dur', 'coordX', 'coordY']}
                ])
def test_dundeedata(request):
    """
    supply different params to transform the dundee corpus,
    the dictionary of parameters to pass to the reader
    and the list of expected columns in the reader output.
    """
    return request.param


def test_dundee_reader(test_dundeedata):
    """
    Test that the dundee corpus is turned into a dataframe with
    - expected columns,
    - uniquely identifiable fixations from all grouping columns
    """

    args = test_dundeedata['read_params']

    df = readers.dundee(test_dundeedata['dundee_dir'],
                        **args)

    assert set(df.columns) == set(test_dundeedata['out_columns'])

    unique_keys = [c for c in df.columns if c not in
                   ['coordX', 'coordY', 'dur', 'label']]

    unique_rows = df.apply(lambda x:
                           '_'.join([str(x[c]) for c in unique_keys]),
                           axis=1)
    assert len(unique_rows.unique()) == len(df.index)


# *******************
# Coordinate reader tests
# *******************

def test_coordinates_bad_paths():
    with pytest.raises(IOError):
        readers.coordinates('data/nonexistent_dir/')

    # not all csv-files have the right format
    with pytest.raises(Exception):  # TODO specify which error!
        readers.coordinates('data/')


def test_coordinates():
    """
    Test that coordinates from csv-files generated with stimulus_builder are
    read into df if the dir contains appropriately formed csv-files.
    """

    df = readers.coordinates('data/coord/')
    assert set(df.columns) == set(['stim', 'id', 'text', 'height', 'width',
                                   'top', 'bottom', 'left', 'right'])
