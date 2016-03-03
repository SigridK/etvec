
from etvec import annotator
from etvec import readers
import pytest


@pytest.fixture(params=[
                # dicts with (input df, annotator params and expected output)
                {'df': readers.tobii('data/testTobii1.csv',
                                     subj='ParticipantName',
                                     stim='MediaName',
                                     groups=['RecordingName',
                                             '[Gender]Value']),
                 'coord_df': readers.coordinates('data/coord/')}
                ])
def test_data(request):
    return request.param


def test_annotator(test_data):
    """

    """

    assert annotator
