import pandas
import logging

from listener.config import conf
from pathlib import Path, PurePath

log = logging.getLogger(__name__)


def _get_source_file_path():
    DATA_SOURCE_DIRECTORY = conf['DEFAULT']['SOURCE_FILE_DIRECTORY']
    DATA_SOURCE_FILE = conf['DEFAULT']['SOURCE_FILE_NAME']
    path = PurePath(DATA_SOURCE_DIRECTORY).joinpath(DATA_SOURCE_FILE)
    return Path.expanduser(path)

def import_csv(is_qualtrics_file: bool=False):
    """Import source datafile containing the raw comment text

    :param bool is_qualtrics_file: Is this file exported directly from Qualtrics
        with 3 initial rows of column headings? If yes, the 2nd row with descriptive
        column headings is used and the 1st and 3rd rows are dropped.
    """
    if is_qualtrics_file:
        data = pandas.read_csv(_get_source_file_path(), header=1)
        data = data.drop(labels=0, axis=0).reset_index()
    else:
        data = pandas.read_csv(_get_source_file_path())
    return data

def isolate_comment_col(data: pandas.DataFrame, focus_col_list: list):
    """isolate the raw comment columns with associated row id so we can
    feed the comments into subsequent pipelines in this package. Use this function
    if the source file has a lot of additional columns not necessary for text analysis.

    :param pandas.DataFrame data: A dataframe containing unstructured comments

    :param list focus_col_list: the location of the row index and text cols. For example,
        Qualtrics uses the Response ID for each unique response so we can
        use this to keep track of the individual comments if we need to join back later.
        Also, include column indexes for the specific comment cols.
    """
    return data.iloc[:, focus_col_list]