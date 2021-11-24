from contextlib import contextmanager
from collections import namedtuple

from airflow.operators import python

from common.utils import ColumnNameSpace, get_columns


class DFHandler:
    def __init__(self, df=None):
        self.columns_obj: ColumnNameSpace = get_columns()

    def handle(self, df):
        raise NotImplementedError

    def change_column_name(self, default_name, custom_name):
        self.columns_obj._replace(**{default_name: custom_name})
