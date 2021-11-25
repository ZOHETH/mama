from collections import namedtuple
from typing import Optional

from pandas import DataFrame


columns = ['customer_id', 'text', 'speaker', 'intent', 'action_name']
ColumnNameSpace = namedtuple('ColumnNameSpace', columns)


def get_columns(necessary_columns=tuple(columns)):
    NecessaryColumnNameSpace = namedtuple('NecessaryColumnNameSpace', necessary_columns)
    return NecessaryColumnNameSpace(*necessary_columns)


class DFHandler:
    def __init__(self):
        self.columns_obj: ColumnNameSpace = get_columns()

    def handle(self, *df_list) -> Optional[DataFrame]:
        raise NotImplementedError

    def change_column_name(self, default_name, custom_name):
        self.columns_obj._replace(**{default_name: custom_name})
