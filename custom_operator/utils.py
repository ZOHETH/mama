from typing import Optional, TYPE_CHECKING

from common.utils import ColumnNameSpace, get_columns

if TYPE_CHECKING:
    from pandas import DataFrame


class DFHandler:
    def __init__(self):
        self.columns_obj: ColumnNameSpace = get_columns()

    def handle(self, df) -> Optional[DataFrame]:
        raise NotImplementedError

    def change_column_name(self, default_name, custom_name):
        self.columns_obj._replace(**{default_name: custom_name})
