import os
from typing import Callable, Dict, Optional
from contextlib import contextmanager

import pandas as pd
from airflow.models.baseoperator import BaseOperator

from custom_operator.utils import DFHandler


class CSVOperator(BaseOperator):
    """
    default path: {data_path}/{ds}/[read_filename|write_filename]
    """

    def __init__(self,
                 read_filename,
                 handler: DFHandler,
                 write_filename=None,
                 sep=',',
                 inplace=False,
                 **kwargs) -> None:
        """

        Args:
            read_filename:
            handler:
            write_filename:
            sep:
            inplace:
            **kwargs:
        """
        super().__init__(**kwargs)
        self.read_filename = read_filename
        self.write_filename = write_filename
        self.handler: DFHandler = handler
        self.inplace = inplace
        self.sep = sep

    def execute(self, context: Dict, **kwargs):
        data_path = context.get('data_path')
        ds = context.get('ds', 'temp')
        if self.read_filename[0] in ('/', '\\'):
            read_path = self.read_filename
        else:
            read_path = os.path.join(ds, self.read_filename)
            if data_path is not None:
                read_path = os.path.join(data_path, read_path)
        df = pd.read_csv(read_path, encoding='utf-8', sep=self.sep)
        res_df = self.handler.handle(df)

        if self.inplace:
            res_df.to_csv(read_path, encoding='utf-8', index=False)
        elif self.write_filename is not None:
            write_path = os.path.join(ds, self.write_filename)
            if data_path is not None:
                write_path = os.path.join(data_path, write_path)
            res_df.to_csv(write_path, encoding='utf-8', index=False)

        return self.write_filename
