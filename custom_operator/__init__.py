import os
import os.path
import logging
import sqlite3
from typing import Callable, Dict, List
from contextlib import contextmanager

import pandas as pd
from airflow import DAG
from airflow.models.baseoperator import BaseOperator

from common.utils import DFHandler


def context_path(context):
    logging.error(context)
    data_path = context.get('data_path')
    dag: DAG = context.get('dag')
    ds = context.get('ds', 'temp')
    cur_path = ds
    if dag is not None:
        cur_path = os.path.join(dag.dag_id, ds)
    if data_path is not None:
        cur_path = os.path.join(data_path, ds)
    if not os.path.exists(cur_path):
        os.makedirs(cur_path)
    return cur_path


class CSVOperator(BaseOperator):
    """
    default path: {data_path}/{ds}/[read_filename|write_filename]
    """

    def __init__(self,
                 read_filenames: List[str],
                 handler: DFHandler,
                 write_filename=None,
                 sep=',',
                 inplace=False,
                 **kwargs) -> None:
        """

        Args:
            read_filenames:
            handler:
            write_filename:
            sep:
            inplace:
            **kwargs:
        """
        super().__init__(**kwargs)
        self.read_filenames = read_filenames
        self.write_filename = write_filename
        self.handler: DFHandler = handler
        self.inplace = inplace
        self.sep = sep

    def execute(self, context: Dict, **kwargs):
        cur_path = context_path(context)
        df_list = []
        for read_filename in self.read_filenames:
            if read_filename[0] in ('/', '\\'):
                read_path = read_filename
            else:
                read_path = os.path.join(cur_path, read_filename)
            df = pd.read_csv(read_path, encoding='utf-8', sep=self.sep)
            df_list.append(df)
        res_df = self.handler.handle(*df_list)

        if self.inplace:
            res_df.to_csv(read_path, encoding='utf-8', index=False)
        elif self.write_filename is not None:
            write_path = os.path.join(cur_path, self.write_filename)
            res_df.to_csv(write_path, encoding='utf-8', index=False)

        return self.write_filename


class SqliteOperator(BaseOperator):
    def __init__(self, db_path, select_sql, write_filename, **kwargs):
        super().__init__(**kwargs)
        self.db_path = db_path
        self.select_sql = select_sql
        self.write_filename = write_filename

    def execute(self, context: Dict, **kwargs):
        cur_path = context_path(context)
        connection = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(self.select_sql, connection)
        write_path = os.path.join(cur_path, self.write_filename)
        df.to_csv(write_path, encoding='utf-8', index=False)
        return self.write_filename
