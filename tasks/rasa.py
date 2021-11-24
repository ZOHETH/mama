import json
import re
from collections import namedtuple
from typing import TYPE_CHECKING, Callable
import requests
import pandas as pd
from tqdm import tqdm
from airflow.decorators import task

from custom_operator.utils import DFHandler
from common.utils import ColumnNameSpace, get_columns

if TYPE_CHECKING:
    from pandas import DataFrame
    from common.rasa import RASAClient


class UploadConvDataHandler(DFHandler):
    def __init__(self, client: RASAClient, df=None):
        super().__init__(df)
        self.client = client
        necessary_columns = ('customer_id', 'text')
        self.columns_obj: ColumnNameSpace = get_columns(necessary_columns)

    def handle(self, df):
        cols = self.columns_obj
        counts = df[cols.customer_id].value_counts()
        id_list = counts[counts > 10][counts < 40].index
        df = df.loc[df[cols.customer_id].isin(id_list)]
        print(len(df))
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            sender_id = row[cols.customer_id]
            text = row[cols.text]
            self.client.send_message(sender_id, text)
