from typing import TYPE_CHECKING

import pandas as pd
from tqdm import tqdm

from common.utils import ColumnNameSpace, get_columns
from custom_operator.utils import DFHandler

if TYPE_CHECKING:
    from common.rasa import RASAClient
    from pandas import DataFrame


class UploadConvDataHandler(DFHandler):
    """

    """

    def __init__(self, client: RASAClient, lower_bound=None, sample_n=None, upper_bound=None, speaker=None):
        """

        Args:
            client:
            lower_bound:
            upper_bound:
            speaker:
        """
        super().__init__()
        self.client = client
        necessary_columns = ('customer_id', 'text')
        self.columns_obj: ColumnNameSpace = get_columns(necessary_columns)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.speaker = speaker
        self.sample_n = sample_n

    def handle(self, df: DataFrame):
        cols = self.columns_obj
        counts = df[cols.customer_id].value_counts()
        ids = counts[counts > self.lower_bound][counts < self.upper_bound]
        if self.sample_n is not None:
            ids = ids.sample(self.sample_n)
        id_list = ids.index
        df = df.loc[df[cols.customer_id].isin(id_list)]
        print(len(df))
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            sender_id = row[cols.customer_id]
            text = row[cols.text]
            self.client.send_message(sender_id, text)
