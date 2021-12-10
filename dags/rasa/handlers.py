import logging
from typing import Optional
import hashlib

from pandas import DataFrame
from tqdm import tqdm

from dags.common.utils import ColumnNameSpace, get_columns, DFHandler
from dags.rasa.utils import RASAClient


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


class MergeHandler(DFHandler):
    def __init__(self):
        super().__init__()

    def handle(self, rasa_df: DataFrame, chat_df: DataFrame) -> Optional[DataFrame]:
        cols = self.columns_obj
        customer_ids = rasa_df[cols.customer_id].value_counts().index
        chat_df: DataFrame = chat_df.loc[chat_df[cols.customer_id].isin(customer_ids)]

        text_intent_map = {}
        text_action_map = {}
        last_hash_key = ''
        for i, row in rasa_df.iterrows():
            if isinstance(row[cols.text], str):
                hash_key = hashlib.md5((row[cols.customer_id] + row[cols.text]).encode("utf-8")).hexdigest()
                last_hash_key = hash_key
                text_intent_map[hash_key] = row[cols.intent]
            elif isinstance(row[cols.action_name], str):
                text_action_map[last_hash_key] = row[cols.action_name]

        intent_list = []
        action_list = []
        logging.error(text_intent_map)
        for i, row in chat_df.iterrows():
            hash_key = hashlib.md5((row[cols.customer_id] + row[cols.text]).encode("utf-8")).hexdigest()
            intent = text_intent_map[hash_key] if hash_key in text_intent_map else ''
            action = text_action_map[hash_key] if hash_key in text_action_map else ''
            intent_list.append(intent)
            action_list.append(action)
        chat_df['action'] = action_list
        chat_df['intent'] = intent_list

        voice_counts = chat_df[cols.customer_id].loc[chat_df['text'] == '<<voice>>'].value_counts()
        customer_ids = voice_counts.loc[voice_counts < 10].index
        chat_df: DataFrame = chat_df.loc[chat_df[cols.customer_id].isin(customer_ids)]
        chat_df = chat_df.loc[:, ['customer_id', 'timestamp', 'speaker', 'text', 'action', 'intent']]
        return chat_df
