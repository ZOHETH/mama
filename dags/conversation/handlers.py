import difflib
import re
from typing import Optional, List
from collections import deque

from emoji import UNICODE_EMOJI
from pandas import DataFrame
import numpy as np
from tqdm import tqdm

from common.utils import DFHandler, DFWorker, mapreduce
from conversation.core import Conversation


def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()


def is_mess(text):
    text = str(text)
    if text.count('\\') > 2:
        return True
    return False


class MessTextWorker(DFWorker):
    def map(self, q):
        df = self.input_data
        s_que = deque(maxlen=100)
        mess_i = set()
        # 通过长句是否与别人重复 判断哪些是假客户
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            if len(row['text']) > 60:
                for s in s_que:
                    if string_similar(
                            s[0],
                            row['text']) > 0.95:
                        mess_i.add(i)
                        mess_i.add(s[1])
                s_que.append((row['text'], i))
                if is_mess(row['text']):
                    mess_i.add(i)
        q.put(mess_i)
        self.result = mess_i

    def reduce(self, other):
        self.result = self.result | other.result.value


class MessTextHandler(DFHandler):
    def __init__(self):
        super().__init__()

    def handle(self, df: DataFrame) -> Optional[DataFrame]:
        col = self.columns_obj
        df.sort_values(by=[col.timestamp], inplace=True)
        # 转换为str 防止数字数据
        df[col.text] = df[col.text].apply(str)

        mess_i = mapreduce(MessTextWorker, df, config={'n': 40})
        print('replace start------------------------------')
        df.loc[df.index.isin(mess_i), col.text] = '<<mess_text>>'
        print('sort start---------------------------------')
        df.sort_values(by=[col.customer_id, col.timestamp], inplace=True)

        return df


class SmallSampleHandler(DFHandler):
    def __init__(self, n=500, sample=False):
        super().__init__()
        self.sample = sample
        self.n = n

    def handle(self, df: DataFrame) -> Optional[DataFrame]:
        if self.sample:
            return df.sample(self.n)
        return df.head(self.n)


class NonTextHandler(DFHandler):
    def __init__(self):
        super().__init__()

    def handle(self, df: DataFrame) -> Optional[DataFrame]:
        col = self.columns_obj
        df[col.text] = df[col.text].apply(str)
        data = []
        # session_no = 0
        # prev_timestamp = 0
        cur_customer = None
        # customer_msg_count = 0
        mess_count = 0
        voice_count = 0
        for i, row in tqdm(df.iterrows(), total=df.shape[0], mininterval=5):
            text = row[col.text]
            if row[col.customer_id] != cur_customer:
                cur_customer = row[col.customer_id]
                mess_count, voice_count = 0, 0
            #     prev_timestamp = row[col.timestamp]
            # if row[col.speaker] == 0:
            #     customer_msg_count += 1
            if text.startswith('<<'):
                if text == '<<mess_text>>':
                    mess_count += 1
                elif text in ('<<voice>>'):
                    voice_count += 1
            elif '您的服务已升级' in text or '我是德华安顾' in text:
                mess_count += 1
            else:
                patten = re.compile(
                    r'https?:?/?/(?:[-\w.//?=&]|(?:%[\da-fA-F]{2}))+')
                text = re.sub(patten, '', text)
                data.append([row[col.customer_id],
                             row[col.timestamp],
                             row[col.speaker],
                             text.replace('\\\\n', ''),
                             mess_count,
                             voice_count])
                mess_count, voice_count = 0, 0
        df = DataFrame(data,
                       columns=[col.customer_id, col.timestamp, col.speaker, col.text, 'mess_count', 'voice_count'])
        return df


class SummaryHandler(DFHandler):
    def __init__(self):
        super().__init__()

    def create_summary_df(self, convs: List[Conversation]):
        col = self.columns_obj
        data = []
        for conv in convs:
            data.append([conv.index, conv.length, conv.num_c_msg,
                        conv.num_s_msg, conv.percent_c_msg])
        df = DataFrame(data, columns=[
            col.customer_id, col.length, col.num_c_msg, col.num_s_msg, col.percent_c_msg])
        return df

    def handle(self, df: DataFrame) -> Optional[DataFrame]:
        col = self.columns_obj
        df[col.text] = df[col.text].apply(str)
        df.set_index(col.customer_id, inplace=True)
        ids = df.index.drop_duplicates()
        convs = []
        for index in ids:
            conv = Conversation(index, df.loc[index])
            convs.append(conv)

        s_df = self.create_summary_df(convs)
        return s_df


class PremiumConversationHandler(DFHandler):
    """
    file1: summary csv
    file2: 
    """

    def __init__(self, sample=100):
        super().__init__()
        self.sample = sample//2

    def handle(self, *df_list: DataFrame) -> Optional[DataFrame]:
        col = self.columns_obj
        df1, df2 = df_list
        n = len(df1)//5  # Pareto principle
        pool1 = df1.nlargest(n, columns=[col.num_c_msg])
        pool2 = df1.nlargest(n, columns=[col.percent_c_msg])
        ids1 = pool1.sample(self.sample)[col.customer_id].values
        ids2 = pool2.sample(self.sample)[col.customer_id].values
        ids = list(ids1)
        ids.extend(list(ids2))
        return df2.loc[df2[col.customer_id].isin(ids)]



if __name__ == '__main__':
    from common.operators import CSVOperator

    a = CSVOperator(task_id='test',
                    read_filenames=[
                        '/home/yangkaixuan/datafile/airflow/nlg_preprocess/2021-12-01/without_mess.csv.copy'],
                    handler=NonTextHandler(),
                    write_filename='result.csv')
    a.execute({'data_path': '/tmp'})
