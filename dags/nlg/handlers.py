from typing import Optional

from pandas import DataFrame
from tqdm import tqdm

from common.utils import DFHandler


class PretrainingTextHandler(DFHandler):
    def __init__(self):
        super().__init__()

    def handle(self, df: DataFrame) -> Optional[DataFrame]:
        col = self.columns_obj
        # 转换为str 防止数字数据
        df[col.text] = df[col.text].apply(str)
        data = []
        cur_customer = None
        last_timestamp = 1700000000
        for i, row in tqdm(df.iterrows(), total=df.shape[0], mininterval=5):
            text = row[col.text]
            if row[col.speaker] == 0:
                prefix = '客户：'
            else:
                prefix = '销售：'
            if row[col.timestamp] - last_timestamp > 3600 or row['mess_count'] + row['voice_count'] > 5:
                if row[col.customer_id] == cur_customer:
                    data[-1] = data[-1] + '<eop>'
            if row[col.customer_id] != cur_customer:
                if cur_customer is not None:
                    data.append('<eod>')
                cur_customer = row[col.customer_id]
            data.append(prefix + text)
            last_timestamp = row[col.timestamp]
        df = DataFrame(data, columns=['data'])

        return df
