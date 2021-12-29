from typing import Optional
from collections import deque

from pandas import DataFrame
from tqdm import tqdm

import sys
sys.path.append('.')
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


class BenchmarkHandler(DFHandler):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path

    def get_input(self, contexts, tokenizer):
        inputs = []
        length = 0
        for ctx in contexts:
            length += len(ctx)
        while length > 200:
            x = contexts.popleft()
            length -= len(x)
        for ctx in contexts:
            inputs.append(tokenizer.encode(ctx, return_tensors='tf'))
        prompt = "销售："
        inputs.append(tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors='tf'))
        import tensorflow as tf
        inputs = tf.concat(inputs, 1)
        return inputs

    def handle(self, df: DataFrame) -> Optional[DataFrame]:
        from transformers import AutoTokenizer
        from transformers.models.xlnet.modeling_tf_xlnet import TFXLNetLMHeadModel
        model = TFXLNetLMHeadModel.from_pretrained(self.model_path)
        model.transformer.attn_type = 'bi'
        tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-xlnet-base")
        pretraining_text_handler = PretrainingTextHandler()
        df = pretraining_text_handler.handle(df)
        contexts = deque()
        data = []
        for i, row in tqdm(df.iterrows(), total=df.shape[0], mininterval=5):
            if row['data'] == '<eod>':
                contexts.clear()
                data.append('')
                continue
            contexts.append(row['data'])
            if row['data'][0] == '客' and len(contexts) > 0:
                inputs = self.get_input(contexts, tokenizer)
                leng = inputs.shape[-1]
                print(leng)
                outputs = model.generate(inputs,
                                         num_beams=20, 
                                         max_length=leng+20,
                                         do_sample=True, top_p=0.95,
                                         num_return_sequences=3,
                                         no_repeat_ngram_size=3,
                                         early_stopping=True)
                result = []
                for i, sample_output in enumerate(outputs):
                    generated = tokenizer.decode(
                        sample_output[leng:], skip_special_tokens=False)
                    result.append(f'{i}: {generated}')
                data.append('\n'.join(result))
            else:
                data.append('')
        df.insert(1, 'nlg', data)
        return df


if __name__ == '__main__':
    import pandas as pd
    bh = BenchmarkHandler('/home/yangkaixuan/eden/transformer/mymodel1')
    df = pd.read_csv(
        '/home/yangkaixuan/datafile/airflow/nlg_preprocess/2021-12-22/premium.csv')
    df = bh.handle(df)
    df.to_csv('result.csv')
