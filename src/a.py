import sqlite3
import pandas as pd

customer_sample_df = pd.read_csv('sample.csv')
df = pd.read_csv('/home/yangkaixuan/rasa_dia/2021-10-15/collated.csv')
df = df.loc[df['customer_id'].isin(customer_sample_df['conversation_id'])]
ids = tuple(customer_sample_df['conversation_id'].values)
ids_str = str(ids)

connection=sqlite3.connect('rasa.db')
intent_df = pd.read_sql_query(f"select text, intent, conversation_id, time from message_log where conversation_id in {ids_str};", connection)