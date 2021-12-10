from airflow.decorators import dag
from airflow.utils.dates import days_ago

from common.operators import CSVOperator
from conversation.handlers import MessTextHandler, SmallSampleHandler, NonTextHandler
from nlg.handlers import PretrainingTextHandler

default_args = {
    'owner': 'zoheth',
}


@dag(default_args=default_args, schedule_interval=None, start_date=days_ago(1))
def nlg_preprocess():
    small = CSVOperator(task_id='sample',
                        read_filenames=['/home/yangkaixuan/download/all_message1130.tsv'], sep='\t',
                        write_filename='sample.csv',
                        handler=SmallSampleHandler(n=200000))

    clean_mess = CSVOperator(task_id='clean_mess',
                             read_filenames=['/home/yangkaixuan/download/all_message1130.tsv'], sep='\t',
                             # read_filenames=['sample.csv'],
                             write_filename='without_mess.csv',
                             handler=MessTextHandler())

    clean_non_text = CSVOperator(task_id='clean_non_text',
                                 read_filenames=['without_mess.csv'],
                                 handler=NonTextHandler(),
                                 write_filename='only_text.csv')

    train_data = CSVOperator(task_id='train_data',
                             read_filenames=[
                                 '/home/yangkaixuan/datafile/airflow/nlg_preprocess/2021-12-01/only_text.csv'],
                             handler=PretrainingTextHandler(),
                             write_filename='train_data.csv')
    clean_mess >> clean_non_text >> train_data


nlg_preprocess_dag = nlg_preprocess()
