from airflow.decorators import dag
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator

from custom_operator.csv import CSVOperator
from common.rasa import RASAClient
from tasks.rasa import UploadConvDataHandler

default_args = {
    'owner': 'zoheth',
}


@dag(default_args=default_args, schedule_interval=None, start_date=days_ago(1))
def rasa_upload_conversation():
    rasa_client = RASAClient()
    clean = PythonOperator(task_id='clean',
                           python_callable=rasa_client.clean_conversations)
    upload = CSVOperator(task_id='upload',
                         handler=UploadConvDataHandler(rasa_client,
                                                       lower_bound=10,
                                                       upper_bound=40,
                                                       sample_n=10),
                         read_filename='/home/yangkaixuan/train_conversation.csv')
    clean >> upload


rasa_upload_conversation()
