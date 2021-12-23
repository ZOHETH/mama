import time

from airflow.decorators import dag
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator

from common.operators import CSVOperator, SqliteOperator
from rasa.utils import RASAClient
from rasa.handlers import UploadConvDataHandler, MergeHandler

default_args = {
    'owner': 'zoheth',
}


@dag(default_args=default_args, schedule_interval=None, start_date=days_ago(1))
def rasa_upload_conversation():
    rasa_client = RASAClient()
    clean = PythonOperator(task_id='clean',
                           python_callable=rasa_client.clean_conversations)
    upload = CSVOperator(task_id='upload',
                         read_filenames=['/home/yangkaixuan/train_conversation.csv'],
                         handler=UploadConvDataHandler(rasa_client,
                                                       lower_bound=10,
                                                       upper_bound=40,
                                                       sample_n=100))
    fetch_from_rasa = SqliteOperator(task_id='fetch_from_rasa', db_path='/home/yangkaixuan/repo/rasa-demo/rasa.db',
                                     select_sql="""
select ce.conversation_id customer_id, text, intent, action_name
from conversation_event ce left join message_log ml on ce.id = ml.event_id
where action_name not in ('action_session_start', 'action_listen') or intent_name is not null
                   """,
                                     write_filename="rasa.csv")
    delay_task: PythonOperator = PythonOperator(task_id="delay_python_task",
                                                python_callable=lambda: time.sleep(10))

    merge = CSVOperator(task_id='merge',
                        read_filenames=['rasa.csv', '/home/yangkaixuan/rasa_dia/2021-10-15/collated.csv'],
                        write_filename='res.csv',
                        handler=MergeHandler())

    clean >> upload >> delay_task >> fetch_from_rasa >> merge


# rasa_upload_conversation_dag = rasa_upload_conversation()
