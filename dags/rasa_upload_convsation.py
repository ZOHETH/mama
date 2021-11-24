from airflow.decorators import dag
from airflow.utils.dates import days_ago

from custom_operator.csv import CSVOperator
from common.rasa import RASAClient
from tasks.rasa import UploadConvDataHandler

default_args = {
    'owner': 'zoheth',
}


@dag(default_args=default_args, schedule_interval=None, start_date=days_ago(1))
def rasa_upload_conversation():
    rasa_client = RASAClient()
    upload = CSVOperator(task_id='upload',
                         handler=UploadConvDataHandler(rasa_client),
                         read_filename='')
