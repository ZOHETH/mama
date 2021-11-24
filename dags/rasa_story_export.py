from airflow.decorators import dag
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'zoheth',
}


@dag(default_args=default_args, schedule_interval=None, start_date=days_ago(1))
def rasa_story_export():
    pass
