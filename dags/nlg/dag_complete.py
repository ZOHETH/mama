from airflow.decorators import dag
from airflow.utils.dates import days_ago

from common.operators import CSVOperator
from conversation.handlers import MessTextHandler, SmallSampleHandler, NonTextHandler, PremiumConversationHandler, SummaryHandler
from nlg.handlers import PretrainingTextHandler, PretrainingLongTextHandler, BenchmarkHandler, IdsHandler

default_args = {
    'owner': 'zoheth',
}

db_data_file = '/data/all_message1130.tsv'


@dag(default_args=default_args, schedule_interval=None, start_date=days_ago(1))
def nlg_preprocess():
    small = CSVOperator(task_id='sample',
                        read_filenames=[db_data_file], sep='\t',
                        write_filename='sample.csv',
                        handler=SmallSampleHandler(n=200000))

    clean_mess = CSVOperator(task_id='clean_mess',
                             read_filenames=[db_data_file], sep='\t',
                             # read_filenames=['sample.csv'],
                             write_filename='without_mess.csv',
                             handler=MessTextHandler())

    clean_non_text = CSVOperator(task_id='clean_non_text',
                                 read_filenames=['without_mess.csv'],
                                 handler=NonTextHandler(),
                                 write_filename='only_text.csv')

    summary = CSVOperator(task_id='summary',
                          read_filenames=['only_text.csv'],
                          handler=SummaryHandler(),
                          write_filename='summary.csv')

    extract = CSVOperator(task_id='extract',
                          read_filenames=['summary.csv', 'only_text.csv'],
                          handler=PremiumConversationHandler(),
                          write_filename='premium.csv')
    benchmark = CSVOperator(task_id='benchmark',
                            read_filenames=['premium.csv'],
                            handler=BenchmarkHandler('/data/mymodel1'),
                            write_filename='benchmark.csv')
    train_data = CSVOperator(task_id='train_data',
                             read_filenames=['only_text.csv'],
                             handler=PretrainingTextHandler(),
                             write_filename='train_data.csv')

    train_data = CSVOperator(task_id='train_data_long',
                             read_filenames=['only_text.csv', 'summary.csv'],
                             handler=PretrainingLongTextHandler(),
                             write_filename='train_data_long.csv')

    ids_data = CSVOperator(task_id='ids_data',
                           read_filenames='train_data.csv',
                           handler=IdsHandler,
                           write_filename='ids.csv')

    clean_mess >> clean_non_text >> train_data
    clean_non_text >> summary >> extract >> benchmark

    # clean_non_text >> summary
    # clean_non_text >> train_data
    # extract >> benchmark


nlg_preprocess_dag = nlg_preprocess()
