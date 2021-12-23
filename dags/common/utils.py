from abc import ABC
from collections import namedtuple
from typing import Optional
from multiprocessing import Queue

from billiard.context import Process

from pandas import DataFrame
import numpy as np

columns = ['customer_id', 'text', 'speaker', 'intent', 'action_name', 'timestamp','num_c_msg','num_s_msg','percent_c_msg', 'length']
ColumnNameSpace = namedtuple('ColumnNameSpace', columns)


def get_columns(necessary_columns=tuple(columns)):
    NecessaryColumnNameSpace = namedtuple('NecessaryColumnNameSpace', necessary_columns)
    return NecessaryColumnNameSpace(*necessary_columns)


class DFHandler:
    def __init__(self):
        self.columns_obj: ColumnNameSpace = get_columns()

    def handle(self, *df_list: DataFrame) -> Optional[DataFrame]:
        raise NotImplementedError

    def change_column_name(self, default_name, custom_name):
        self.columns_obj._replace(**{default_name: custom_name})


class GenericInputData:
    @classmethod
    def generate_inputs(cls, config):
        raise NotImplementedError


class GenericWorker:
    def __init__(self, input_data):
        self.input_data = input_data
        self.result = None

    def map(self,q):
        raise NotImplementedError

    def reduce(self, other):
        raise NotImplementedError

    @classmethod
    def create_workers(cls, input_class, config):
        raise NotImplementedError


class DFWorker(GenericWorker, ABC):
    @classmethod
    def create_workers(cls, df, config):
        n = config.get('n', 5)
        assert n > 1
        workers = []
        for chunk in np.array_split(df, n):
            workers.append(cls(chunk))
        return workers


def execute(workers):
    q = Queue()
    processes = [Process(target=w.map, args=(q,)) for w in workers]
    for process in processes:
        process.start()
    first, *rest = workers
    print('reduce start---------------------------------------')
    first_result = q.get()
    for worker in rest:
        # first.reduce(worker)
        first_result |= q.get()
    print('reduce end  ---------------------------------------')
    for process in processes:
        process.join()
    return first_result


def mapreduce(worker_class, input_class, config):
    workers = worker_class.create_workers(input_class, config)
    return execute(workers)
