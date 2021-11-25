from collections import namedtuple

columns = ['customer_id', 'text', 'speaker']
ColumnNameSpace = namedtuple('ColumnNameSpace', columns)


def get_columns(necessary_columns=tuple(columns)):
    NecessaryColumnNameSpace = namedtuple('NecessaryColumnNameSpace', necessary_columns)
    return NecessaryColumnNameSpace(*necessary_columns)
