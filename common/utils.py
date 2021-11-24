from collections import namedtuple

columns = ['customer_id', 'text']
ColumnNameSpace = namedtuple('ColumnNameSpace', columns)


def get_columns(necessary_columns=columns):
    NecessaryColumnNameSpace = namedtuple('NecessaryColumnNameSpace', necessary_columns)
    return NecessaryColumnNameSpace(*necessary_columns)
