class DFHandler:
    def __init__(self, df):
        self.df = df

    def execute(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        self.execute(*args, **kwargs)
