import pandas as pd

def get_counts(data, field, sep):
    # returns the count of each element in a series of lists seperated by sep
    return data[field].apply(lambda s: str(s).split(sep)).apply(pd.Series).melt(value_name='counts')['counts'].value_counts().sort_values(ascending=False)
