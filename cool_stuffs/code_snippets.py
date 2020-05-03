import numpy as np
import pandas as pd


# returns the count of each element in a series of lists seperated by sep
def get_counts(data, field, sep):
    return data[field].apply(lambda s: str(s).split(sep)).apply(pd.Series).melt(value_name='counts')['counts'].value_counts().sort_values(ascending=False)


# duplicate of add_datepart in fastai
def add_datepart(df, fldnames, drop=True, time=False, errors="raise"):
    for fldname in fldnames:
        fld = df[fldname]
        if isinstance(fld.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64

        if not np.issubdtype(fld.dtype, np.datetime64):
            df[fldname] = fld = pd.to_datetime(
                fld, infer_datetime_format=True, errors=errors)
        targ_pre = re.sub('[Dd]ate$', '', fldname)
        attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
                'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
        if time:
            attr = attr + ['Hour', 'Minute', 'Second']
        for n in attr:
            df[targ_pre + n] = getattr(fld.dt, n.lower())
        df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
        if drop:
            df.drop(fldname, axis=1, inplace=True)
    return df


def categorify(data, category_list):
    """
    Arguments
    ---------
    data: pd.DataFrame
        DataFrame to be used for transformation

    category_lists: list
        List of categorical columns to be transformed

    Return
    ------
    data: pd.DataFrame
        Categorified DataFrame

    category_mapping: dict
        Mapping of categorical to numerical columns
    """
    category_mapping = {}
    for col in category_list:
        data.loc[:, col] = data.loc[:, col].astype('category').cat.as_ordered()
        category_mapping[col] = data[col].cat.categories
    return data, category_mapping


def fill_missing_cat(data, category_list, add_nan_col=False):
    """
    Fills Missing Values for a list of categorical columns

    Arguments
    ---------
    data: pd.DataFrame
        DataFrame to be used for transformation

    category_list: list
        List of categorical columns to be transformed

    Return
    ------
    data: pd.DataFrame
        DataFrame with missing values filled by mode

    category_mapping: dict
        Mapping of categorical columns with the filled values
    """
    category_mapping = {}
    for col in category_list:
        if add_nan_col:
            data[col+'_na'] = pd.isnull(data[col])
        filler = data[name].dropna().value_counts().idxmax()
        category_mapping[col] = filler
        data[col] = data[col].fillna(filler)
    return data, category_mapping


def fill_missing_cont(data, continous_list, add_nan_col=False, strategy='mean'):
    """
    Fills missing values for continous data

    Arguments
    ---------
    data: pd.DataFrame
        DataFrame to be used for transformation

    continous_list: list
        List of continous columns to be transformed

    Return
    ------
    data: pd.DataFrame
        DataFrame with missing values filled by mean/median

    continous_mapping: dict
        Mapping of categorical columns with the filled values
    """
    continous_mapping = {}
    for col in continous_list:
        if add_nan_col:
            data[col+'_na'] = pd.isnull(data[col])
        if strategy == 'mean'
        filler = data[col].mean()
        elif strategy == 'median':
            filler = data[col].median()
        else:
            filler = np.nan
        continous_mapping[col] = filler
        data[name] = data[name].fillna(filler)
    return data, continous_mapping
