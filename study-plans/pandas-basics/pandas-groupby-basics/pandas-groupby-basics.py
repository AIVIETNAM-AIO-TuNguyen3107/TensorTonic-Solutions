import pandas as pd

def groupby_basics(data, group_col, value_col):
    """
    Returns: dict with 'sum', 'mean', 'count' (each a dict)
    """
    df = pd.DataFrame(data)

    grouped = df.groupby(group_col)

    return grouped[value_col].agg(['sum', 'mean', 'count']).to_dict()