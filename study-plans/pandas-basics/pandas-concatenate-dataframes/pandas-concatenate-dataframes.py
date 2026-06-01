import pandas as pd

def concat_dataframes(dfs):
    """
    Returns: list [shape, data] where shape is [rows, cols]
    """
    dfs = [pd.DataFrame(data) for data in dfs]
    dfs = pd.concat(dfs)

    return [list(dfs.shape), dfs.to_dict('list')]