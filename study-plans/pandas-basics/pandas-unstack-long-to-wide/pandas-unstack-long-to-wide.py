import pandas as pd

def unstack_to_wide(data, index_col, columns_col, values_col):
    """
    Returns: dict with index_col plus one key per unique value in columns_col
    """
    df = pd.DataFrame(data)

    df.set_index([index_col, columns_col], inplace=True)

    return df[values_col].unstack().reset_index().to_dict("list")