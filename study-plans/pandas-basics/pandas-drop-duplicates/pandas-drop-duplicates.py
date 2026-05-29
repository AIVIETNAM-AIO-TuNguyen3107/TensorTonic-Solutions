import pandas as pd

def drop_duplicates(data):
    """
    Returns: list [rows_before, rows_after, cleaned_data]
    """
    df = pd.DataFrame(data)
    df_dedup = df.drop_duplicates()
    return [df.shape[0], df_dedup.shape[0], df_dedup.to_dict("list")]
