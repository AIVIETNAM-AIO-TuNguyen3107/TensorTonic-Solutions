import pandas as pd

def multi_groupby(data, group_cols, value_col, aggfunc):
    """
    Returns: dict of lists (flat table with group columns + value column)
    """
    df = pd.DataFrame(data)

    grouped = df.groupby(group_cols)

    agg_applied = grouped[value_col].agg(aggfunc)

    return agg_applied.reset_index().to_dict("list")