import pandas as pd

def apply_transform(data, column, operation):
    """
    Returns: dict with original columns plus column_transformed
    """
    df = pd.DataFrame(data)
    if operation=='normalize':
        min = df[column].min()
        max = df[column].max()
        df[f'{column}_transformed'] = df[column].apply(
            lambda x: round((x - min) / (max - min), 4)
        )
    elif operation == 'rank':
        df[f'{column}_transformed'] = df[column].rank()
    elif operation == 'cumsum':
        df[f'{column}_transformed'] = df[column].cumsum()
    elif operation == 'double':
        df[f'{column}_transformed'] = df[column] * 2
    else:
        df[f'{column}_transformed'] = df[column].apply(operation)

    return df.to_dict("list")