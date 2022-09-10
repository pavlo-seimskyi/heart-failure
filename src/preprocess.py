import pandas as pd


def apply_ordinal_mapping(data: pd.DataFrame, colname: str, mapping: dict):
    """
    Replace non-numeric values in a column with numeric ones.

    :param data: data frame
    :param colname: name of the column
    :param mapping: dictionary with {non-numeric: numeric} values to replace
    :return: transformed data frame
    """
    data[colname] = data[colname].map(mapping)
    return data


def one_hot_encode(data: pd.DataFrame, colname: str, drop_first: bool):
    """
    Apply one-hot encoding to a selected column and drop the initial one.

    :param data: data frame
    :param colname: name of the column
    :param drop_first: Whether the first level should be removeds
    :return: transformed data frame
    """
    return pd.concat(
        (
            data.drop(colname, axis=1),
            pd.get_dummies(data=data[colname], drop_first=drop_first, prefix=colname),
        ),
        axis=1,
    )
