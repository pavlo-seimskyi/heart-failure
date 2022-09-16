import pandas as pd


class Preprocessor:
    def __init__(self):
        pass

    def preprocess(self, data):
        return (
            data.copy()
            .pipe(self.one_hot_encode, colname="Sex", drop_first=True)
            .pipe(self.one_hot_encode, colname="ChestPainType", drop_first=False)
            .pipe(self.one_hot_encode, colname="RestingECG", drop_first=True)
            .pipe(
                self.apply_ordinal_mapping,
                colname="ExerciseAngina",
                mapping={"N": 0, "Y": 1},
            )
            .pipe(
                self.apply_ordinal_mapping,
                colname="ST_Slope",
                mapping={"Down": -1, "Flat": 0, "Up": 1},
            )
        ).select_dtypes(
            ["number"]
        )  # Drop all non-numerical columns

    def apply_ordinal_mapping(self, data: pd.DataFrame, colname: str, mapping: dict):
        """
        Replace non-numeric values in a column with numeric ones.

        :param data: data frame
        :param colname: name of the column
        :param mapping: dictionary with {non-numeric: numeric} values to replace
        :return: transformed data frame
        """
        data[colname] = data[colname].map(mapping)
        return data

    def one_hot_encode(self, data: pd.DataFrame, colname: str, drop_first: bool):
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
                pd.get_dummies(
                    data=data[colname], drop_first=drop_first, prefix=colname
                ),
            ),
            axis=1,
        )
