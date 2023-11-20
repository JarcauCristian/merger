import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime


def read_datasets(first_path: str, second_path: str) -> (pd.DataFrame, pd.DataFrame):
    df_1 = pd.read_csv(first_path)
    df_2 = pd.read_csv(second_path)

    return df_1, df_2


def extract_features(df: pd.DataFrame):
    columns = list(df.columns)

    column_types = {}

    for column in columns:
        if df[column].dtype.name == "object" and len(df[column].unique()) > len(df) / 2:
            column_types[column] = "datetime"
        else:
            column_types[column] = df[column].dtype.name

    unique_values = {}

    for k, v in column_types.items():
        if v == 'object':
            unique_values[k] = df[k].unique().tolist()
        elif v.find("int") != -1 or v.find("float") != -1:
            col_min = df[k].min()
            col_max = df[k].max()
            unique_values[k] = (col_min, col_max)

    description = [
        "Date of the Observation in",
        "The temperature at 9 am.",
        "The temperature at 3 pm.",
        "The Minimum temperature during a particular day.",
        "The maximum temperature during a particular day.",
        "Rainfall during a particular day.",
        "If today is rainy then ‘Yes’. If today is not rainy then ‘No’.",
        "Evaporation during a particular day.",
        "Bright sunshine during a particular day.",
        "The direction of the strongest gust during a particular day. (16 compass points)",
        "The direction of the strongest gust during a particular day.",
        "Speed of strongest gust during a particular day."
        "The direction of the wind for 10 min prior to 9 am.",
        "The direction of the wind for 10 min prior to 3 pm.",
        "Speed of the wind for 10 min prior to 9 am."
        "Speed of the wind for 10 min prior to 3 pm."
        "The humidity of the wind at 9 am."
        "The humidity of the wind at 3 pm.",
        "Atmospheric pressure at 9 am. ",
        "Atmospheric pressure at 3 pm.",
        "Cloud-obscured portions of the sky at 9 am."
        "Cloud-obscured portions of the sky at 3 pm."
    ]

    description = [
        "Unique ID of observations.",
        "Name of the city from Australia.",
        "The minimum temperature during a particular day.",
        "The maximum temperature during a particular day.",
        "Rainfall during a particular day.",
        "Evaporation during a particular day.",
        "Bright sunshine during a particular day.",
        "The direction of the strongest gust during a particular day. (16 compass points)",
        "Speed of strongest gust during a particular day.",
        "The direction of the wind for 10 min prior to 9 am.",
        "The direction of the wind for 10 min prior to 3 pm.",
        "Speed of the wind for 10 min prior to 9 am."
        "Speed of the wind for 10 min prior to 3 pm."
        "The humidity of the wind at 9 am."
        "The humidity of the wind at 3 pm.",
        "Atmospheric pressure at 9 am. ",
        "Atmospheric pressure at 3 pm.",
        "Cloud-obscured portions of the sky at 9 am.",
        "Cloud-obscured portions of the sky at 3 pm.",
        "The temperature at 9 am.",
        "The temperature at 3 pm."
        "If today is rainy then ‘Yes’. If today is not rainy then ‘No’."
    ]

    return column_types


d1, d2 = read_datasets("Weather_Data.csv", "Weather Test Data.csv")

print(extract_features(d1))
