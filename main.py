import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn.cluster import KMeans, Birch, BisectingKMeans
from rapidfuzz.fuzz import ratio
from sklearn.feature_extraction.text import TfidfVectorizer


def read_datasets(first_path: str, second_path: str) -> (pd.DataFrame, pd.DataFrame):
    df_1 = pd.read_csv(first_path)
    df_2 = pd.read_csv(second_path)

    return df_1, df_2


def extract_features(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    columns = list(df.columns)

    column_types = {}
    means = {}
    stds = {}
    for column in columns:
        if len(df[column].unique()) == len(df) - 2:
            column_types[column] = "unique identifier"
        else:
            column_types[column] = df[column].dtype.name
            if pd.api.types.is_numeric_dtype(df[column]):
                means[column] = df[column].mean()
                stds[column] = df[column].std()
            else:
                means[column] = 0
                stds[column] = 0

    unique_values = {}

    for k, v in column_types.items():
        if v == 'object' or v == "unique identifier":
            arr = df[k].unique().tolist()
            arr = [a for a in arr if isinstance(a, str)]
            unique_values[k] = ','.join(arr)
        elif v.find("int") != -1 or v.find("float") != -1:
            col_min = df[k].min()
            col_max = df[k].max()
            unique_values[k] = (col_min, col_max)

    rows = []
    for column in columns:
        rows.append({
            "column_name": column,
            "column_dtype": column_types[column],
            "min_value": unique_values[column][0] if isinstance(unique_values[column], tuple) else None,
            "max_value": unique_values[column][1] if isinstance(unique_values[column], tuple)else None,
            "mean": means[column],
            "std": stds[column],
            "unique_values": unique_values[column] if not isinstance(unique_values[column], tuple) else "",
            "file_prov": file_name
        })

    return pd.DataFrame(rows)


def prepare_features(concat_features: pd.DataFrame):
    numerical_features = concat_features[['min_value', 'max_value', 'mean', 'std']].fillna(0)
    scaler = StandardScaler()
    x_numerical = scaler.fit_transform(numerical_features)
    x_numerical_sparse = csr_matrix(x_numerical)

    vectorizer = TfidfVectorizer(stop_words='english')

    x_names = vectorizer.fit_transform(concat_features["column_name"].tolist())
    x_dtypes = vectorizer.fit_transform(concat_features["column_dtype"].tolist())
    if len(concat_features["unique_values"].tolist()[0]) > 0:
        x_unique_values = vectorizer.fit_transform(concat_features["unique_values"].tolist())

        return hstack([x_names, x_dtypes, x_unique_values, x_numerical_sparse])
    else:
        return hstack([x_names, x_dtypes, x_numerical_sparse])


def look_through_clusters(clusters: pd.DataFrame, df1: pd.DataFrame, file_name1: str, file_name2: str, df2: pd.DataFrame):
    cluster_numbers = clusters["cluster"].unique().tolist()
    single_columns = []
    for cluster_number in cluster_numbers:
        values_in_cluster = clusters.loc[clusters["cluster"] == cluster_number]
        if len(values_in_cluster) == 1:
            if values_in_cluster["column_dtype"].values[0] == "unique identifier":
                continue
            else:
                single_columns.append(values_in_cluster["column_name"])
        elif len(values_in_cluster) > 2:
            km = BisectingKMeans(n_clusters=2, n_init=10, random_state=42)
            prep_features = prepare_features(values_in_cluster)
            km.fit(prep_features)
            cl_data = pd.DataFrame({
                'column_name': values_in_cluster["column_name"],
                'column_dtype': values_in_cluster["column_dtype"],
                'min_value': values_in_cluster['min_value'],
                'max_value': values_in_cluster['max_value'],
                'mean': values_in_cluster['mean'],
                'std': values_in_cluster['std'],
                'unique_values': values_in_cluster['unique_values'],
                'file_prov': values_in_cluster['file_prov'],
                'cluster': km.labels_
            })

        else:
            print(values_in_cluster)
            if values_in_cluster["file_prov"].tolist()[0] == file_name1:
                c1 = df1[values_in_cluster["column_name"].tolist()[0]]
                c2 = df2[values_in_cluster["column_name"].tolist()[1]]
            else:
                c1 = df1[values_in_cluster["column_name"].tolist()[1]]
                c2 = df2[values_in_cluster["column_name"].tolist()[0]]

            new_pd = [*c1.tolist(), *c2.tolist()]
            new_pd = pd.Series(new_pd)
            print(new_pd)


file_name_d1 = "Weather Test Data.csv"
file_name_d2 = "Weather_Data.csv"
d1, d2 = read_datasets(file_name_d1, file_name_d2)

extracted_features_d1 = extract_features(d1, file_name_d1)
extracted_features_d2 = extract_features(d2, file_name_d2)

concat = extracted_features_d1.merge(extracted_features_d2, how="outer")

prepared_features = prepare_features(concat_features=concat)

model = BisectingKMeans(n_clusters=max(len(d1.columns), len(d2.columns)), n_init=10, random_state=42)

model.fit(prepared_features)
labels_combined_all = model.labels_

clustered_data_combined_all = pd.DataFrame({
    'column_name': concat["column_name"],
    'column_dtype': concat["column_dtype"],
    'min_value': concat['min_value'],
    'max_value': concat['max_value'],
    'mean': concat['mean'],
    'std': concat['std'],
    'unique_values': concat['unique_values'],
    'file_prov': concat['file_prov'],
    'cluster': labels_combined_all
})
clustered_data_combined_all.sort_values('cluster', inplace=True)
print(clustered_data_combined_all)
look_through_clusters(clustered_data_combined_all, d1, file_name_d1, file_name_d2, d2)
