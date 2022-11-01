import pandas as pd
from scipy.stats import gaussian_kde


def resample(df_label, n_sample):
    categorical_feats = []
    numeric_feats = []
    for col in df_label.columns:
        if df_label[col].nunique() < 10:
            categorical_feats.append(col)
        else:
            numeric_feats.append(col)

    kernel = gaussian_kde(df_label.T)
    synt = kernel.resample(n_sample)
    synt = pd.DataFrame(synt.T, columns=df_label.columns)
    mins = df_label.min()
    maxs = df_label.max()
    for col, col_ty in zip(df_label.columns, df_label.dtypes):
        synt.loc[synt[synt[col] < mins[col]].index, [col]] = mins[col]
        synt.loc[synt[synt[col] > maxs[col]].index, [col]] = maxs[col]
        synt[col] = synt[col].astype(col_ty)

    return synt


def resample_cat(df_label, cat_cols, n_sample):

    kernel = gaussian_kde(df_label.T)
    synt = kernel.resample(n_sample)
    synt = pd.DataFrame(synt.T, columns=df_label.columns)
    for col in cat_cols:
        synt[col] = (synt[col]).astype(int)
    mins = df_label.min()
    maxs = df_label.max()
    for col in df_label.columns:
        synt[synt[col] < mins[col]] = mins[col]
        synt[synt[col] > maxs[col]] = maxs[col]
    return synt[cat_cols]


def create_data_like(dataset_path, target_col, n_samples):
    df = pd.read_csv(dataset_path)

    class_balance = df[target_col].value_counts() / len(df)
    synt = pd.DataFrame()

    target = []
    for label in class_balance.index:

        n_label = int(class_balance[label] * n_samples)
        if len(synt) + n_label > n_samples:
            n_label = n_samples - len(synt)
        target += [label] * n_label
        df_label = df.loc[df[target_col] == label].drop(target_col, axis=1)
        synt = synt.append(resample(df_label, n_label))

    synt = pd.concat((synt.reset_index(), pd.DataFrame(target, columns=[target_col])), axis=1)
    return synt.drop(columns=['index'])


def check_viz(data, data_like):
    pass  # TODO: viz


if __name__ == '__main__':
    dataset_path = r"..\..\data\raw\heart_cleveland_upload.csv"
    df_new = create_data_like(dataset_path, 'condition', 50)
    print(df_new)
