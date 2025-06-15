from sklearn.feature_extraction import DictVectorizer

def transform_data(df):
    categorical = ['PULocationID', 'DOLocationID']
    train_dicts = df[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df['duration'].values

    print(f'Shape de X_train: {X_train.shape}')
    return X_train, y_train, dv
