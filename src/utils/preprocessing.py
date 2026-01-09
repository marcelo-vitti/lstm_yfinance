from sklearn.preprocessing import MinMaxScaler


def scale_train_test(train_df, test_df):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    return train_scaled, test_scaled, scaler
