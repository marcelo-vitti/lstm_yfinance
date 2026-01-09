def validate_raw_data(df):
    assert df.index.is_monotonic_increasing
    assert not df.index.duplicated().any()
    assert (df[["Open", "High", "Low", "Close"]] > 0).all().all()
