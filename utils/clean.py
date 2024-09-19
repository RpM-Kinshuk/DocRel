def clean_text(df):
    # Drop duplicates based on a specific column, e.g., 'dc:title'
    df.drop_duplicates(subset='dc:title', inplace=True)

    # Handle missing values if necessary
    df.fillna('', inplace=True)

    return df
    # print(f"Saved cleaned records to scopus_results.csv")
