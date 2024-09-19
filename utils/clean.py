import logging

class IgnoreSpecificLogsFilter(logging.Filter):
    def filter(self, record):
        # Filter out specific HTTP request patterns
        if "GET /progress" in record.getMessage() or \
           "GET https://api.elsevier.com/content/abstract/doi/" in record.getMessage():
            return False
        return True

def clean_text(df):
    # Drop duplicates based on a specific column, e.g., 'dc:title'
    df.drop_duplicates(subset='dc:title', inplace=True)

    # Handle missing values if necessary
    df.fillna('', inplace=True)

    # Remove @_fa, link columns if they exist
    if '@_fa' in df.columns:
        df.drop(columns=['@_fa'], inplace=True)
    if 'link' in df.columns:
        df.drop(columns=['link'], inplace=True)

    return df
    # print(f"Saved cleaned records to scopus_results.json")
