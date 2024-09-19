import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('scopus_results.csv')

# Drop duplicates based on a specific column, e.g., 'dc:title'
df.drop_duplicates(subset='dc:title', inplace=True)

# Handle missing values if necessary
df.fillna('', inplace=True)

# Save the cleaned DataFrame back to CSV
df.to_csv('scopus_results_cleaned.csv', index=False)
print(f"Saved cleaned records to scopus_results_cleaned.csv")
