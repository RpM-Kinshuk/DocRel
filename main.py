from itertools import count
import time
import httpx
import logging
import requests
import pandas as pd
from utils.sts import similarity_scores
from flask import Flask, render_template, request, jsonify
from utils.clean import clean_text, IgnoreSpecificLogsFilter

app = Flask(__name__)

API_KEY = '89072218d14bbda5d9acdad8d14bcfbd'
HEADERS = {
    'X-ELS-APIKey': API_KEY,
    'Accept': 'application/json'
}
BASE_URL = 'https://api.elsevier.com/content/search/scopus'

# Global variable to track progress
progress = 0

def get_results_for_year(keywords, year, results_per_year, total_years):
    global progress
    results = []
    query = ' OR '.join([f'TITLE-ABS-KEY({kw})' for kw in keywords])
    
    params = {
        'query': f'({query}) AND PUBYEAR = {year}',
        'count': min(25, results_per_year),  # Use 25 as the max per request
        'start': 0,
        'sort': 'rowTotal'
    }
    
    while True:
        response = requests.get(BASE_URL, headers=HEADERS, params=params)
        # print(f"Request URL: {response.url}") 
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break

        data = response.json()
        entries = data.get('search-results', {}).get('entry', [])
        results.extend(entries)

        total_results = int(data.get('search-results', {}).get('opensearch:totalResults', 0))
        start_index = int(data.get('search-results', {}).get('opensearch:startIndex', 0))
        items_per_page = int(data.get('search-results', {}).get('opensearch:itemsPerPage', 0))

        # Update progress
        progress_step = 50 / (results_per_year * total_years)
        progress = min(100, progress + progress_step)

        # Check if we have retrieved enough results for the year or if we reached the API limits
        if start_index + items_per_page >= total_results or len(results) >= results_per_year:
            break

        # Continue from the next start index
        params['start'] = start_index + items_per_page
        time.sleep(1)  # Sleep to avoid rate-limiting or throttling issues

    return results[:results_per_year]  # Return only the specified number of results


def get_abstract(doi, api_key, max_retries=3, backoff_factor=0.3):
    headers = {
        'X-ELS-APIKey': api_key,
        'Accept': 'application/json'
    }
    url = f"https://api.elsevier.com/content/abstract/doi/{doi}"

    for attempt in range(max_retries):
        try:
            response = httpx.get(url, headers=headers, timeout=30.0)
            if response.status_code == 200:
                data = response.json()
                coredata = data.get('abstracts-retrieval-response', {}).get('coredata', {})
                
                # Attempt to get the abstract (dc:description)
                abstract = coredata.get('dc:description', None)
                
                # Fallback to prism:teaser or dc:title if no abstract is available
                if abstract:
                    return abstract
                else:
                    teaser = coredata.get('prism:teaser', None)
                    if teaser:
                        return teaser
                    title = coredata.get('dc:title', 'No abstract or title available')
                    return f"No abstract available. Title: {title}"

            elif response.status_code == 404:
                print(f"DOI not found: {doi}")
                return 'DOI not found'
            elif response.status_code == 401:
                print(f"Authentication failed for DOI: {doi}")
                return 'Authentication failed'
            elif response.status_code == 429:
                print("Quota exceeded, please try again later.")
                return 'Quota exceeded'
            else:
                print(f"Error {response.status_code} for DOI: {doi}")
                return f"Error {response.status_code}"
        except httpx.ReadTimeout:
            print(f"Read timeout for DOI: {doi}, attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(backoff_factor * (2 ** attempt))  # Exponential backoff
            else:
                return 'Error fetching content due to timeout'
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    return 'Error fetching content'


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/semantic', methods=['GET', 'POST'])
def semantic():
    if request.method == 'POST':
        query = str(request.form['query'])
        selected_model = request.form['model']  # Get the selected model
        top_n = request.form['top_n']  # Get the number of results to return
        sim_measure = request.form['sim_measure']  # Get the similarity measure
        top_results = perform_semantic_analysis(query, top_n, selected_model, sim_measure)
        return jsonify({'results': top_results})
    return render_template('semantic.html')

def perform_semantic_analysis(query, top_n, model, sim_measure):
    # Mocking results for each model, you can replace this with real logic
    df = similarity_scores(query, top_n, model, sim_measure)
    # Only keep 'abstract', 'Title' and 'Link' columns
    # check if dc:title is present, rename it to Title
    if 'dc:title' in df.columns:
        df['Title'] = df['dc:title']
    if 'prism:doi' in df.columns:
        df['DOI'] = df['prism:doi']
    if 'DOI' in df.columns:
        df['DOI'] = df['DOI'].apply(lambda x: f"https://doi.org/{x}")
    if 'dc:creator' in df.columns:
        df['Authors'] = df['dc:creator']
    if 'Abstract' in df.columns:
        df['abstract'] = df['Abstract']
    count = 0
    for rnd_str in df['abstract']:
        if 'No abstract' in rnd_str:
            count += 1
    if count > 0.5 * len(df):
        df['abstract'] = ''
    df = df[['Title', 'Authors', 'abstract', 'DOI']]
    print(df['Authors'].head())
    results = df.to_dict(orient='records')
    return results

@app.route('/fetch', methods=['POST'])
def fetch():
    global progress
    progress = 0  # Reset progress

    keywords = request.form['keywords'].split(',')
    start_year = int(request.form['start_year'])
    end_year = int(request.form['end_year'])
    results_per_year = int(request.form['results_per_year'])
    total_years = end_year - start_year + 1

    all_results = []
    for year in range(start_year, end_year + 1):
        year_results = get_results_for_year(keywords, year, results_per_year, total_years)
        all_results.extend(year_results)

    # Convert results to DataFrame
    df = pd.DataFrame(all_results)

    # Get abstracts using DOIs, allocate 50% of progress to this
    total_records = len(df)
    for index, row in df.iterrows():
        doi = row.get('prism:doi', None)
        if pd.notna(doi):
            df.at[index, 'abstract'] = get_abstract(doi, API_KEY)
        
        # Update progress
        progress_step = 50 / total_records
        progress = min(100, progress + progress_step)

    # Clean results
    df = clean_text(df)

    # Save to CSV
    df.to_csv('scopus_results.csv', index=False)
    
    progress = 100  # Set progress to 100% upon completion
    return jsonify({"message": f"Saved {len(df)} records to scopus_results.csv"})

@app.route('/progress', methods=['GET'])
def get_progress():
    global progress
    return jsonify({'progress': progress})

@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')

if __name__ == '__main__':
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the overall logging level
    
    # Add a filter to the default handler
    handler = logging.StreamHandler()
    handler.addFilter(IgnoreSpecificLogsFilter())
    logger.addHandler(handler)
    
    app.run(debug=True)

