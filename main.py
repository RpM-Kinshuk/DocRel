from flask import Flask, render_template, request, jsonify, url_for
import requests
import logging
import pandas as pd
import httpx
import time

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
        'count': 25,
        'start': 0,
        'sort': 'rowTotal'
    }

    while True:
        response = requests.get(BASE_URL, headers=HEADERS, params=params)
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

        if start_index + items_per_page >= total_results or len(results) >= results_per_year:
            break

        params['start'] = start_index + items_per_page
        time.sleep(1)

    return results[:results_per_year]

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
                return data.get('abstracts-retrieval-response', {}).get('coredata', {}).get('dc:description', 'No abstract available')
            else:
                print(f"Error {response.status_code} for DOI: {doi}")
                return 'Error fetching content'
        except httpx.ReadTimeout:
            print(f"Read timeout for DOI: {doi}, attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(backoff_factor * (2 ** attempt))
            else:
                return 'Error fetching content due to timeout'

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/semantic', methods=['GET', 'POST'])
def semantic():
    if request.method == 'POST':
        query = request.form['query']
        top_results = perform_semantic_analysis(query)
        return jsonify({'results': top_results})
    return render_template('semantic.html')

def perform_semantic_analysis(query):
    mock_results = ['Result 1', 'Result 2', 'Result 3']
    return mock_results

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

    # Save to CSV
    df.to_csv('scopus_results_with_abstracts.csv', index=False)
    
    progress = 100  # Set progress to 100% upon completion
    return jsonify({"message": f"Saved {len(df)} records to scopus_results_with_abstracts.csv"})

@app.route('/progress', methods=['GET'])
def get_progress():
    global progress
    return jsonify({'progress': progress})

class IgnoreSpecificLogsFilter(logging.Filter):
    def filter(self, record):
        # Filter out specific HTTP request patterns
        if "GET /progress" in record.getMessage() or \
           "GET https://api.elsevier.com/content/abstract/doi/" in record.getMessage():
            return False
        return True

if __name__ == '__main__':
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the overall logging level
    
    # Add a filter to the default handler
    handler = logging.StreamHandler()
    handler.addFilter(IgnoreSpecificLogsFilter())
    logger.addHandler(handler)
    
    app.run(debug=True)
