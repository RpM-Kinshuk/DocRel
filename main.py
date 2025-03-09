import os
import time
import httpx
import logging
import requests
import pandas as pd
from utils.mailing import doc_mail
from utils.sts import similarity_scores
from utils.util import save_data, load_data, modify_data
from flask import Flask, render_template, request, jsonify
from utils.clean import clean_text, IgnoreSpecificLogsFilter

app = Flask(__name__)

API_KEY = "89072218d14bbda5d9acdad8d14bcfbd"
HEADERS = {"X-ELS-APIKey": API_KEY, "Accept": "application/json"}
BASE_URL = "https://api.elsevier.com/content/search/scopus"

# Global variable to track progress
progress = 0


def get_results_for_year(keywords, year, results_per_year, total_years):
    global progress
    results = []
    query = " OR ".join([f"TITLE-ABS-KEY({kw})" for kw in keywords])

    params = {
        "query": f"({query}) AND PUBYEAR = {year}",
        "count": min(25, results_per_year),  # Use 25 as the max per request
        "start": 0,
        "sort": "rowTotal",
        "view": "COMPLETE",
    }

    while True:
        response = requests.get(BASE_URL, headers=HEADERS, params=params)
        # print(f"Request URL: {response.url}")
        if response.status_code == 401:
            print("IP uthentication to Scopus failed. Please check the network.")
            # delete the view key from params and retry
            if "view" in params:
                del params["view"]
            continue
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break

        data = response.json()
        entries = data.get("search-results", {}).get("entry", [])
        results.extend(entries)

        total_results = int(
            data.get("search-results", {}).get("opensearch:totalResults", 0)
        )
        start_index = int(
            data.get("search-results", {}).get("opensearch:startIndex", 0)
        )
        items_per_page = int(
            data.get("search-results", {}).get("opensearch:itemsPerPage", 0)
        )

        # Update progress
        progress_step = 50 / (results_per_year * total_years)
        progress = min(100, progress + progress_step)

        # Check if we have retrieved enough results for the year or if we reached the API limits
        if (
            start_index + items_per_page >= total_results
            or len(results) >= results_per_year
        ):
            break

        # Continue from the next start index
        params["start"] = start_index + items_per_page
        time.sleep(1)  # Sleep to avoid rate-limiting or throttling issues

    return results[:results_per_year]  # Return only the specified number of results


def get_abstract(doi, api_key, max_retries=3, backoff_factor=0.3):
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    url = f"https://api.elsevier.com/content/abstract/doi/{doi}"

    for attempt in range(max_retries):
        try:
            response = httpx.get(url, headers=headers, timeout=30.0)
            if response.status_code == 200:
                data = response.json()
                coredata = data.get("abstracts-retrieval-response", {}).get(
                    "coredata", {}
                )

                # Attempt to get the abstract (dc:description)
                abstract = coredata.get("dc:description", None)

                # Fallback to prism:teaser or dc:title if no abstract is available
                if abstract:
                    return abstract
                else:
                    teaser = coredata.get("prism:teaser", None)
                    if teaser:
                        return teaser
                    title = coredata.get("dc:title", "No abstract or title available")
                    return f"Title: {title}"
            elif response.status_code == 404:
                print(f"DOI not found: {doi}")
                return "DOI not found"
            elif response.status_code == 401:
                print(f"Authentication failed for DOI: {doi}")
                return "Authentication failed"
            elif response.status_code == 429:
                print("Quota exceeded, please try again later.")
                return "Quota exceeded"
            else:
                print(f"Error {response.status_code} for DOI: {doi}")
                return f"Error {response.status_code}"
        except httpx.ReadTimeout:
            print(f"Read timeout for DOI: {doi}, attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(backoff_factor * (2**attempt))  # Exponential backoff
            else:
                return "Error fetching content due to timeout"
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    return "Error fetching content"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/semantic", methods=["GET", "POST"])
def semantic():
    if not os.path.exists("scopus_results.csv"):
        return jsonify({"message": "Please fetch data first!"})

    if request.method == "POST":
        query = str(request.form["query"])
        modify_data("query", query)  # Update the stored query

        selected_model = request.form["model"]  # Get the selected model
        modify_data("model", selected_model)  # Update the stored model

        top_n = int(request.form["top_n"])  # Get the number of results to return
        modify_data("top_n", top_n)  # Update the stored top_n

        sim_measure = request.form["sim_measure"]  # Get the similarity measure
        modify_data("sim_measure", sim_measure)

        top_results = perform_semantic_analysis(
            query, selected_model, top_n, sim_measure
        )
        modify_data("results", top_results)  # Update the stored results

        return jsonify({"results": top_results})
    return render_template("semantic.html")


def perform_semantic_analysis(query, model, top_n=5, sim_measure="cosine"):
    df = similarity_scores(query, top_n, model, sim_measure)
    # Only keep 'abstract', 'Title' and 'Link' columns
    # check if dc:title is present, rename it to Title
    if "dc:title" in df.columns:
        df["Title"] = df["dc:title"]

    if "prism:doi" in df.columns:
        df["DOI"] = df["prism:doi"]

    if "DOI" in df.columns:
        df["DOI"] = df["DOI"].apply(lambda x: f"https://doi.org/{x}")

    if "author" in df.columns:
        df["Authors"] = df["author"]
    elif "dc:creator" in df.columns:
        df["Authors"] = df["dc:creator"]

    if "Abstract" in df.columns:
        df["abstract"] = df["Abstract"]

    count = 0
    for rnd_str in df["abstract"]:
        if isinstance(rnd_str, str) and "No abstract" in rnd_str:
            count += 1
    if count > 0.5 * len(df):
        df["abstract"] = ""

    df["Similarity"] = df["Similarity"].apply(lambda x: f"{x:.4f}")
    df["Overlap"] = df["Keyword Overlap"].apply(lambda x: f"{x:.2f}")

    df = df[["Title", "Authors", "abstract", "DOI", "Similarity", "Overlap"]]

    for key in df.keys():
        df[key] = df[key].apply(lambda x: x if pd.notna(x) else "Not available")
        df[key] = df[key].astype(str)

    print(df["Authors"].head())

    results = df.where(pd.notna(df), None).to_dict(orient="records")
    return results


@app.route("/fetch", methods=["POST"])
def fetch():
    global progress
    progress = 0  # Reset progress

    keywords = request.form["keywords"].split(",")
    start_year = int(request.form["start_year"])
    end_year = int(request.form["end_year"])
    results_per_year = int(request.form["results_per_year"])
    total_years = end_year - start_year + 1

    all_results = []
    for year in range(start_year, end_year + 1):
        year_results = get_results_for_year(
            keywords, year, results_per_year, total_years
        )
        all_results.extend(year_results)

    # Convert results to DataFrame
    df = pd.DataFrame(all_results)

    if "dc:description" in df.columns:
        df["abstract"] = df["dc:description"]
        df.drop(columns=["dc:description"], inplace=True)
    elif "dc:title" in df.columns:
        df["abstract"] = df["dc:title"]

    # Get abstracts using DOIs, allocate 50% of progress to this
    total_records = len(df)
    for index, row in df.iterrows():
        doi = row.get("prism:doi", None)
        abs = row.get("abstract", None)
        # check for empty abstracts
        if abs is None and pd.notna(doi):
            df.at[index, "abstract"] = get_abstract(doi, API_KEY)

        # Update progress
        progress_step = 50 / total_records
        progress = min(100, progress + progress_step)

    # Clean results
    df = clean_text(df)

    # Save to CSV
    df.to_csv("scopus_results.csv", index=False)

    # Save global variables to JSON
    save_data(
        {
            "keywords": keywords,
            "start_year": start_year,
            "end_year": end_year,
            "results_per_year": results_per_year,
            # 'records': df.to_dict(orient='records')  # Save results as well
        }
    )

    progress = 100  # Set progress to 100% upon completion
    return jsonify({"message": f"Saved {len(df)} records to scopus_results.csv"})


@app.route("/progress", methods=["GET"])
def get_progress():
    global progress
    return jsonify({"progress": progress})


@app.route("/about", methods=["GET", "POST"])
def about():
    return render_template("about.html")


@app.route("/mail", methods=["GET"])
def mail():
    return render_template("mail.html")


@app.route("/send", methods=["POST"])
def send():
    try:
        global results
        email = request.form["email"]

        # Load the stored variables
        stored_data = load_data()

        for v in ["keywords", "start_year", "end_year", "results_per_year"]:
            if v not in stored_data or stored_data[v] is None:
                stored_data[v] = "Unknown"
        if isinstance(stored_data["keywords"], list):
            stored_data["keywords"] = ", ".join(stored_data["keywords"])

        # Check if any required data is missing
        for v in [
            email,
            stored_data.get("query"),
            stored_data.get("start_year"),
            stored_data.get("end_year"),
            stored_data.get("results_per_year"),
            stored_data.get("top_n"),
            stored_data.get("results"),
        ]:
            if v is None:
                return jsonify({"message": "Please fetch data first!"})

        doc_mail(
            email,
            stored_data["query"],
            stored_data["keywords"],
            stored_data["start_year"],
            stored_data["end_year"],
            stored_data["results_per_year"],
            stored_data["top_n"],
            stored_data["model"].upper(),
            stored_data["sim_measure"].capitalize(),
            stored_data["results"],
        )

        return jsonify({"message": "Mail sent successfully!"})
    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"})


if __name__ == "__main__":
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the overall logging level

    # Add a filter to the default handler
    handler = logging.StreamHandler()
    handler.addFilter(IgnoreSpecificLogsFilter())
    logger.addHandler(handler)

    app.run(debug=True)
