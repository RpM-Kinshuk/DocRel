import httpx

api_key = '89072218d14bbda5d9acdad8d14bcfbd'
doi = '10.1071/RS10018'
url = f'https://api.elsevier.com/content/abstract/doi/{doi}'

headers = {
    'X-ELS-APIKey': api_key,
    'Accept': 'application/json'
}

response = httpx.get(url, headers=headers)

# Print the response status code and content
print(f'Status Code: {response.status_code}')
print(response.text)
