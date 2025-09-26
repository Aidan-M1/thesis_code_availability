"""
squidle_retrieval.py

Gathers all available datasets on squidle+ API and downloads to csv.

Author: Aidan Murray
Date: 2025-09-26
"""

import requests
import json
import time


API_KEY = ""
BASE_URL = "https://squidle.org"

annotation_url = f"{BASE_URL}/api/annotation_set"
query = json.dumps({
    "filters": [
        {"name": "label_scheme_id", "op": "in", "val": [21, 24, 8]},
        {"name": "current_user_can_view", "op": "==", "val": True}
    ]
})
headers = {
    "Authorization": f"ApiKey {API_KEY}"
}
page = 1
annotation_sets = []

while True:
    params = {
        "q": query,
        "page": page
    }
    r = requests.get(annotation_url, headers=headers, params=params)
    print(f"Fetching page {page}: {r.url}")
    r.raise_for_status()
    annotation_data = r.json()
    annotation_objects = annotation_data.get("objects", [])
    if not annotation_objects:
        break
    annotation_sets.extend(annotation_objects)
    if page >= annotation_data.get("total_pages", 1):
        break
    page += 1
print(
    f"Retrieved {len(annotation_sets)} annotation sets across {page} page(s)."
    )


operations = {
    "operations": [{
        "module": "pandas",
        "method": "json_normalize"
    }]
}
params = {
    "template": "dataframe.csv",
    "f": json.dumps(operations) 
}
skipped = []
for i, aset in enumerate(annotation_sets):
    aset_id = aset['id']
    aset_name = aset['name']

    print(
        f"Exporting to csv, Iteration \
          {i}, ID: {aset_id}, Annotation Set: {aset_name}"
        )
    
    export_url = f"{annotation_url}/{aset_id}/export"
    r = requests.get(export_url, headers=headers, params=params)
    export_data = r.json()

    status_url = BASE_URL + export_data['status_url']
    result_url = BASE_URL + export_data['result_url']
    print("Waiting for response to complete")
    while True:
        r = requests.get(status_url, headers=headers)
        if r.status_code != 200:
            break
        status_data = r.json()
        if status_data['result_available']:
            print("Export ready!")
            break
        elif status_data['status'] == 'failed':
            raise RuntimeError("Export failed.")
        else:
            print("Still processing...")
            time.sleep(2)
    if r.status_code != 200:
        skipped.append([aset_id, aset_name])
        with open("../datasets/error_log.txt", "a") as error_file:
            error_file.write(f"{r}\n{r.json()}")
            error_file.write("-" * 40 + "\n")
        continue

    r = requests.get(result_url, headers=headers)
    r.raise_for_status()
    print("Downloading file ...")
    with open(f"../datasets/annotations_{aset_id}.csv", "wb") as f:
        f.write(r.content)
    print(f"Export complete. Saved as 'annotations_{aset_id}.csv'.")
print(f"All {i} exports finished.")