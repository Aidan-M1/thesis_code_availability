"""
prompts.py

This file contains a systematic testing of the effect of different
prompts on the QA accuracy of the qwen2.5vl:3b model in the 
classification of images in the RLS dataset. A custom version of the
model was used as specified in Modelfile. The model's outputs were
evaluated using f1 score. 

Author: Aidan Murray
Date: 2025-06-03
"""
print("Initialising prompts.py ...")
import numpy as np
import pandas as pd
from pathlib import Path
import ast
import warnings
from f1_score_custom import f1_score
import os
import time
import requests
import json
import base64

def call_ollama_api(messages, model='qwen2.5vl:3b', timeout=120, delay=10, max_retries=3):
    headers = {
        "Content-Type": "application/json"
    }

    new_messages = []
    for m in messages:
        m = m.copy()
        if 'images' in m:
            encoded_images = []
            for img_path in m['images']:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                encoded = base64.b64encode(img_bytes).decode("utf-8")
                encoded_images.append(encoded)
            m['images'] = encoded_images
        new_messages.append(m)


    payload = {
        "model": model,
        "messages": new_messages,
        "stream": False
    }
    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(OLLAMA_URL, headers=headers, data=json.dumps(payload), timeout=timeout)
            response.raise_for_status()
            return {"message": response.json()['message']['content'],
                    "error": None}
        
        except requests.Timeout:
            print(f"⚠ Request timed out after {timeout} seconds.")
            return {"error": "Timeout",
                    "message": None}
        
        except requests.RequestException as e:
            print(f"⚠ Request failed: {e}")
            retries += 1
            if retries < max_retries:
                print("Retrying ...")
                time.sleep(delay)
                continue
            return {"error": str(e),
                    "message": None}


np.random.seed(42)
OLLAMA_URL = os.getenv('OLLAMA_URL')
N_DEMOS = 2
TIMEOUT = 600
OUTPUT_PATH = Path("./data/output")
DATASET_PATH = "./data/combined.csv"
VAL_PATH = "./data/validation.csv"
FOLDER_PATH = Path('./images')
DELAY = 10

# get annotations
annotations = pd.read_csv(DATASET_PATH)
allowed_labels = str(list(annotations['label.name'].unique()))

# get db
val_set = pd.read_csv(VAL_PATH)

# design prompts
# 0. basic concise prompt
prompt_0 = """
### Task ###
Analyse the entire image and decide which of the label names in AllowedLabels are visible.
AllowedLabels:
""" + allowed_labels + """
Never invent labels, synonyms, descriptions or taxonomic levels that are not in AllowedLabels.
Never repeat labels.
If it is unclear whether a label is present, treat it as absent and instead return "Unscorable".

### Output Format ###
Output the answer as a JSON array of label strings, alphabetically sorted, without extra keys or text.
Output format example: ['Crustose coralline algae', 'Sponges (encrusting)']
"""

# 1. basic prompt with more emphasis
prompt_1 = """
### Task ###
Analyse the entire image carefully and decide which (if any) of the exact label names in AllowedLabels correspond to features that are clearly visible in the image.
AllowedLabels:
""" + allowed_labels + """
Never invent labels, synonyms, descriptions or taxonomic levels that are not in AllowedLabels.
Never repeat labels.
If it is unclear whether a label is present, treat it as absent and instead return "Unscorable".

### Output Format ###
Output the answer strictly as a JSON array of distinct label strings, alphabetically sorted, without extra keys or text.
Output format example: ['Crustose coralline algae', 'Sponges (encrusting)']
"""

# 2. prompt with extra context
prompt_2 = """
### Task ###
Analyse the entire image carefully and decide which (if any) of the exact label names in AllowedLabels correspond to features that are clearly visible in the image.
AllowedLabels:
""" + allowed_labels + """
Never invent labels, synonyms, descriptions or taxonomic levels that are not in AllowedLabels.
Never repeat labels.
If it is unclear whether a label is present, treat it as absent and instead return "Unscorable".

### Context ###
Points overlaid on each image are assigned a category from AnnotatedImages. For most images, categories are assigned to five points overlayed in the quincunx layout.
Where multiple layers of biology are found under a point, the uppermost one is to be labelled. For example, if a point lands on an epiphyte covering a kelp, then attribute the point’s label to the epiphyte and not the kelp.
Where no biota is present under the point, the bare substrate is scored (e.g. sand, bare coral rubble, gravel or bare rock).
If there is a matt-forming covering of short filamentous algae intermixed with any sediment in it then it is considered “Turf”.
If there is a matted mass of slimy algae or cyanobacteria, but with no sediment in it then it is considered “Slime” (usually covers dead coral).
If there are medium to large sized clumps or long strands of filamentous algae not forming a matt, but growing loosely over plants, invertebrates, or substrate they are considered “filamentous (red, green of brown) algae_epiphytic”.
If the algae is medium sized and has a structure other than filamentous, ie. branching, sheetlike, or globular then the category assigned should be medium foliose (red, green or brown) algae. NB. There are also separate categories for certain green algae and canopy algae.

### Output Format ###
Output the answer strictly as a JSON array of distinct label strings, alphabetically sorted, without extra keys or text.
Output format example: ['Crustose coralline algae', 'Sponges (encrusting)']
"""

# 3. few-shot prompt
# same basic prompt but chat history is different
prompt_3 = """
### Task ###
Analyse the entire image carefully and decide which (if any) of the exact label names in AllowedLabels correspond to features that are clearly visible in the image.
AllowedLabels:
""" + allowed_labels + """
Never invent labels, synonyms, descriptions or taxonomic levels that are not in AllowedLabels.
Never repeat labels.
If it is unclear whether a label is present, treat it as absent and instead return "Unscorable".

### Output Format ###
Output the answer strictly as a JSON array of distinct label strings, alphabetically sorted, without extra keys or text.
Output format example: ['Crustose coralline algae', 'Sponges (encrusting)']
"""

# 4. zero-shot COT prompt
prompt_4 = """
### Task ###
Analyse the entire image carefully and decide which (if any) of the exact label names in AllowedLabels correspond to features that are clearly visible in the image.
AllowedLabels:
""" + allowed_labels + """
Never repeat labels.
If it is unclear whether a label is present, treat it as absent and instead return "Unscorable".

### Output Format ###
Output the answer strictly as a Python dictionary consisting of two keys named "reasoning" and "labels".
"reasoning" should correspond to a string where the the image is analysed and conclusions are come to.
"labels" must correspond to a Python list, only containing the labels from AllowedLabels that are visible in the image.
Never add to "labels" invented labels, synonyms, descriptions or taxonomic levels that are not in AllowedLabels.

Output format example: 
{'reasoning': 'There is a matt-forming covering of short filamentous algae intermixed with any sediment in it. This means that "Turf" is present. There is medium sized algae, red in colour with a non-filamentous structure (globular), therefore medium foliose red algae is present.',
'labels': ['Turfing algae (<2 cm high algal/sediment mat on rock)', 'Medium foliose red algae']}
"""
print("randomizing prompt order...")
# randomize prompt order
prompts = {0 : prompt_0,
           1 : prompt_1,
           2 : prompt_2,
           3 : prompt_3,
           4 : prompt_4}
prompt_order = [i for i in range(5)]
np.random.shuffle(prompt_order)

# get image paths and randomise order
image_paths = [str(file_path) for file_path in FOLDER_PATH.iterdir() if file_path.is_file()]
np.random.shuffle(image_paths)

# set aside examples for few shot demonstrations and get their labels
demo_image_paths = [image_paths.pop(0) for _ in range(N_DEMOS)]
demo_image_ids = [int(Path(path).stem) for path in demo_image_paths]
demo_y_true = [str(list(ast.literal_eval(val_set.loc[val_set['point.media.id'] == image_id, 'label.name'].iloc[0]))) for image_id in demo_image_ids]

true_labels = []
predicted_labels = {k : [] for k in prompt_order}
times = {k: [] for k in prompt_order}
failed_parse = {k: 0 for k in prompt_order}
timeouts = {k: 0 for k in prompt_order}


print("beginning api calls...")
# make a seperate api call for each image, for each prompt
for i, path in enumerate(image_paths):
    print(f"Image {i}")

    image_id = int(Path(path).stem)
    y_true = ast.literal_eval(val_set.loc[val_set['point.media.id'] == image_id, 'label.name'].iloc[0])
    true_labels.append(y_true)

    for j in prompt_order:
        print(f"Prompt {j}...")
        start_time = time.time()
        # zero shot prompts
        if j != 3:
            messages=[{'role'       : 'user',
                        'content'    : prompts[j],
                        'images'     : [path]}]
        # few-shot prompt
        else:
            messages=[{ 'role'       : 'user',
                        'content'    : prompts[j],
                        'images'     : [demo_image_paths[0]]},
                        {'role'      : 'assistant',
                        ' content'   : demo_y_true[0]}]
            
            for k in range(1, N_DEMOS):
                messages.append({'role'     : 'user',
                                 'images'   : [demo_image_paths[k]]})
                messages.append({'role'     : 'assistant',
                                 'content'  : demo_y_true[k]})
                
            messages.append({'role'     : 'user',
                             'images'   : [path]})

        
        response = call_ollama_api(messages=messages, model="Benthiq:3b", timeout=TIMEOUT)

        if response['message'] is not None:
            end_time = time.time()
            execution_time = end_time - start_time
            try:
                y_pred = ast.literal_eval(response['message'])
                if j == 4:
                    y_pred = y_pred['labels']
            except (ValueError, SyntaxError) as e:
                warnings.warn(f"Warning: Failed to parse model output at image {path}. Error: {e}")
                y_pred = ['Failed']
                failed_parse[j] += 1
        elif response['error'] == "Timeout":
            y_pred = ['Timeout']
            timeouts[j] += 1
            execution_time = TIMEOUT
            time.sleep(DELAY)
        else:
            print("Failed to connect to ollama server...\nExiting app")
            exit()
        times[j].append(execution_time)
        predicted_labels[j].append(y_pred)



print("evaluating predictions...")
evals = { i : [f1_score(true_labels[j], predicted_labels[i][j])
           for j in range(len(true_labels))] for i in prompt_order}

df_eval = pd.DataFrame()
for k, v in evals.items():
    df_eval[f"Prompt {k}"] = v
df_eval.to_csv(OUTPUT_PATH / "prompt_evals.csv")
df_eval.describe().to_csv(OUTPUT_PATH / "prompt_eval_stats.csv")

df_times = pd.DataFrame()
for k, v in times.items():
    df_times[f"Prompt {k}"] = v
df_times.to_csv(OUTPUT_PATH / "prompt_times.csv")
df_times.describe().to_csv(OUTPUT_PATH / "prompt_time_stats.csv")   

with open(OUTPUT_PATH / "prompt_failed_parses.txt", "w") as f:
    for k, v in failed_parse.items():
        f.write(f"Prompt {k} failed to parse {v} times\n")

with open(OUTPUT_PATH / "prompt_timeouts.txt", "w") as f:
    for k, v in timeouts.items():
        f.write(f"Prompt {k} timed out {v} times\n")

print("evaluations complete")