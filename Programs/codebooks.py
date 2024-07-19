# Import libraries
import os
import json
import requests
from glob import glob

# Import functions and directories
from functions import *
from directories import *

# Define the parameters of the model
MODEL = "gpt-4o"
URL = "https://api.openai.com/v1/chat/completions"
API_KEY = 'sk-None-hO8SyBtYbgt8z65D54qeT3BlbkFJjxeODok8EBlysTp6RPB6'

# List of files
years = range(1984, 1994 + 1, 1)
filepaths = [os.path.join(cex_r_data, "intrvw" + str(year)[2:], "codebook.txt") for year in years]
print(f"{len(filepaths)} files found")

# Define several functions
def complete(messages=None, max_tokens=4096):

    # Headers
    headers = {
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {API_KEY}"
    }

    # Define the payload
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "model": MODEL,
        "response_format": { "type": "json_object" }
    }
    
    # Send the POST request
    response = requests.post(URL, headers=headers, data=json.dumps(payload))
    try:
        response = response.json()
    except:
        print('ERROR: Could not parse response')
        return None

    # Validate
    if 'choices' not in response: 
        print(response)
        return None

    # Extract completion
    completion = response['choices'][0]['message']['content']

    return completion

def format_prompt_for_openai(user_message, system_message=None):

    # Initialize message list
    messages = []

    # Add system message
    if system_message is not None and system_message != "":
        messages.append({"role": "system", "content": system_message})

    # Add user message
    messages.append({"role": "user", "content": user_message})

    return messages

# Loop through files
for filepath in filepaths:

    # Log
    print(f"Processing file: {filepath}.")

    # Save the processed codebook
    year = years[filepaths.index(filepath)]
    output_filepath = os.path.join(cex_r_data, "intrvw" + str(year)[2:], "codebook_cleaned.json")

    # Skip if already exists
    if os.path.exists(output_filepath):
        print(f"Codebook already exists: {output_filepath}.")
        continue

    # Load the file
    with open(filepath, 'r') as f:
        file = f.read()

    # Initialize returned object
    codebook = {}
    
    # How you batch the files is up to you
    batches = []
    num_characters_per_batch = 1500 * 7 # We process 1500 words at a time, 7 characters per word on average
    overlap = 500 # We want some overlap between batches of ~500 characters

    # Split the file into batches
    for i in range(0, len(file), num_characters_per_batch - overlap):
        batches.append(file[i:i + num_characters_per_batch])

    # Loop through batches
    for excerpt in batches:

        # --- Compose your prompt ---
        prompt = f"""

Here's a file containing codes and descriptions:

{excerpt}

Extract as a JSON each code and their descriptions. Example:

    "220122": "Same as 220121 - owned vacation home, vacation coops",
    "190903": "Food and non-alc beverages at restaurants, cafes, fast food places on trips",
    "980330": "Percent vehicle owner",
    ...

""".strip()

        # Format the prompt
        messages = format_prompt_for_openai(prompt)

        # Send for completion
        completion = complete(messages=messages, max_tokens=4096)
        
        # Check
        if completion is None: 
            print("ERROR: Could not get completion.")
            continue
        
        # Convert to JSON
        completion = json.loads(completion)

        # Add to codebook
        for key, value in completion.items():
            codebook[key] = value

        # Pretty print the codebook
        print(json.dumps(codebook, indent=4))

        # Save the codebook
        with open(output_filepath, 'w') as f:
            json.dump(codebook, f, indent=4)