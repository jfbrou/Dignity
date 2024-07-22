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
years = range(1997, 1997 + 1, 1)
filepaths = [os.path.join(cex_r_data, "stubs", "CE-HG-Inter-" + str(year) + ".txt") for year in years]
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
    output_filepath = os.path.join(cex_r_data, "stubs", "codebook_" + str(year) + "_cleaned.json")

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

Below is a file containing 7 columns of data. 
Some rows start with a *. These rows are comments and should be ignored.
The first column takes two values, either 1 or 2. When it takes the value of 2, it means that the corresponding text in the third column is a continuation of the text in the row above the current row. For those rows where the first column is equal to 2, the text in the third column should be concatenated with the text in the third column of the row above the current row.
The second column contains integers that determine the level of aggregation of the data in the row.
The third column contains text.
The fourth column contains either a code or some text. When it contains text, it is a description of an aggregation category. When it contains a code, it is a specific element in that category.
The fifth column takes three values, either "I", "G", or "S". When it takes the value of "I", it means that it is an element in a category. When it takes the values of "G" or "S", it means that it is a cateogry.
The sixth column contains values.
The seventh column contains text.

{excerpt}

Transform this file as a JSON file. Example:

    1  1  Average annual expenditures                                    TOTALEXP  G  1  EXPEND                            
    1  2    Food                                                         FOODTOTL  G  1  EXPEND                            
    1  3      Food at home                                               FOODHOME  G  1  FOOD                              
    1  4        Grocery stores                                           790220    I  1  FOOD                              
    1  4        Convenience stores                                       790230    I  1  FOOD                              
    1  4        Food prepared by consumer unit on out-of-town trips      190904    I  1  FOOD                              
    1  3      Food away from home                                        FOODAWAY  G  1  FOOD                              
    1  4        Meals at restaurants, carry-outs and other               790410    I  1  FOOD                              
    1  4        Board (including at school)                              190901    I  1  FOOD                              
    1  4        Catered affairs                                          190902    I  1  FOOD                              
    1  4        Food on out-of-town trips                                190903    I  1  FOOD                              
    1  4        School lunches                                           790430    I  1  FOOD                              
    1  4        Meals as pay                                             800700    I  1  FOOD

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