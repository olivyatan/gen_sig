#TODO: this should be replaced by using a pipeline template.

import json
import re
import os
import ast
import string
import traceback
import pandas as pd
import numpy as np

from tqdm import tqdm
import xlsxwriter

from itertools import permutations
import itertools

import nltk
import string
from functools import partial
import sys
#sys.path.append('src')
#sys.path.append('src/generation')
from tqdm import tqdm
import time
import pandas as pd
# from langchain import PromptTemplate
# from langchain.chains import LLMChain

from dev.generation.generators.gpt_generator import SignalsGeneratorGPT ,SignalLabelGeneratorGPT
from dev.generation.generators.mistral_generator import SignalsGeneratorMistral
from dev.data_extraction.extraction_funcs_utils import filter_aspects
from dev.data_extraction.extraction_funcs_utils import forbidden_words, forbidden_words_post

# from pychomsky.chchat import AzureOpenAIChatWrapper


verbose = True
debug = True



# '''
# Prompt to generate signal from an item's details
# '''
# item_sigs_template = '''
# You are an expert salesperson that always excels in providing the best buying tips to buyers given a product of interest they wish to buy.
#
# Product details:
# * Product title: {}
# * Product features: {}
# * Product description: {}
#
# Given the product's details above, your aim is first to identify which of the product characteristics is the most appealing to make a convincing offer to the buyer.
# The characteristic you identify must appear in the product's details, with higher priority to a characteristic that is not part of the product's title.
# Notice that the product features is organized in the form: {{feature name : feature value }}, so your answer must not consist of feature name only, and preferabely include only the feture value.
# Think carefully why such a characteristic is the best to mention to the buyer and what would be your pitch strategy.
# As your answer, mention the product's characteristic and shortly generate one pitch sentence based on that feature.
# The characteristic should be informative, yet short, only three or four words, and (this is important!) in no case should it exceed 32 characters.
# Generate only a single json answer in the format: {{"characteristic": "short characteristic", "sale pitch": "short sentence with characteristic-related sale pitch", "explanation": "your reasoning why this characteristic
# is important"}}. If one of the tests you generate between double quotes contains single or double quotes, make sure you escape these intermediate quotes with a backslash.
#
# '''


def extract_first_dict(text):
    # Regular expression to match the first dictionary-like string
    # dict_pattern =  #r"\{[\s\S]*?\}" #r"\{.*?\}"
    # Regular expression pattern to match dictionary-like substrings, allowing for \n and \t, but ensuring valid dictionary format
    dict_pattern = r"\{(?:\s*'[^']*'\s*:\s*'[^']*'\s*,?|\s*\"[^\"]*\"\s*:\s*\"[^\"]*\"\s*,?)*\}"

    try:
        # Find all matches
        matches = re.findall(dict_pattern, text)
        if matches:
            # Extract the first match
            first_dict_str = matches[0]
            # Parse the dictionary string using ast.literal_eval
            first_dict = ast.literal_eval(first_dict_str)
            return first_dict
        else:
            print(f"Couldn't find a match to dict string in {text}")
            return None

    except:
        print(f"Couldn't properly parse the dict string in {text}")
        return None
            # raise ValueError("No dictionary-like string found in the text")



def check_words(input_text, text_to_check):
    # Split the texts into lists of words
    input_words = input_text.lower().split()
    check_words = text_to_check.lower().split()

    # Convert the lists to sets
    input_set = set(input_words)
    check_set = set(check_words)

    # Check if all words from the input text appear in the text to check
    return input_set.issubset(check_set)

def signal_match_text_relaxed(text, signal):
    translator = str.maketrans('', '', string.punctuation)
    signal_no_punct = str(signal).lower().translate(translator)
    sentences = nltk.tokenize.sent_tokenize(text)
    for sentence in sentences:
        sent_no_punct = str(sentence).lower().translate(translator)
        if check_words(signal_no_punct, sent_no_punct):
            return True
    return False



def signal_match_text(text, signal):
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    sentence_no_punct = str(signal).lower().translate(translator)
    text_no_punct = str(text).lower().translate(translator)
    return sentence_no_punct in text_no_punct


#Mistral tends to genegate several dictionaries in the text.
#Ad-hoc cleanups. Remove when fixed (mainly due to the prompt having non-escapted quotes)
def extract_sig_fix(gen_text, colnames):
    # print(f"in extract_sig_fix: {gen_text}")
    first_dict = extract_first_dict(gen_text)
    if first_dict is not None and 'characteristic' in first_dict:
        first_sig = first_dict.get('characteristic').title()
    else:
        first_sig = ''
    return pd.Series([first_dict, first_sig], index=colnames)


def extract_sig_fix_multi(gen_text, colnames, num_signals_per_item=5, model_name="gpt"):
    """ Extracts multiple signals from the response and ensures structured storage. """

    extracted_signals = extract_first_dict(gen_text)  # Use improved extraction function

    # Ensure `colnames` has at least two elements
    col_text = colnames[0]  # "gen_text_gpt"
    col_sig = colnames[1]  # "gen_sig_gpt"
    col_exp_prefix = f'gen_exp_{model_name}'

    if num_signals_per_item == 1:
        characteristic = extracted_signals.get("characteristic", "").title()
        explanation = extracted_signals.get("explanation", "")

        return pd.Series(
            [extracted_signals, characteristic, explanation],
            index=[col_text, col_sig, col_exp_prefix]  # Ensure correct column naming
        )

    elif extracted_signals and any(f"characteristic_{i + 1}" in extracted_signals for i in range(num_signals_per_item)):
        # Extract multiple characteristics
        signals = [extracted_signals.get(f"characteristic_{i + 1}", "").title() for i in range(num_signals_per_item)]
        explanations = [extracted_signals.get(f"explanation_{i + 1}", "") for i in range(num_signals_per_item)]

        # Debugging
        # print(f"Extracted Explanations: {explanations}")

        return pd.Series(
            [extracted_signals] + signals + explanations,
            index=[col_text] +
                  [f'{col_sig}_{i + 1}' for i in range(num_signals_per_item)] +
                  [f'{col_exp_prefix}_{i + 1}' for i in range(num_signals_per_item)]
        )

    else:
        # Default empty placeholders
        signals = [""] * num_signals_per_item
        explanations = [""] * num_signals_per_item

        return pd.Series(
            [extracted_signals] + signals + explanations,
            index=[col_text] +
                  [f'{col_sig}_{i + 1}' for i in range(num_signals_per_item)] +
                  [f'{col_exp_prefix}_{i + 1}' for i in range(num_signals_per_item)]
        )

def extract_label_fix(label_text, colnames):
    """
    Extracts the raw dictionary, signal_label, reason, and explanation from the raw GPT output.

    Args:
        label_text (str): Raw GPT output as a JSON string.
        colnames (list): List of column names to map the extracted values.

    Returns:
        pd.Series: A Pandas Series containing the extracted values, mapped to the given column names.
    """
    try:
        # Clean the input to remove Markdown-like formatting
        cleaned_text = label_text.strip().lstrip('```json').rstrip('```').strip()

        # Parse the cleaned JSON
        first_dict = json.loads(cleaned_text)

        # Extract fields if they exist
        signal_label = first_dict.get('signal_label', None)
        reason = first_dict.get('reason', '')
        explanation = first_dict.get('explanation', '')

    except json.JSONDecodeError as e:
        print(f"Couldn't properly parse the dict string in {label_text}: {e}")
        first_dict = None
        signal_label = None
        reason = ''
        explanation = ''

    # Return the extracted values as a Pandas Series
    return pd.Series([first_dict, signal_label, reason, explanation], index=colnames)


def remove_sigs_with_forbidden_words(data_pdf, models_to_apply, forbidden_words):
    data_pdf_filtered = data_pdf.__deepcopy__()
    if 'mistral' in models_to_apply:
        data_pdf_filtered['mistral_valid'] = ~data_pdf_filtered['gen_sig_mistral'].astype(str).str.contains('|'.join(forbidden_words), case=False)
    else:
        data_pdf_filtered['mistral_valid'] = True
    if 'gpt' in models_to_apply:
        data_pdf_filtered['gpt_valid'] = ~data_pdf_filtered['gen_sig_gpt'].astype(str).str.contains('|'.join(forbidden_words), case=False)
    else:
        data_pdf_filtered['gpt_valid'] = True
    if '|' in models_to_apply:
        data_pdf_filtered['valid'] = data_pdf_filtered['mistral_valid'] | data_pdf_filtered['gpt_valid']
    else:
        data_pdf_filtered['valid'] = data_pdf_filtered['mistral_valid'] & data_pdf_filtered['gpt_valid']
    return data_pdf_filtered[data_pdf_filtered['valid']], data_pdf_filtered[~data_pdf_filtered['valid']]

# Removes rows for which the generated signal is not entirely included in a sentence
def remove_not_in_sentence(data_pdf, models_to_apply):
    data_pdf_filtered = data_pdf.__deepcopy__()
    data_pdf_filtered["all_text"] = data_pdf_filtered["title"].astype(str) + " " + data_pdf_filtered["aspects"].astype(str) + " " + data_pdf_filtered["desc"].astype(str)
    # check that gen_sig_mistral is present in the all_text
    if 'mistral' in models_to_apply:
        data_pdf_filtered['mistral_in_text'] = data_pdf_filtered.apply(lambda x: signal_match_text_relaxed(x["all_text"], x[f"gen_sig_mistral"]), axis=1)
    else:
        data_pdf_filtered['mistral_in_text'] = True
    if 'gpt' in models_to_apply:
        data_pdf_filtered['gpt_in_text'] = data_pdf_filtered.apply(
            lambda x: signal_match_text_relaxed(x["all_text"], x[f"gen_sig_gpt"]), axis=1)
    else:
        data_pdf_filtered['gpt_in_text'] = True
    if '|' in models_to_apply:
        data_pdf_filtered['in_text'] = data_pdf_filtered['mistral_in_text'] | data_pdf_filtered['gpt_in_text']
    else:
        data_pdf_filtered['in_text'] = data_pdf_filtered['mistral_in_text'] & data_pdf_filtered['gpt_in_text']

    return data_pdf_filtered[data_pdf_filtered['in_text']], data_pdf_filtered[~data_pdf_filtered['in_text']]

# models_to_apply: 'mistral', 'gpt', 'mistral|gpt', 'mistral&gpt'
def curate_gen_sigs_data(data_pdf, how=['fix_non_extracted_sigs','filter_invalid_aspects',
                                        'remove_sigs_with_forbidden_words', 'remove_not_in_sentence'],
                        models_to_apply = 'mistral|gpt', forbidden_words = forbidden_words):
    num_items = len(data_pdf)
    print(f"Starting with {num_items} items")
    # Fix misparsing, e.g. mistral generating non-json output.
    if 'fix_non_extracted_sigs' in how:
        # Handle cases where the models (esp. Misral) generate text which is not in json form.
        data_pdf.loc[data_pdf['gen_sig_mistral'] == '', ['gen_text_mistral', 'gen_sig_mistral']] = data_pdf.loc[
            data_pdf['gen_sig_mistral'] == '', 'gen_text_mistral'].apply(
            lambda r: extract_sig_fix(r, ['gen_text_mistral', 'gen_sig_mistral']))
        data_pdf.loc[data_pdf['gen_sig_gpt'] == '', 'gen_sig_gpt'] = data_pdf.loc[
            data_pdf['gen_sig_gpt'] == '',  ['gen_text_gpt', 'gen_sig_gpt']].apply(
            lambda r: extract_sig_fix(r, ['gen_text_gpt', 'gen_sig_gpt']))
        data_pdf = data_pdf[(data_pdf['gen_sig_mistral'] != '') & (data_pdf['gen_sig_gpt'] != '')]
        lenf = len(data_pdf)
        print(f'after fixing non extrated sigs: {lenf} ({lenf / num_items * 100:.2f}%)')
    # This is due to bug that the filtered dict is not properly saved as parquet. Basically should have been done before
    # signals generation
    if 'filter_invalid_aspects' in how:
        data_pdf['aspects'] = data_pdf['aspects'].apply(filter_aspects)
    if 'remove_sigs_with_forbidden_words' in how:
        data_pdf_filtered, sigs_with_forbidden = remove_sigs_with_forbidden_words(data_pdf, models_to_apply, forbidden_words)
        lenf = len(data_pdf_filtered)
        print(f'after filtering forbidden words: {lenf} ({lenf/num_items*100:.2f}%)')
    if 'remove_not_in_sentence' in how:
        data_pdf_filtered, bad_rows_sentence = remove_not_in_sentence(data_pdf_filtered, models_to_apply)
        lenf = len(data_pdf_filtered)
        print(f'after filtering non-sentence signals: {lenf} ({lenf/num_items*100:.2f}%)')

    return data_pdf_filtered, sigs_with_forbidden, bad_rows_sentence



def merge_mistral_gpt_dfs(gen_gpt_pdf_path, gen_mistral_pdf_path, models_to_apply = 'mistral|gpt'):
    df_collct_gpt = pd.read_parquet(gen_gpt_pdf_path)
    df_collct_mistral = pd.read_parquet(gen_mistral_pdf_path)
    if not all(df_collct_mistral['item_id'] == df_collct_gpt['item_id']):
        raise ValueError("The item ids in the two dataframes are not aligned")
    df_collct_mistral[['gen_text_gpt', 'gen_sig_gpt']] = df_collct_gpt[['gen_text_gpt', 'gen_sig_gpt']]
    data_pdf = df_collct_mistral
    curated_pdf, sigs_with_forbidden, bad_rows_sentence = curate_gen_sigs_data(data_pdf, models_to_apply=models_to_apply)
    return curated_pdf, sigs_with_forbidden, bad_rows_sentence



    # # pretty print first 30 gen_sig_mistral values of the filtered data in camel case
    # print(data_filtered["gen_sig_mistral"].head(30).apply(lambda x: x.title()))


# This is implemented since json.loads() didn't work when the generated text (e.g. sales pitch) had quotes in it.
def extract_sig_from_gen_txt(json_string):

    if json_string is None:
        print("json_string is None, can't extract signal from it!")
        return ''
    try:

        data_pdf.loc[data_pdf['gen_sig_mistral'] == '', ['gen_text_mistral', 'gen_sig_mistral']] = data_pdf.loc[
            data_pdf['gen_sig_mistral'] == '', 'gen_text_mistral'].apply(
            lambda r: extract_sig_fix(r, ['gen_text_mistral', 'gen_sig_mistral']))


        json_dict = ast.literal_eval(json_string.strip('.'))
        if 'characteristic' not in json_dict:
            print(f" 'characteristic' not in json_dict: {json_dict}")
            res = ''
            # if 'error' in json_dict:
            #     res = json_dict.get('error')
            # else:
            #     res = "No characteristic found in the generated text"
        else:
            res = json_dict.get('characteristic')

        return res
    except Exception as e:
        if debug:
            print(e)
            traceback.print_exc()
        return ''


''''
model_name - mistral, gpt4 
'''
async def generate_signals_per_item_multi(all_items_df, cols_for_prompt=['title', 'aspects', 'desc'], max_items=None,
                                          model_name='', outfile=None, debug=False,
                                          prompt_instructions_version="default",
                                          num_signals_per_item=5):
    """ Generate multiple signals per item based on the updated logic. """

    if 'gpt' in model_name:
        signals_generator = SignalsGeneratorGPT(cols_for_prompt=cols_for_prompt,
                                                prompt_instructions_version=prompt_instructions_version,
                                                num_signals_per_item=num_signals_per_item)
    elif 'mistral' in model_name:
        signals_generator = SignalsGeneratorMistral(cols_for_prompt=cols_for_prompt)
    else:
        raise ValueError(f"model_name: {model_name} is not supported")

    if max_items is None:
        max_items = len(all_items_df)

    if outfile is not None:
        with open(outfile, 'w') as file:
            file.write('')

    for ind, item in all_items_df[:max_items].iterrows():
        if debug:
            print(f"Index: {ind}, item: {item['item_id']}")

        # Generate signals (expecting multiple signals)
        gen_text = await signals_generator.generate_sig(item.to_frame().T, debug=debug)

        if outfile is not None:
            with open(outfile, "a") as f:
                json.dump(gen_text, f, ensure_ascii=False)
                f.write("\n")

        if num_signals_per_item == 1:
            all_items_df.loc[ind, [f'gen_text_{model_name}', f'gen_sig_{model_name}', f'gen_exp_{model_name}']] = \
                extract_sig_fix_multi(gen_text,
                                      [f'gen_text_{model_name}', f'gen_sig_{model_name}', f'gen_exp_{model_name}'],
                                      num_signals_per_item)

        else:
            extracted_values = extract_sig_fix_multi(gen_text, [f'gen_text_{model_name}', f'gen_sig_{model_name}'],
                                                     num_signals_per_item)

            expected_cols = [f'gen_text_{model_name}'] + \
                            [f'gen_sig_{model_name}_{i + 1}' for i in range(num_signals_per_item)] + \
                            [f'gen_exp_{model_name}_{i + 1}' for i in range(num_signals_per_item)]

            all_items_df.loc[ind, expected_cols] = extracted_values

    return all_items_df


async def generate_signals_per_item(all_items_df, cols_for_prompt=['title', 'aspects', 'desc'], max_items=None,
                                    model_name='', outfile=None, debug=False, prompt_instructions_version="default"):
    if model_name == 'gpt':
        signals_generator = SignalsGeneratorGPT(cols_for_prompt=cols_for_prompt,
                                                prompt_instructions_version=prompt_instructions_version)
    elif model_name == 'mistral':
        signals_generator = SignalsGeneratorMistral(cols_for_prompt=cols_for_prompt)
    else:
        raise ValueError(f"model_name: {model_name} is not supported")
    if max_items is None:
        max_items = len(all_items_df)

    if outfile is not None:
        with open(outfile, 'w') as file:
            file.write('')

    for ind, item in all_items_df[:max_items].iterrows():
        if verbose or debug:
            #print(f"item: {item['item_id']}")
            print(f"Index: {ind}, item: {item['item_id']}")
        # num_trials2 = 0
        gen_text = await signals_generator.generate_sig(item.to_frame().T,
                                                        debug=debug)  # Passing a dataframe since gpt expcets one.
        if outfile is not None:
            with open(outfile, "a") as f:
                json.dump(gen_text, f, ensure_ascii=False)
                f.write("\n")

        all_items_df.loc[ind, [f'gen_text_{model_name}', f'gen_sig_{model_name}']] = \
            extract_sig_fix(gen_text, [f'gen_text_{model_name}', f'gen_sig_{model_name}'])

        # all_items_df.loc[ind, f'gen_text_{model_name}'] = gen_text
        # all_items_df.loc[ind, f'gen_sig_{model_name}'] = extract_sig_from_gen_txt(gen_text)

    return all_items_df


def expand_signals_dataframe(df, id_column: str, text_columns: list, signal_prefix: str, explanation_prefix: str):
    """
    Expands a dataframe so that each signal and its corresponding explanation become separate rows.

    Parameters:
    - df: pandas DataFrame, input dataframe containing signals and explanations.
    - id_column: str, the name of the column representing unique item IDs.
    - text_columns: list, columns that should be retained in each row (e.g., 'title', 'aspects', 'desc').
    - signal_prefix: str, the prefix of generated signal columns (e.g., "gen_sig_gpt").
    - explanation_prefix: str, the prefix of generated explanation columns (e.g., "gen_exp_gpt").

    Returns:
    - pandas DataFrame, expanded dataframe with each signal and explanation in separate rows.
    """
    records = []

    for _, row in df.iterrows():
        for i in range(1, 6):  # Assuming max 5 signals per item
            signal_col = f"{signal_prefix}_{i}"
            explanation_col = f"{explanation_prefix}_{i}"

            # Check if signal is non-empty
            if pd.notna(row[signal_col]) and row[signal_col] != "":
                record = {
                    id_column: row[id_column],
                    signal_prefix: row[signal_col],  # Flattening signals
                    explanation_prefix: row[explanation_col]  # Flattening explanations
                }
                # Include the other specified columns
                for col in text_columns:
                    record[col] = row[col]

                records.append(record)

    # Convert to DataFrame
    expanded_df = pd.DataFrame(records)

    return expanded_df

async def generate_labels_per_item(all_items_df, cols_for_prompt=['title', 'aspects', 'desc', 'signal'],
                                   max_items=None, model_name='', outfile=None, debug=False):
    if model_name == 'gpt':
        label_generator = SignalLabelGeneratorGPT(cols_for_prompt=cols_for_prompt)
    elif model_name == 'mistral':
        label_generator = SignalsGeneratorMistral(cols_for_prompt=cols_for_prompt)  # Placeholder for Mistral class
    else:
        raise ValueError(f"model_name: {model_name} is not supported")

    if max_items is None:
        max_items = len(all_items_df)

    if outfile is not None:
        with open(outfile, 'w') as file:
            file.write('')

    for ind, item in all_items_df[:max_items].iterrows():
        if debug:
            print(f"Processing item ID: {item['item_id']}")

        label_text = await label_generator.generate_label(item.to_frame().T, debug=debug)
        if outfile is not None:
            with open(outfile, "a") as f:
                json.dump(label_text, f, ensure_ascii=False)
                f.write("\n")

        # Store the label and explanation in the DataFrame
        # all_items_df.loc[ind, f'label_text_{model_name}'] = label_text
        # Store the raw dict, label, reason, and explanation in the DataFrame
        all_items_df.loc[
            ind, ['raw_dict', f'label_{model_name}', f'label_reason_{model_name}', f'label_explanation_{model_name}']] = \
            extract_label_fix(label_text, ['raw_dict', f'label_{model_name}', f'label_reason_{model_name}',
                                           f'label_explanation_{model_name}'])
    # print (label_generator.generate_prompt('title', 'aspects', 'desc', 'signal'))

    return all_items_df
#------------------------Post processing---------------------------
#------------------------clean text--------------------------------


# Clean text function (independent and reusable)
def clean_text(text):
    """
    Clean text by replacing unwanted characters, removing duplicate spaces,
    and converting to lowercase.
    """
    if isinstance(text, str):  # Only process strings
        text = re.sub(r"[^a-zA-Z0-9 x./]", " ", text)  # Replace unwanted characters with space
        return re.sub(r'\s+', ' ', text.lower()).strip()  # Remove duplicate spaces and strip
    return text  # Return non-string data unchanged


# Function to clean text columns in a DataFrame
def clean_text_columns(df, columns=None, original_col_mapping=None):
    """
    Clean text columns in a DataFrame using the `clean_text` function.
    Ensures columns are strings and handles NaN values.
    Can optionally save the original values of specific columns to new columns.
    Removes rows where `gen_sig_gpt` is an empty string.

    Parameters:
    - df (pd.DataFrame): The DataFrame to clean.
    - columns (list or None): A list of column names to clean. If None, all columns are cleaned.
    - original_col_mapping (dict or None): A mapping of columns to their original value columns
      (e.g., {'gen_sig_gpt': 'gen_sig_gpt_orig'}) to save the original values.

    Returns:
    - pd.DataFrame: A deep copy of the DataFrame with the cleaned text and filtered rows.
    """
    # Create a deep copy of the DataFrame
    df_cleaned = df.copy(deep=True)

    # Ensure specified columns are strings and handle NaN values
    if columns:
        df_cleaned[columns] = df_cleaned[columns].fillna('').astype(str)

    # Save original values if mapping is provided
    if original_col_mapping:
        for col, orig_col in original_col_mapping.items():
            if col in df_cleaned.columns:
                df_cleaned[orig_col] = df_cleaned[col]

    # Apply cleaning
    if columns:  # Clean specific columns
        for column in columns:
            if column in df_cleaned.columns:
                df_cleaned[column] = df_cleaned[column].apply(clean_text)
    else:  # Clean all columns
        df_cleaned = df_cleaned.applymap(clean_text)

    # Remove rows where `gen_sig_gpt` is an empty string
    if 'gen_sig_gpt' in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned['gen_sig_gpt'] != '']

    return df_cleaned

def reorder_subset_columns(df, columns):
    """
    Creates a deep copy of the DataFrame, selects a subset of columns by names, 
    and reorders them according to the input order.

    Parameters:
    - df (pd.DataFrame): The original DataFrame.
    - columns (list): A list of column names to subset and reorder.

    Returns:
    - pd.DataFrame: A deep copy of the DataFrame with only the selected columns, reordered.
    """
    # Create a deep copy of the DataFrame
    df_copy = df.copy(deep=True)
    
    # Ensure the columns exist in the DataFrame
    valid_columns = [col for col in columns if col in df_copy.columns]
    
    # Select and reorder the subset of columns
    return df_copy[valid_columns] 

#------------------------Post processing---------------------------
#------------------------coverage----------------------------------
# Function to check for exact sequential substring match
def check_exact_match(gen_sig, column_text):
    """Check for exact sequential substring match with normalization."""
    gen_sig_clean = ' '.join(gen_sig.lower().split())  # Normalize spaces and case
    column_text_clean = ' '.join(column_text.lower().split())
    return gen_sig_clean in column_text_clean

def check_permutation_in_sequence(gen_sig, column_text):
    """
    Check if words in gen_sig appear as a contiguous permutation subset in column_text.
    """
    gen_words = [word for word in gen_sig.split() if len(word) > 1]
    col_words = [word for word in column_text.split() if len(word) > 1]
    

    permutations = [' '.join(p) for p in itertools.permutations(gen_words)]
    
    # searching perms in text. 
    col_text = ' '.join(col_words)  
    for perm in permutations:
        if perm in col_text:
            return True
    return False

def check_ngram_match(gen_sig, column_text, threshold=0.5):
    """
    Check for n-grams in the column text and additional criteria:
    1. If all non-ngram tokens appear in the column text (not in order).
    2. Count how many times the matched n-grams appear.
    """
    # Normalize and filter words with length > 1
    gen_words = [word.lower() for word in gen_sig.split() if len(word) > 1]
    col_words = [word.lower() for word in column_text.split() if len(word) > 1]

    n = len(gen_words)
    if n < 2:  # If gen_sig has fewer than 2 valid words, return no match
        return False, False, 0, False, {}, set(), []  # Added empty list for ngrams

    # Dynamically set the n-gram sizes to check (up to len(gen_words) - 1)
    sizes_to_check = [size for size in range(2, n)]  # n-grams from size 2 to n-1

    # Generate n-grams based on sizes_to_check
    ngrams = []
    for size in sizes_to_check:
        ngrams += [' '.join(gen_words[i:i + size]) for i in range(len(gen_words) - size + 1)]

    # Check for n-gram matches in column_text
    matches = []
    match_counts = {}
    for ngram in ngrams:
        ngram_words = ngram.split()
        count = 0
        for i in range(len(col_words) - len(ngram_words) + 1):
            if ' '.join(col_words[i:i + len(ngram_words)]) == ngram:
                matches.append(ngram)
                count += 1
        if count > 0:
            match_counts[ngram] = count

    # If we have found any n-gram matches, check non-matching tokens
    matched_ngram_words = set(word for ngram in matches for word in ngram.split())
    non_ngram_words = set(gen_words) - matched_ngram_words

    # Check if all non-ngram words are found in the column text
    non_ngram_match = all(word in col_words for word in non_ngram_words)

    # Calculate coverage as the percentage of matched n-grams
    all_matched_ngrams = set(matches)  # Ensure unique matches
    unique_matches_count = len(all_matched_ngrams)
    total_ngrams_count = len(set(ngrams))  # Use the set of all generated n-grams

    coverage = unique_matches_count / total_ngrams_count if total_ngrams_count > 0 else 0

    # Return values:
    # - Whether any n-gram matches were found
    # - Whether coverage exceeds the threshold
    # - Coverage percentage
    # - Non-ngram match flag
    # - Match counts (as a dictionary)
    # - Unique matched n-grams (as a set)
    # - All generated ngrams (as a list)
    return len(all_matched_ngrams) > 0, coverage >= threshold, coverage, non_ngram_match, match_counts, all_matched_ngrams, ngrams
def calculate_coverage(all_matched_ngrams, all_ngrams):
    """Calculate the coverage of matched n-grams."""
    unique_matches_count = len(all_matched_ngrams)
    total_ngrams_count = len(all_ngrams)
    return unique_matches_count / total_ngrams_count if total_ngrams_count > 0 else 0


def find_exact_match(gen_sig, row):
    """Check for exact matches across specified columns."""
    for col in ['title', 'aspects', 'desc']:
        if check_exact_match(gen_sig, str(row[col])):
            return 'exact', col
    return None, None


def find_permutation_match(gen_sig, row):
    """Check for permutation matches across specified columns."""
    for col in ['title', 'aspects', 'desc']:
        if check_permutation_in_sequence(gen_sig, str(row[col])):
            return 'permutation', col
    return None, None


def find_ngram_match(gen_sig, row):
    """Check for n-gram matches across specified columns."""
    best_match_col = None
    best_match_score = -1
    best_coverage = 0
    all_matched_ngrams = []  # Use a list to track all matched n-grams
    all_ngrams = set()
    match_counts = {}
    non_ngram_match = False

    for col in ['title', 'aspects', 'desc']:
        column_text = str(row[col])
        (
            ngram_match_found,
            coverage_above_threshold,
            coverage_in_col,
            non_ngram_match_in_col,
            col_match_counts,
            matched_ngrams,
            generated_ngrams,
        ) = check_ngram_match(gen_sig, column_text, threshold=0.5)

        if ngram_match_found:
            all_matched_ngrams.extend(matched_ngrams)  # Append to the list
            all_ngrams.update(generated_ngrams)  # Use set to ensure unique generated n-grams

            for ngram in matched_ngrams:
                match_counts[ngram] = match_counts.get(ngram, 0) + 1

            longest_ngram = max((len(ngram.split()) for ngram in matched_ngrams), default=0)
            match_score = (1 if non_ngram_match_in_col else 0) * 1000 + longest_ngram

            if match_score > best_match_score or (
                match_score == best_match_score and coverage_in_col > best_coverage
            ):
                best_match_col = col
                best_match_score = match_score
                best_coverage = coverage_in_col

        if non_ngram_match_in_col:
            non_ngram_match = True

    coverage = calculate_coverage(all_matched_ngrams, all_ngrams)
    return best_match_col, coverage, non_ngram_match, all_matched_ngrams, all_ngrams


def process_matches(df):
    """
    Processes matches for each row in the DataFrame.
    Adds multiple columns to reflect match types, n-gram coverage, and related metrics.
    Returns a deep copy of the processed DataFrame.
    """
    # Create a deep copy of the DataFrame
    df_copy = df.copy(deep=True)

    # Initialize lists to store results
    match_types = []
    match_columns = []
    ngram_coverages = []
    non_ngram_matches = []
    ngram_match_counts = []
    all_ngrams_list = []

    for _, row in df_copy.iterrows():
        gen_sig = str(row['gen_sig_gpt'])

        # Step 1: Check exact matches
        match_found, match_col = find_exact_match(gen_sig, row)

        # Step 2: Check permutation matches
        if not match_found:
            match_found, match_col = find_permutation_match(gen_sig, row)

        # Step 3: Check n-gram matches
        if not match_found:
            match_col, coverage, non_ngram_match, all_matched_ngrams, all_ngrams = find_ngram_match(gen_sig, row)
            match_found = 'ngram' if len(all_matched_ngrams) > 0 else 'no_match'

            if non_ngram_match and match_found == 'ngram':
                match_found = 'ngram+non ngram'

            # Append values for n-gram specific metrics
            ngram_coverages.append(coverage)
            non_ngram_matches.append(non_ngram_match)
            ngram_match_counts.append({ngram: all_matched_ngrams.count(ngram) for ngram in set(all_matched_ngrams)})
            all_ngrams_list.append(list(all_ngrams))
        else:
            # Default values if no n-gram match is found
            ngram_coverages.append(0)
            non_ngram_matches.append(False)
            ngram_match_counts.append({})
            all_ngrams_list.append([])

        # Append results for match type and column
        match_types.append(match_found)
        match_columns.append(match_col)

    # Add results to the DataFrame
    df_copy['match_type'] = match_types
    df_copy['match_column'] = match_columns
    df_copy['ngram_coverage'] = ngram_coverages
    df_copy['non_ngram_match'] = non_ngram_matches
    df_copy['ngram_match_counts'] = ngram_match_counts
    df_copy['all_ngrams'] = all_ngrams_list

    return df_copy

def calculate_match_summary(df_match, match_column):
    """
    Calculate match counts and percentages for a given column in the match DataFrame.

    Parameters:
    - df_match (pd.DataFrame): The DataFrame containing match data.
    - match_column (str): The column name for which to calculate match counts.

    Returns:
    - pd.DataFrame: A summary DataFrame with metrics, counts, and percentages.
    """
    # Calculate match counts and percentages
    match_counts = df_match[match_column].value_counts()
    match_percentages = (match_counts / len(df_match)) * 100

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'Match_type': match_counts.index,
        'Count': match_counts.values,
        'Percentage': match_percentages.values.round(1)
    })

    # Add row for total
    total_count = len(df_match)
    summary_df.loc[len(summary_df)] = ['Total Rows', total_count, 100.00]

    return summary_df 
#------------------------Post processing---------------------------------------
#------------------------to_excel_highlighted----------------------------------

import re
from tqdm import tqdm
import xlsxwriter

def clean_html_tags(text):
    """
    Remove HTML tags from the input text.
    """
    return re.sub(r'<[^>]*>', '', text)

def highlight_text_to_xsl(df, gensig_colname='gen_sig_gpt', source_colnames=['title', 'aspects', 'desc'],
                       output_excelfile_path='highlighted_text.xlsx', dropdown_config=None):

    # Ensure item_id is a decimal string
    df = df[df["item_id"].astype(str).str.isdigit()]
    df['item_id'] = df['item_id'].apply(lambda x: f'{int(x):.0f}')
   # df["ngram_match_counts"] = df["ngram_match_counts"].astype(str)
  #  df["all_ngrams"] = df["all_ngrams"].astype(str)
        # Convert all other columns to string but skip item_id and gen_sig_gpt
    for col in df.columns:
        if col not in ['item_id', gensig_colname]:  # Avoid converting these columns
            df[col] = df[col].astype(str)
    # Ensure there are no leading/trailing spaces and convert to string
    df[gensig_colname] = df[gensig_colname].astype(str).str.strip()
    # Filter rows where gen_sig_gpt is NOT empty
    df = df[df[gensig_colname] != ""]

    # Clean HTML tags from the source columns
    for col in source_colnames:
        if col in df.columns:
            df[col] = df[col].apply(clean_html_tags)

    with xlsxwriter.Workbook(output_excelfile_path) as workbook:
        # Create a worksheet with a title
        worksheet = workbook.add_worksheet('Highlighted')

        # Set up some formats to use
        black = workbook.add_format({"color": "black"})
        gray = workbook.add_format({"color": "gray"})
        red = workbook.add_format({"color": "red"})
        cell_format_header = workbook.add_format({"bold": True})
        cell_format_wrap = workbook.add_format({})

        # Calculate column widths
        col_widths = {}
        for col in df.columns:
            col_widths[col] = 20

        # Set column widths
        for col_idx, col_name in enumerate(df.columns):
            worksheet.set_column(col_idx, col_idx, col_widths[col_name])

        # Insert a header row
        for col_idx, col_name in enumerate(df.columns):
            worksheet.write(0, col_idx, col_name, cell_format_header)

        row_idx = 1

        # Process and insert rows
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            tokens = [token for token in re.split(r'[ ,:{}\']', row[gensig_colname]) if token]

            for col_idx, col_name in enumerate(df.columns):
                cell_value = row[col_name]
                if col_name in source_colnames and isinstance(cell_value, str):
                    # Create the highlighted description
                    rich_texts = []
                    positions = []

                    # Find all occurrences of each token in the cell_value
                    for token in tokens:
                        if token and cell_value:
                            start = 0
                            while start < len(cell_value):
                                token_start = cell_value.lower().find(token.lower(), start)
                                if token_start == -1:
                                    break
                                positions.append((token_start, token))
                                start = token_start + len(token)

                    # Sort positions by the starting index
                    positions.sort()

                    # Construct the rich_texts list based on the sorted positions
                    start = 0
                    for token_start, token in positions:
                        if token_start > start:
                            rich_texts.append(cell_value[start:token_start])
                        rich_texts.extend([red, token])
                        start = token_start + len(token)
                    if start < len(cell_value) and start > 0:
                        rich_texts.append(cell_value[start:])

                    # Write the highlighted description
                    if rich_texts:
                        worksheet.write_rich_string(row_idx, col_idx, *rich_texts)
                    else:
                        worksheet.write(row_idx, col_idx, cell_value, cell_format_wrap)
                elif col_name == gensig_colname:
                    cell_val = []
                    # Write the 'gen_sig_gpt' column as a rich string (black color)
                    if cell_value:
                        cell_val.append(' ')
                        cell_val.extend([black, cell_value])
                        worksheet.write_rich_string(row_idx, col_idx, *cell_val)
                    else:
                        worksheet.write(row_idx, col_idx, cell_value, cell_format_wrap)
                elif col_name == 'item_id':
                    # Add hyperlink to item_id column
                    url = f'https://www.ebay.com/itm/{cell_value}'
                    worksheet.write_url(row_idx, col_idx, url, string=cell_value)
                else:
                    worksheet.write(row_idx, col_idx, cell_value, cell_format_wrap)

            row_idx += 1
            # Add dropdowns if a configuration is provided
        if dropdown_config:
            headers = {col_name: idx for idx, col_name in enumerate(df.columns)}
            for column_name, dropdown_values in dropdown_config.items():
                if column_name not in headers:
                    print(f"Column '{column_name}' not found in the DataFrame.")
                    continue

                col_idx = headers[column_name]
                # Define data validation for the column
                worksheet.data_validation(1, col_idx, len(df), col_idx, {
                    'validate': 'list',
                    'source': dropdown_values,
                    'error_message': f"Please select a valid option from {dropdown_values}.",
                })

    print(f"Excel file with dropdowns saved to {output_excelfile_path}.")
def highlight_text_to_xsl_orig(df, gensig_colname='gen_sig_gpt', source_colnames=['title', 'aspects', 'desc'],
                       output_excelfile_path='highlighted_text.xlsx'):
#returns hyperlink for item_id.
    # Ensure item_id is a decimal string
    df = df[df["item_id"].astype(str).str.isdigit()]
    df['item_id'] = df['item_id'].apply(lambda x: f'{int(x):.0f}')
   # df["ngram_match_counts"] = df["ngram_match_counts"].astype(str)
  #  df["all_ngrams"] = df["all_ngrams"].astype(str)
        # Convert all other columns to string but skip item_id and gen_sig_gpt
    for col in df.columns:
        if col not in ['item_id', gensig_colname]:  # Avoid converting these columns
            df[col] = df[col].astype(str)
    # Ensure there are no leading/trailing spaces and convert to string
    df[gensig_colname] = df[gensig_colname].astype(str).str.strip()
    # Filter rows where gen_sig_gpt is NOT empty
    df = df[df[gensig_colname] != ""]

    # Clean HTML tags from the source columns
    for col in source_colnames:
        if col in df.columns:
            df[col] = df[col].apply(clean_html_tags)

    with xlsxwriter.Workbook(output_excelfile_path) as workbook:
        # Create a worksheet with a title
        worksheet = workbook.add_worksheet('Highlighted')

        # Set up some formats to use
        black = workbook.add_format({"color": "black"})
        gray = workbook.add_format({"color": "gray"})
        red = workbook.add_format({"color": "red"})
        cell_format_header = workbook.add_format({"bold": True})
        cell_format_wrap = workbook.add_format({})

        # Calculate column widths
        col_widths = {}
        for col in df.columns:
            col_widths[col] = 20

        # Set column widths
        for col_idx, col_name in enumerate(df.columns):
            worksheet.set_column(col_idx, col_idx, col_widths[col_name])

        # Insert a header row
        for col_idx, col_name in enumerate(df.columns):
            worksheet.write(0, col_idx, col_name, cell_format_header)

        row_idx = 1

        # Process and insert rows
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            tokens = [token for token in re.split(r'[ ,:{}\']', row[gensig_colname]) if token]

            for col_idx, col_name in enumerate(df.columns):
                cell_value = row[col_name]
                if col_name in source_colnames and isinstance(cell_value, str):
                    # Create the highlighted description
                    rich_texts = []
                    positions = []

                    # Find all occurrences of each token in the cell_value
                    for token in tokens:
                        if token and cell_value:
                            start = 0
                            while start < len(cell_value):
                                token_start = cell_value.lower().find(token.lower(), start)
                                if token_start == -1:
                                    break
                                positions.append((token_start, token))
                                start = token_start + len(token)

                    # Sort positions by the starting index
                    positions.sort()

                    # Construct the rich_texts list based on the sorted positions
                    start = 0
                    for token_start, token in positions:
                        if token_start > start:
                            rich_texts.append(cell_value[start:token_start])
                        rich_texts.extend([red, token])
                        start = token_start + len(token)
                    if start < len(cell_value) and start > 0:
                        rich_texts.append(cell_value[start:])

                    # Write the highlighted description
                    if rich_texts:
                        worksheet.write_rich_string(row_idx, col_idx, *rich_texts)
                    else:
                        worksheet.write(row_idx, col_idx, cell_value, cell_format_wrap)
                elif col_name == gensig_colname:
                    cell_val = []
                    # Write the 'gen_sig_gpt' column as a rich string (black color)
                    if cell_value:
                        cell_val.append(' ')
                        cell_val.extend([black, cell_value])
                        worksheet.write_rich_string(row_idx, col_idx, *cell_val)
                    else:
                        worksheet.write(row_idx, col_idx, cell_value, cell_format_wrap)
                elif col_name == 'item_id':
                    # Add hyperlink to item_id column
                    url = f'https://www.ebay.com/itm/{cell_value}'
                    worksheet.write_url(row_idx, col_idx, url, string=cell_value)
                else:
                    worksheet.write(row_idx, col_idx, cell_value, cell_format_wrap)

            row_idx += 1
def save_highlighted_files_by_annotator(df, gensig_colname, source_colnames, base_path, file_name, annotators_names,annotator_colname='annotator_name', dropdown_config=None):
    """
    Filters rows by annotator name and saves highlighted text to separate Excel files.

    Args:
        df (pd.DataFrame): The full DataFrame to filter and process.
        gensig_colname (str): Column name for general signals.
        source_colnames (list): Columns to process for highlighting.
        base_path (str): Base path to save the files.
        file_name (str): Base file name to modify for each annotator.
        annotators_names (list): List of annotator names to filter by and generate files for.
        annotator_colname (str): Column name containing annotator names in the DataFrame.
        dropdown_config (dict, optional): Configuration for adding dropdowns to specific columns.

    Returns:
        None
    """
    # Ensure the base path exists
    os.makedirs(base_path, exist_ok=True)

    for annotator in annotators_names:
        # Filter DataFrame for the current annotator
        annotator_df = df[df[annotator_colname] == annotator]

        if annotator_df.empty:
            print(f"No rows found for annotator: {annotator}")
            continue

        # Generate the file name with the annotator's name as a suffix
        annotator_file_name = file_name.replace('.xlsx', f'_{annotator}.xlsx')
        output_excel_path = os.path.join(base_path, annotator_file_name)

        print(f"Generating file for annotator: {annotator}")
        print(f"Saving to: {output_excel_path}")

        # Call the highlight function for the filtered DataFrame
        highlight_text_to_xsl(
            df=annotator_df,
            gensig_colname=gensig_colname,
            source_colnames=source_colnames,
            output_excelfile_path=output_excel_path,
            dropdown_config=dropdown_config  # Pass the dropdown configuration
        )

    print("All files have been successfully generated!")


def highlight_text_to_xsl_cleand_df(df, gensig_colname='gen_sig_gpt', source_colnames=['title', 'aspects', 'desc'],
                       output_excelfile_path='highlighted_text.xlsx'):
    # Ensure item_id is a decimal string
    df = df[df["item_id"].astype(str).str.isdigit()]
    df['item_id'] = df['item_id'].apply(lambda x: f'{int(x):.0f}')
   # df["ngram_match_counts"] = df["ngram_match_counts"].astype(str)
  #  df["all_ngrams"] = df["all_ngrams"].astype(str)
        # Convert all other columns to string but skip item_id and gen_sig_gpt
    for col in df.columns:
        if col not in ['item_id', gensig_colname]:  # Avoid converting these columns
            df[col] = df[col].astype(str)
    # Ensure there are no leading/trailing spaces and convert to string
    df["gen_sig_gpt"] = df["gen_sig_gpt"].astype(str).str.strip()
    # Filter rows where gen_sig_gpt is NOT empty
    df = df[df["gen_sig_gpt"] != ""]

    # Clean HTML tags from the source columns
    for col in source_colnames:
        if col in df.columns:
            df[col] = df[col].apply(clean_html_tags)

    with xlsxwriter.Workbook(output_excelfile_path) as workbook:
        # Create a worksheet with a title
        worksheet = workbook.add_worksheet('Highlighted')

        # Set up some formats to use
        black = workbook.add_format({"color": "black"})
        gray = workbook.add_format({"color": "gray"})
        red = workbook.add_format({"color": "red"})
        cell_format_header = workbook.add_format({"bold": True})
        cell_format_wrap = workbook.add_format({})

        # Calculate column widths
        col_widths = {}
        for col in df.columns:
            col_widths[col] = 20

        # Set column widths
        for col_idx, col_name in enumerate(df.columns):
            worksheet.set_column(col_idx, col_idx, col_widths[col_name])

        # Insert a header row
        for col_idx, col_name in enumerate(df.columns):
            worksheet.write(0, col_idx, col_name, cell_format_header)

        row_idx = 1

        # Process and insert rows
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            tokens = row[gensig_colname].split()

            for col_idx, col_name in enumerate(df.columns):
                cell_value = row[col_name]
                if col_name in source_colnames and isinstance(cell_value, str):
                    # Create the highlighted description
                    rich_texts = []
                    positions = []

                    # Find all occurrences of each token in the cell_value
                    for token in tokens:
                        start = 0
                        while start < len(cell_value):
                            token_start = cell_value.find(token, start)
                            if token_start == -1:
                                break
                            positions.append((token_start, token))
                            start = token_start + len(token)

                    # Sort positions by the starting index
                    positions.sort()

                    # Construct the rich_texts list based on the sorted positions
                    start = 0
                    for token_start, token in positions:
                        if token_start > start:
                            rich_texts.append(cell_value[start:token_start])
                        rich_texts.extend([red, token])
                        start = token_start + len(token)
                    if start < len(cell_value) and start > 0:
                        rich_texts.append(cell_value[start:])

                    # Write the highlighted description
                    if rich_texts:
                        worksheet.write_rich_string(row_idx, col_idx, *rich_texts)
                    else:
                        worksheet.write(row_idx, col_idx, cell_value, cell_format_wrap)
                elif col_name == gensig_colname:
                    cell_val = []
                    # Write the 'gen_sig_gpt' column as a rich string (black color)
                    if cell_value:
                        cell_val.append(' ')
                        cell_val.extend([black, cell_value])
                        worksheet.write_rich_string(row_idx, col_idx, *cell_val)
                    else:
                        worksheet.write(row_idx, col_idx, cell_value, cell_format_wrap)
                else:
                    worksheet.write(row_idx, col_idx, cell_value, cell_format_wrap)

            row_idx += 1
#------------------------Post processing---------------------------------------
#------------------------evaluation--------------------------------------------

def add_no_match_column(df_result):
    """
    Adds a column 'no_match' with 1 if match_type is 'no_match', 0 otherwise.
    """
    df_result['no_match'] = df_result.apply(
        lambda row: 1 if row['match_type'] == 'no_match' else 0, axis=1
    )
    return df_result

def add_title_exact_column(df_result):
    """
    Adds a column 'title_exact' with 1 if match_column appears in title with exact match_type, 0 otherwise.
    """
    df_result['title_exact'] = df_result.apply(
        lambda row: 1 if (row['match_column']=='title') and (row['match_type'] == 'exact') else 0, axis=1
    )
    return df_result


def add_repeated_words_column(df_result):
    """
    Adds a column 'repeated_words' with 1 if gen_sig_gpt contains redundant repeated words
    that reduce meaning, and 0 otherwise. Accepts meaningful repeated phrases.
    """
    def has_repeated_meaningful_words(text):
        # Define explicitly acceptable patterns (return 0)
        acceptable_patterns = [
            r'\bno\s+\w+\s+no\s+\w+\b',  # Examples: "no fade no fingerprints", "no screws no nails"
            r'\brelieve\s+\w+\s+relieve\s+\w+\b',  # Examples: "relieve stress relieve boring"
            r'\b\w+\s+maker\s+\w+\s+maker\b',  # Examples: "waffle maker bread maker"
            r'\bzero\s+\w+\s+zero\s+\w+\b',  # Examples: "zero calories zero sugar"
            r'\bframed\s+\w+\s+modern\s+\w+\b',  # Examples: "framed art modern wall art"
        ]

        # Define explicitly unacceptable patterns (return 1)
        unacceptable_patterns = [
            r'\b(?:very|extra)\s+\w+\s+(?:very|extra)\s+\w+\b',  # Examples: "very fine extra fine"
            r'\bthe\s+\w+\s+the\s+\w+\b',  # Examples: "the longer the better"
        ]

        # Exclude numbers, units, and connectors
        excluded_tokens = re.compile(r'^\d+(\.\d+)?$|^(mm|cm|in|ft|x|inches|anti|in.)$', re.IGNORECASE)

        # Tokenize text and filter out excluded tokens
        tokens = [token.lower() for token in text.split() if not excluded_tokens.match(token)]

        # Check explicitly acceptable patterns
        for pattern in acceptable_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 0  # Acceptable repeated phrase

        # Check explicitly unacceptable patterns
        for pattern in unacceptable_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 1  # Redundant repeated phrase

        # Generic logic for repeated words
        unique_tokens = set(tokens)
        if len(unique_tokens) < len(tokens):  # Repeated words detected
            repeated_words = [token for token in tokens if tokens.count(token) > 1]
            if len(repeated_words) > 2:  # More than 2 repeated words = redundant
                return 1

        return 0  # Default: Not redundant

    # Apply the function to the DataFrame
    df_result['repeated_words'] = df_result['gen_sig_gpt'].apply(has_repeated_meaningful_words)
    return df_result

def add_forbidden_words_column(df_result, forbidden_words):
    """
    Adds a column 'forbidden_words' with 1 if gen_sig_gpt contains a forbidden word, 0 otherwise.
    """
    # Create a regex pattern for matching forbidden words
    forbidden_pattern = r'\b(?:' + '|'.join(map(re.escape, forbidden_words)) + r')\b'

    df_result['forbidden_words'] = df_result.apply(
        lambda row: 1 if re.search(forbidden_pattern, row['gen_sig_gpt'], re.IGNORECASE) else 0, axis=1
    )
    return df_result

def add_has_number_column(df_result):
    """
    Adds a column 'has_number' with 1 if gen_sig_gpt has at least one number token, 0 otherwise.
    """
    df_result['has_number'] = df_result.apply(
        lambda row: 1 if any(token.isdigit() for token in row['gen_sig_gpt'].split()) else 0, axis=1
    )
    return df_result

def add_numerically_dominant_column(df_result):
    """
    Adds a column 'numerically_dominant' with 1 if gen_sig_gpt is predominantly numeric,
    while excluding valid descriptive patterns such as "age 12 18 months" or "lifespan 3 5 years."
    """
    def count_numbers_with_rules(text):
        # Normalize text (remove extra spaces and handle "10 000" as one number)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\b(\d+)\s0+\b', lambda m: m.group(1), text)  # Handle numbers with extra zeros as a single number

        # Descriptive patterns to exclude
        num_x_num_unit_pattern = r'(\d+\s*x\s*\d+\s*(inches|in|cm|mm|ft))|(\d+x\d+\s*(inches|in|cm|mm|ft))'
        num_x_num_word_pattern = r'\d+\s*x\s*\d+\s+\w{3,}|\d+x\d+\s+\w{3,}'
        num_with_context_pattern = r'\b\d+\s+(hours|lifespan|life|capacity|gal|years|months|btus)\b'
        num_word_num_word_pattern = r'^\d+\s+[a-zA-Z]{3,}\s+\d+\s+[a-zA-Z]{3,}$'
        num_in_num_word_pattern = r'\b\d+\s+in\s+\d+\s+\w{3,}\b'
        age_or_time_range_pattern = r'\b(age|lifespan)\s+\d+\s+\d+\s+(years|months)\b'
        multi_number_with_context_pattern = r'\b\d+\s+\d+(\s+\d+)?\s+(servings|hrs|btus)\b'
        context_with_two_numbers_pattern = r'\b\w+\s+\d+\s+\w{3,}\s+\d+(\.\d+)?\b'

        # Handle exclusions
        if re.search(num_x_num_unit_pattern, text):
            return 0
        if re.search(num_x_num_word_pattern, text):
            return 0
        if re.search(num_with_context_pattern, text):
            return 0
        if re.search(num_word_num_word_pattern, text):
            return 0
        if re.search(num_in_num_word_pattern, text):
            return 0
        if re.search(age_or_time_range_pattern, text):
            return 0
        if re.search(multi_number_with_context_pattern, text):
            return 0
        if re.search(context_with_two_numbers_pattern, text):
            return 0

        # Count numeric and word tokens
        tokens = text.split()
        numeric_pattern = r'^\d+(\.\d+)?(mm|cm|w|ft|in|lbs|kg|m|g|oz|gal|hours|life|years|months|btus|watts)?$'
        num_count = sum(bool(re.match(numeric_pattern, token)) for token in tokens)
        word_count = len(tokens) - num_count

        # Numerical dominance check
        if len(tokens) <= 2:
            return 0  # Not numerically dominant if there are <= 2 tokens
        return 1 if (num_count >= len(tokens) / 2) or (num_count > 2) else 0

    # Apply the function
    df_result['numerically_dominant'] = df_result['gen_sig_gpt'].apply(count_numbers_with_rules)
    return df_result

def add_exclude_numbers_by_patterns_column(df_result):
    """
    Adds a column 'numbers_to_exclude_by_pattern' with 1 if gen_sig_gpt contains a number in unwanted contexts,
    and ensures legitimate cases are acceptable (e.g., numeric patterns followed by meaningful words).
    """
    def add_validate_numbers_by_patterns(text):
        """
        Validates numerical patterns, ensuring legitimate cases like "life 2000 volts 130"
        are accepted while excluding unwanted patterns.
        """
        # Normalize text (merge large numbers with spaces and remove extra spaces)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\b(\d+)\s0+\b', lambda m: m.group(1), text)  # Merge "10 000" -> "10000"

        # Acceptable patterns with examples
        acceptable_patterns = [
            r'\d+\s+month\s+warranty',  # Matches "12 month warranty"
            r'\d+\s+(?:watts|oz|psa|color modes|modes|speeds|height adjustable|height|adjustable|plates|set|graded)',  # Matches "500 watts", "4 color modes"
            r'\d+%',  # Matches "10%"
            r'\d+\s*x\s*\d+\s+(?:\w+\s+){0,2}(?:photo|size|frame|poster|print|mat|bench|door)',  # Matches "4x6 photo frame", "24x18 wall print"
            r'\d+\s+(?:polyester|cotton|nylon|spandex|wool)\s+\d+\s+(?:polyester|cotton|nylon|spandex|wool)',  # Matches "60 polyester 40 cotton"
            r'\d+\s+(?:lbs|pieces|piece|capacity|compartments|watts)',  # Matches "100 lbs", "2 compartments"
            r'\d+\s+x\s*\d+\s+(?:inches|in)\s+(?:size|frame|poster|print|photo|mat|bench|door)',  # Matches "18x24 inches frame"
            r'^\d+\s+\w{3,}\s+\d+\s+\w{3,}$',  # Matches "70 faster 60 lighter"
            r'\b\d+\s+\d+\s+\d+\s+\d+\s+(hrs|hours|servings|btus)\b',  # Matches "25 000 29 999 hrs"
            r'\b\d+\s+\d+\s+(hrs|hours|servings|btus)\b',  # Matches "40 000 44 999 hrs"
            r'\b(?:life)\s+\d+\s+(watts|volts)\s+\d+(\.\d+)?\b'  # Matches "life 2000 volts 130"
        ]

        # Not acceptable patterns with examples
        not_acceptable_patterns = [
            r'(\b\d+\b.*){4,}(?!.*\b(hrs|hours|btus|servings|watts|volts)\b)',  # Matches "123 456 789 101112", excludes valid contexts like "hrs"
            r'\d+/\d+\s*(?:scale)',  # Matches "10/5 scale"
            r'\d+ (?:wins|losses|off)',  # Matches "5 wins", "10 losses"
            r'\d+ savings free freight',  # Matches "500 savings free freight"
            r'\b(?!(\d{1,2}0{3,}$))\d{5,}\b(?!.*\b(?:hours|watts|capacity|volts)\b)',  # Matches "123456" but excludes "30000 hours"
            r'\d+\s+pc\.\s+\d+\s+in\.',  # Matches "4 pc. 10 in."
            r'\b(?:\w+\s+y\s+\w+\s+){1,}\w+',  # Matches overly verbose patterns
            r'\d+\s+in\.\s*x\s*\d+\s+in\.',  # Matches "10 in. x 20 in."
            r'\b(\w)\s(\1\s){1,3}\w{2,}\b',  # Matches "a a a word"
            r'\d+\s*x\s*\d+\s*x\s*\d+\s*\w+',  # Matches "10x20x30 weave"
            r'\d+\s+in\s+\w\s+\d+\s+in\s+\w',  # Matches "10 in A 20 in B"
            r'\d+\s*x\s*\d+\s+weave',  # Matches "10x20 weave"
            r'\d+\s*x\s*\d+\s*(?:in|inches|x)\b(?!.*\b(?:size|photo|frame|poster|mat|bench|door|print|warranty|set|colors|lighter|faster|stronger)\b)',  # Excludes generic dimensions like "10x20"
            r'\blot\s+\d+',  # Matches "lot 10"
            r'\d+\s*x\s*\d+(\.\d+)?\s+\d+\s+size',  # Matches "10x20 30 size"
            r'\d+\s*ft\s+x\s*\d+\s*ft',  # Matches "10 ft x 20 ft"
            r'\d+(\.\d+)?\s*x\s*\d+(\.\d+)?\s+\w+',  # Matches "10.5x20.5 weave"
            r'\d+\w+\s+\w+\s+\d+(\.\d+)?',  # Matches "10w motor 20.5"
            r'rated\s+\d+[a-zA-Z]?\s+\d+/\d+[a-zA-Z]+',  # Matches "rated 15A 120/277Vac"
            r'\d+(\.\d+)?x\d+(\.\d+)?x\d+(\.\d+)?',  # Matches "35.4x17.7x70.9"
            r'\b\w+\s+\w*\s+\d{5,}\w*\b',  # Matches "replaces ferris 5101987x2"
            r'\b\d{4,}(,\d{3})?\s+(lumens|output)\b'  # Matches "17520 lumens output"
        ]

        # Check acceptable patterns first
        for pattern in acceptable_patterns:
            if re.search(pattern, text):
                return 0  # Acceptable

        # Check not acceptable patterns
        for pattern in not_acceptable_patterns:
            if re.search(pattern, text):
                return 1  # Excluded (Not Acceptable)

        return 0  # Default return

    # Apply the exclusion logic to the DataFrame
    df_result['numbers_to_exclude_by_pattern'] = df_result['gen_sig_gpt'].apply(add_validate_numbers_by_patterns)
    return df_result
#---------------------------------------------------------------------
#--------------------calls heuristics : ------------------------------
def post_process_heuristic_df(df_result, forbidden_words, how=['no_match','title_exact','repeated_words',
                                                              'forbidden_words','numerically_dominant',
                                                              'numbers_to_exclude_by_pattern']):
    """
    Orchestrates the application of specified heuristic functions on the DataFrame.

    Parameters:
    - df_result (pd.DataFrame): The input DataFrame to process.
    - forbidden_words (list): A list of forbidden words for relevant heuristics.
    - how (list): A list of function names to apply. If None, applies all available functions.

    Returns:
    - pd.DataFrame: A deep copy of the updated DataFrame after applying the specified heuristics.
    """
    # Default list of heuristic functions
    if how is None:
        how = [
            'no_match',
            'title_exact',
            'repeated_words',
            'forbidden_words',
            'has_number',
            'numerically_dominant',
            'numbers_to_exclude_by_pattern'
        ]

    # Dictionary of available heuristic functions
    heuristics = {
        'no_match': lambda df: add_no_match_column(df),
        'title_exact': lambda df: add_title_exact_column(df),
        'repeated_words': lambda df: add_repeated_words_column(df),
        'forbidden_words': lambda df: add_forbidden_words_column(df, forbidden_words),
        'has_number': lambda df: add_has_number_column(df),
        'numerically_dominant': lambda df: add_numerically_dominant_column(df),
        'numbers_to_exclude_by_pattern': lambda df: add_exclude_numbers_by_patterns_column(df)
        
    }

    # Work with a deep copy of the DataFrame
    df_copy = df_result.copy(deep=True)

    # Apply specified heuristics dynamically
    applied_columns = []  # Keep track of applied heuristic column names
    for heuristic in how:
        if heuristic in heuristics:
            df_copy = heuristics[heuristic](df_copy)
            column_name = heuristic.replace('add_', '').replace('_column', '')
            applied_columns.append(column_name)
        else:
            raise ValueError(f"Function '{heuristic}' is not available in the heuristics.")

    # Create exclusion sums based on applied heuristic columns
    if applied_columns:
        valid_columns = [col for col in applied_columns if col in df_copy.columns]
        df_copy['sum_exclusions_by_how'] = df_copy[valid_columns].sum(axis=1)
    
    # Example strict sum and rule-based sum (if needed)
    strict_columns = ['no_match', 'title_exact', 'repeated_words', 'forbidden_words', 
                      'has_number', 'numerically_dominant', 'numbers_to_exclude_by_pattern']
    if set(strict_columns).issubset(df_copy.columns):
        df_copy['sum_exclusions_strict'] = df_copy[strict_columns].sum(axis=1)
    
    rules_columns = ['no_match', 'title_exact', 'repeated_words', 'forbidden_words', 
                     'numerically_dominant', 'numbers_to_exclude_by_pattern']
    if set(rules_columns).issubset(df_copy.columns):
        df_copy['sum_exclusions_rules'] = df_copy[rules_columns].sum(axis=1)
        
    df_copy.drop(['ngram_coverage', 'non_ngram_match', 'ngram_match_counts', 'all_ngrams','sum_exclusions_rules'], axis=1, inplace=True)

    return df_copy

 
def get_non_filtered_rows_by_heuristic(df_result, forbidden_words, how=['no_match', 'title_exact', 'repeated_words', 
                                                                       'forbidden_words', 'numerically_dominant', 
                                                                       'numbers_to_exclude_by_pattern']):
    """
    Filters the DataFrame and returns only the rows that are not flagged by the specified heuristics.
    
    Parameters:
    - df_result (pd.DataFrame): The input DataFrame to be filtered.
    - forbidden_words (list): List of forbidden words for the heuristics.
    - how (list): List of heuristic names to apply. Default includes common heuristic functions.
    
    Returns:
    - pd.DataFrame: A DataFrame containing only the rows that are not flagged by the specified heuristics.
    """
    # Apply the heuristics and process the DataFrame
    df_result_post_process = post_process_heuristic_df(df_result, forbidden_words, how=how)
    
    # Filter rows where 'sum_exclusions_by_how' is 0 (not flagged by heuristics)
    non_filtered_df = df_result_post_process[df_result_post_process['sum_exclusions_by_how'] == 0].copy()
    
    return non_filtered_df


def evaluate_and_filter(df, columns_to_clean, original_col_mapping, forbidden_words, 
                        columns_to_select, heuristic_how=['no_match', 'title_exact', 'repeated_words', 
                                                          'forbidden_words', 'numerically_dominant', 
                                                          'numbers_to_exclude_by_pattern']):
    """
    Orchestrates the entire process: cleaning, coverage, heuristic evaluation, and filtering.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame to process.
    - columns_to_clean (list): List of text columns to clean.
    - original_col_mapping (dict): Mapping of original column names to cleaned column names.
    - forbidden_words (list): List of forbidden words for heuristic evaluation.
    - columns_to_select (list): List of columns to select and reorder after cleaning.
    - heuristic_how (list): List of heuristic names to apply. Default includes common heuristic functions.
    
    Returns:
    - pd.DataFrame: Cleaned and filtered DataFrame with rows that are not flagged by heuristics.
    - pd.DataFrame: Excluded DataFrame with rows flagged by heuristics (`value == 1`).
    """
    # Step 1: Clean text columns
    print("Cleaning text columns...")
    df_cleaned = clean_text_columns(df, columns=columns_to_clean, original_col_mapping=original_col_mapping)
    
    # Step 2: Subset and reorder columns
    print("Reordering and selecting subset of columns...")
    df_subset_cleaned = reorder_subset_columns(df_cleaned, columns_to_select)
    
    # Step 3: Process matches for coverage and analysis
    print("Processing matches for coverage...")
    df_match = process_matches(df_subset_cleaned)
    
    # Step 4: Apply heuristics
    print("Applying heuristics...")
    df_post_heuristics = post_process_heuristic_df(df_match, forbidden_words, how=heuristic_how)
    
    
    # Step 5: Filter non-flagged and flagged rows
    print("Separating non-flagged and flagged rows...")
    filtered_df = df_post_heuristics[df_post_heuristics['sum_exclusions_by_how'] == 0].copy()
    excluded_df = df_post_heuristics[df_post_heuristics['sum_exclusions_by_how'] > 0].copy()
    all_data_heuristic_df=df_post_heuristics.copy()
    
    return filtered_df, excluded_df, df_post_heuristics


def calculate_total_exclusions_and_sums(df_result, heuristic_how=['no_match', 'title_exact', 'repeated_words',
                                                                  'forbidden_words', 'numerically_dominant',
                                                                  'numbers_to_exclude_by_pattern']):
    """
    Calculates total exclusion counts and sums for specified heuristics and columns,
    ensuring `sum_exclusions_by_how` counts each row only once (if value >= 1).

    Parameters:
    - df_result (pd.DataFrame): The input DataFrame containing heuristic columns.
    - heuristic_how (list): List of heuristic columns to evaluate. Defaults to commonly used heuristics.

    Returns:
    - pd.DataFrame: A summary DataFrame with each heuristic/column, its total exclusion count,
                    and its percentage of `sum_exclusions_by_how`.
    """
    # Exclusion sums for heuristics
    heuristic_sums = {col: df_result[col].sum() for col in heuristic_how if col in df_result.columns}

    # Count rows with at least one exclusion (sum_exclusions_by_how >= 1)
    heuristic_sums['sum_exclusions_by_how'] = (
                df_result['sum_exclusions_by_how'] >= 1).sum() if 'sum_exclusions_by_how' in df_result.columns else 0

    # Calculate percentages
    total_exclusions = heuristic_sums['sum_exclusions_by_how']
    heuristic_percentages = {col: (round(total / total_exclusions * 100, 1) if total_exclusions > 0 else 0)
                             for col, total in heuristic_sums.items()}

    # Combine results into a DataFrame
    summary_df = pd.DataFrame({
        'Column': list(heuristic_sums.keys()),
        'Total': list(heuristic_sums.values()),
        'Percentage': list(heuristic_percentages.values())
    })

    # Sort by total in descending order
    summary_df = summary_df.sort_values(by='Total', ascending=False).reset_index(drop=True)

    return summary_df


def create_sampled_data(df, output_path, columns_to_use, sample_size=300, num_annotators_per_signal=1,
                        annotators_names=None):
    """
    Create a sampled dataset from the given DataFrame, reorder columns, and add new empty columns.
    Optionally, assign annotators to the rows.

    Parameters:
    - df (pd.DataFrame): The DataFrame to sample from.
    - output_path (str): The path to save the sampled data as a parquet file.
    - columns_to_use (list): List of columns to include in the sampled dataset.
    - sample_size (int): Number of rows to sample (default: 300).
    - num_annotators_per_signal (int): Number of annotators per signal (default: 1).
    - annotators_names (list or None): List of annotator names. If None, no annotator columns are added.

    Returns:
    - pd.DataFrame: The sampled DataFrame with reordered and additional columns.
    """
    import math

    # New columns to add
    new_cols = ['label', 'reason', 'comment']
    if num_annotators_per_signal == 2:
        new_cols += ['label2', 'reason2', 'comment2']

    # Check if required columns exist in the input DataFrame
    for col in columns_to_use:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing in the input DataFrame.")

    # Reorder and subset the DataFrame
    df_subset = df[columns_to_use]

    # Add new empty columns
    for col in new_cols:
        df_subset[col] = ''

    # Add annotator columns if provided
    if annotators_names:
        df_subset['annotator_name'] = ''  # Placeholder for annotator names
        if num_annotators_per_signal == 2:
            df_subset['annotator_name2'] = ''

    # Define the final column order
    final_columns = ['item_id', 'gen_sig_gpt_orig', 'gen_sig_gpt'] + \
                    (['annotator_name'] if annotators_names else []) + \
                    new_cols + \
                    (['annotator_name2'] if num_annotators_per_signal == 2 and annotators_names else []) + \
                    ['title', 'aspects', 'desc', 'match_type', 'match_column']

    # Ensure all final columns exist and reorder
    for col in final_columns:
        if col not in df_subset.columns:
            raise ValueError(f"Column '{col}' is missing from the reordered DataFrame.")

    df_reordered = df_subset[final_columns]

    # Sample the data
    df_sampled = df_reordered.sample(n=min(sample_size, len(df)), random_state=42)

    # Assign annotator names if provided
    if annotators_names:
        num_annotators = len(annotators_names)
        rows_per_annotator = math.ceil(sample_size / num_annotators)

        # Create the annotator_name column
        annotator_column = []
        for name in annotators_names:
            annotator_column.extend([name] * rows_per_annotator)
        annotator_column = annotator_column[:len(df_sampled)]  # Adjust length to match the sampled DataFrame
        df_sampled['annotator_name'] = annotator_column

        # Create the annotator_name2 column if required
        if num_annotators_per_signal == 2:
            annotator_column2 = annotator_column.copy()
            annotator_column2.reverse()  # Distribute the second annotators differently
            annotator_column2 = annotator_column2[:len(df_sampled)]
            df_sampled['annotator_name2'] = annotator_column2

    # Save the sampled data to the specified output path
    df_sampled.to_parquet(output_path, index=False)

    print(f"Sampled data with {len(df_sampled)} rows saved to {output_path}")
    return df_sampled


def create_sampled_data_pairs(df, output_path, columns_to_use, sample_size_per_pair, annotators_names):
    """
    Samples `sample_size_per_pair` rows per pair of annotators, ensuring that each pair receives identical samples.

    Args:
        df (pd.DataFrame): The input dataframe to sample from.
        output_path (str): Path to save the sampled dataframe.
        columns_to_use (list): List of columns to retain in the output.
        sample_size_per_pair (int): Number of records to sample per annotator pair.
        annotators_names (list): List of annotators.

    Returns:
        pd.DataFrame: The sampled dataset including annotators.
    """

    num_annotators = len(annotators_names)

    if num_annotators % 2 != 0:
        raise ValueError("Number of annotators must be even for pairing.")

    total_unique_samples = sample_size_per_pair * (num_annotators // 2)  # Total unique samples

    # Randomly sample unique rows
    sampled_df = df[columns_to_use].sample(n=total_unique_samples, random_state=42).reset_index(drop=True)

    # Create duplicate samples for each pair
    repeated_samples = pd.concat([sampled_df] * 2, ignore_index=True)  # Each sample appears twice

    # Create annotator pairs
    annotator_pairs = [(annotators_names[i], annotators_names[i + 1]) for i in range(0, num_annotators, 2)]

    # Assign annotators to the sampled rows
    annotator_column = []
    for annotator_1, annotator_2 in annotator_pairs:
        annotator_column.extend([annotator_1] * sample_size_per_pair)
        annotator_column.extend([annotator_2] * sample_size_per_pair)

    repeated_samples["annotator_name"] = annotator_column

    # Add empty columns for annotation
    repeated_samples["label"] = ''
    repeated_samples["reason"] = ''
    repeated_samples["comment"] = ''

    # Ensure correct column order
    final_columns = [
        'item_id', 'gen_sig_gpt_orig', 'gen_sig_gpt', 'annotator_name',
        'label', 'reason', 'comment', 'title', 'aspects', 'desc', 'match_type', 'match_column'
    ]

    # Reorder the dataframe
    repeated_samples = repeated_samples[final_columns]

    # Save to parquet
    repeated_samples.to_parquet(output_path, index=False)

    print(f"Sampled data with {len(repeated_samples)} rows saved to {output_path}")
    print(annotator_pairs)

    return repeated_samples


def create_sampled_data_pairs_multi(df, output_path, columns_to_use, num_unique_items_per_pair, annotators_names):
    """
    Samples unique `item_id`s for each pair of annotators, ensuring that:
    - Each pair receives a distinct set of `item_id`s (no overlap between pairs).
    - Both annotators in the pair receive the same sample of item rows.

    Args:
        df (pd.DataFrame): The input dataframe containing duplicate item_ids.
        output_path (str): Path to save the sampled dataframe.
        columns_to_use (list): List of columns to retain in the output.
        num_unique_items_per_pair (int): Number of unique item_ids per annotator pair.
        annotators_names (list): Ordered list of annotators (pairs assigned in order).

    Returns:
        pd.DataFrame: The sampled dataset including annotators.
    """

    num_annotators = len(annotators_names)

    if num_annotators % 2 != 0:
        raise ValueError("Number of annotators must be even for pairing.")

    # Create annotator pairs deterministically
    annotator_pairs = [(annotators_names[i], annotators_names[i + 1]) for i in range(0, num_annotators, 2)]

    total_unique_items = num_unique_items_per_pair * len(annotator_pairs)

    # Step 1: Sample all unique item_ids for all pairs (global sampling, no replacement)
    unique_item_ids_available = df['item_id'].drop_duplicates()
    if len(unique_item_ids_available) < total_unique_items:
        print(
            f"⚠️ Not enough unique item_ids available. Reducing sample size from {total_unique_items} to {len(unique_item_ids_available)}.")
        total_unique_items = len(unique_item_ids_available)

    global_sampled_item_ids = unique_item_ids_available.sample(n=total_unique_items, random_state=42)

    # Step 2: Divide the sampled item_ids for each pair (unique sets per pair)
    item_ids_split = np.array_split(global_sampled_item_ids, len(annotator_pairs))

    all_pairs_samples = []

    for (annotator_1, annotator_2), pair_item_ids in zip(annotator_pairs, item_ids_split):
        # Step 3: Retrieve all rows for this pair's unique item_ids
        sampled_df = df[df['item_id'].isin(pair_item_ids)][columns_to_use]

        # Step 4: Ensure each sampled item_id has exactly 5 rows
        sampled_df = sampled_df.groupby("item_id").head(5)

        # Step 5: Duplicate for both annotators in the pair (same sample for pair)
        repeated_samples = pd.concat([sampled_df] * 2, ignore_index=True)

        # Step 6: Assign annotators
        annotator_column = [annotator_1] * len(sampled_df) + [annotator_2] * len(sampled_df)
        repeated_samples["annotator_name"] = annotator_column

        # Step 7: Add empty annotation columns
        repeated_samples["label"] = ''
        repeated_samples["reason"] = ''
        repeated_samples["comment"] = ''

        all_pairs_samples.append(repeated_samples)

    # Step 8: Combine samples from all pairs
    final_sampled_df = pd.concat(all_pairs_samples, ignore_index=True)

    # Step 9: Ensure correct column order
    final_columns = [
        'item_id', 'gen_sig_gpt_orig', 'gen_sig_gpt', 'annotator_name',
        'label', 'reason', 'comment', 'title', 'aspects', 'desc', 'match_type', 'match_column'
    ]
    final_columns = [col for col in final_columns if col in final_sampled_df.columns]
    final_sampled_df = final_sampled_df[final_columns]

    # Step 10: Save the output
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    try:
        final_sampled_df.to_parquet(output_path, index=False)
        print(f"✅ Sampled data saved as Parquet: {output_path}")
    except ImportError:
        fallback_output_path = output_path.replace(".parquet", ".csv")
        final_sampled_df.to_csv(fallback_output_path, index=False)
        print(f"⚠️ Parquet not available, saved as CSV instead: {fallback_output_path}")

    # Print summary
    print(f"✅ Sampled data with {len(final_sampled_df)} rows saved.")
    print(f"✅ Number of rows per annotator pair: {len(final_sampled_df) // len(annotator_pairs)}")
    print(f"Annotator Pairs Assigned: {annotator_pairs}")

    return final_sampled_df


def add_original_columns_data(sampled_df, original_df, orig_cols):
    """
    Replace specific columns in the sampled dataset with their original values from the input DataFrame.

    Parameters:
    - sampled_df (pd.DataFrame): The Sampled DataFrame.
    - original_df (pd.DataFrame): The Original DataFrame containing full data.
    - orig_cols (list): List of column names to replace in the sampled DataFrame.

    Returns:
    - pd.DataFrame: The updated sampled DataFrame with original column values replacing those in the sampled DataFrame.
    """
    # Ensure all required columns exist in the original DataFrame
    for col in orig_cols + ['item_id']:
        if col not in original_df.columns:
            raise ValueError(f"Column '{col}' is missing in the original DataFrame.")

    # Perform the merge to include all columns
    merged_df = sampled_df.merge(
        original_df,
        left_on=['item_id', 'gen_sig_gpt_orig'],
        right_on=['item_id', 'gen_sig_gpt'],  # Merge on item_id and gen_sig_gpt
        how='left',
        suffixes=('', '_original')  # Add suffix to distinguish original values
    )

    # Replace columns in orig_cols with their corresponding '_original' values
    for col in orig_cols:
        original_col = f"{col}_original"
        if original_col in merged_df.columns:
            merged_df[col] = merged_df[original_col]

    # Drop all '_original' columns to avoid duplicates
    merged_df.drop(
        columns=[f"{col}_original" for col in orig_cols if f"{col}_original" in merged_df.columns],
        inplace=True,
        errors='ignore'
    )
    sampled_df.drop(columns=['gen_sig_gpt_orig'], inplace=True, errors='ignore')

    # Return only the columns present in sampled_df with replaced values
    return merged_df[sampled_df.columns]