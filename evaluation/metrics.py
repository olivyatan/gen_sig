import re
import ssl
import string
from collections import Counter

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from dev.data_extraction.extraction_funcs import forbidden_words

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('punkt_tab')


# import nltk
# nltk.download('punkt_tab')

# Initialize the stemmer
stemmer = PorterStemmer()

def normalize_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize and stem each word
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    # Join tokens back into a single string
    return ' '.join(stemmed_tokens), stemmed_tokens

def check_substring(T, s):
    # Normalize both texts
    normalized_T = normalize_text(T)
    normalized_s = normalize_text(s)
    # Use regex to check if normalized_s appears in normalized_T
    return re.search(r'\b' + re.escape(normalized_s) + r'\b', normalized_T)


def calc_forbid_words_frequencies(sigs_df, sigs_colname, forbidden_words = forbidden_words):
    # Convert the 'gen_sig_mistral' column to lowercase
    sigs_lowered_colname = f'{sigs_colname}_lower'
    sigs_df[sigs_lowered_colname] = sigs_df[sigs_colname].str.lower()

    # Initialize a dictionary to store counts
    word_counts = {word: 0 for word in forbidden_words}

    # Count occurrences of each forbidden word
    for word in forbidden_words:
        word_counts[word] = sigs_df[sigs_lowered_colname].str.contains(word).sum()

    # Create a DataFrame from the dictionary
    result_df = pd.DataFrame(list(word_counts.items()), columns=['Forbidden Word', 'Count']).sort_values('Count',
                                                                                                         ascending=False)

    # Display the result
    return(result_df)


def calc_words_frequency(df, colname, density_as_perc=True, stem_normalized=False):

    # Convert the column to lowercase
    processed_colname = f'{colname}_lower' + ('' if not stem_normalized else '_stemmed')
    df[processed_colname] = df[colname].str.lower()
    # If stem_normalized is True, normalize the text
    if stem_normalized:
        df[processed_colname] = df[processed_colname].apply(lambda x: normalize_text(x)[0])

    # Tokenize the text and count word occurrences
    word_counts = Counter()
    df[processed_colname].apply(lambda x: word_counts.update(list(set(re.findall(r'\b\w+\b', x)))))

    # Create a DataFrame from the word counts
    word_distribution_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Count']).sort_values(by='Count',
                                                                                                    ascending=False)

    # Normalize the counts to get word densities
    word_distribution_df['DensityTokens'] = word_distribution_df['Count'] / word_distribution_df['Count'].sum()
    if density_as_perc:
        word_distribution_df['DensityTokens(%)'] = word_distribution_df['DensityTokens'].apply(lambda x : int(x*10000)/100)
    word_distribution_df['DensityRows'] = word_distribution_df['Count'] / len(df)
    if density_as_perc:
        word_distribution_df['DensityRows(%)'] = word_distribution_df['DensityRows'].apply(lambda x : int(x*10000)/100)

    return word_distribution_df, processed_colname

def create_xsls_for_curators(gensigs_df, output_excelfile_path='highlighted_text.xlsx'):
    # Add a column 'gen_sig_gpt_lower' - the lowercased version of 'gen_sig_gpt'
    gensigs_df['gen_sig_gpt'] = gensigs_df['gen_sig_gpt'].str.lower()
    gensigs_df['desc'] = gensigs_df['desc'].str.lower()
    gensigs_df['title'] = gensigs_df['title'].str.lower()
    gensigs_df['aspects'] = gensigs_df['aspects'].str.lower()

    word_distribution_df, normalized_colname = calc_words_frequency(gensigs_df, 'gen_sig_gpt', stem_normalized=True)
    # Add 3 columns - 'Token1', 'Token2', 'Token3' - the most frequent words in 'gen_sig_gpt_lower', ordered in descending
    # order of frequency, where frequency is the 'Density' column in word_distribution_df.

    # Per each row, apply the following logic:
    # Split the 'gen_sig_gpt_lower' column into tokens
    # Sort the tokens by their frequency in descending order. The frequency is based on word_distribution_df
    # Take the first 3 tokens and assign them to the 'Token1', 'Token2', 'Token3' columns
    # If there are less than 3 tokens, assign None to the remaining columns

    gensigs_df['Tokens'] = gensigs_df[normalized_colname].apply(lambda x: \
                            sorted(re.findall(r'\b\w+\b', x),
                            key=lambda y: -word_distribution_df.loc[word_distribution_df['Word'] == y, 'Count'].values[0]))
    gensigs_df['Token1'] = gensigs_df['Tokens'].apply(lambda x: x[0] if len(x) > 0 else None)
    gensigs_df['Token2'] = gensigs_df['Tokens'].apply(lambda x: x[1] if len(x) > 1 else None)
    gensigs_df['Token3'] = gensigs_df['Tokens'].apply(lambda x: x[2] if len(x) > 2 else None)

    cols_to_keep = ['item_id', normalized_colname, 'Token1', 'Token2', 'Token3', 'gen_sig_gpt', 'title', 'aspects', 'desc']
    df_rel_cols = gensigs_df[cols_to_keep]
    highlight_text_xsl(df_rel_cols, output_excelfile_path = output_excelfile_path)
    return df_rel_cols


import pandas as pd
import xlsxwriter
from tqdm import tqdm


def highlight_text_xsl(df, gensig_colname='gen_sig_gpt', source_colnames=['title', 'aspects', 'desc'],
                       output_excelfile_path='highlighted_text.xlsx'):
    # Ensure item_id is a decimal string
    df['item_id'] = df['item_id'].apply(lambda x: f'{x:.0f}')

    with xlsxwriter.Workbook(output_excelfile_path) as workbook:
        # create a worksheet with a title
        worksheet = workbook.add_worksheet('Highlighted')

        # Set up some formats to use
        gray = workbook.add_format({"color": "gray", "text_wrap": True})
        red = workbook.add_format({"color": "red", "text_wrap": True})
        cell_format_header = workbook.add_format({"bold": True, "text_wrap": True})
        cell_format_wrap = workbook.add_format({"text_wrap": True})

        # Calculate column widths
        col_widths = {}
        for col in df.columns:
            if col in ['desc', 'aspects']:
                col_widths[col] = 70 if col == 'desc' else 50
            else:
                avg_len = df[col].astype(str).apply(len).mean()
                std_len = df[col].astype(str).apply(len).std()
                col_widths[col] = avg_len + 2 * std_len

        # Set column widths
        for col_idx, col_name in enumerate(df.columns):
            worksheet.set_column(col_idx, col_idx, col_widths[col_name])

        # insert a header row
        for col_idx, col_name in enumerate(df.columns):
            worksheet.write(0, col_idx, col_name, cell_format_header)

        row_idx = 1

        # process and insert rows
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            tokens = row[gensig_colname].split()

            for col_idx, col_name in enumerate(df.columns):
                cell_value = row[col_name]
                if col_name in source_colnames and isinstance(cell_value, str):
                    # Handle JSON-like strings in 'aspects'
                    # if col_name == 'aspects':
                    #     try:
                    #         cell_value = ' '.join(ast.literal_eval(cell_value).values())
                    #     except (ValueError, SyntaxError):
                    #         pass

                    # create the highlighted description
                    # rich_texts = []
                    # start = 0
                    # for token in tokens:
                    #     token_start = cell_value.find(token, start)
                    #     if token_start != -1:
                    #         if token_start > start:
                    #             rich_texts.append(cell_value[start:token_start])
                    #         rich_texts.extend([red, token])
                    #         start = token_start + len(token)
                    # if start < len(cell_value) and start > 0:
                    #     rich_texts.append(cell_value[start:])

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



                    # write the highlighted description
                    if rich_texts:
                        worksheet.write_rich_string(row_idx, col_idx, *rich_texts)
                    else:
                        worksheet.write(row_idx, col_idx, cell_value, cell_format_wrap)
                elif col_name == gensig_colname:
                    cell_val = []
                    # Write the 'gen_sig_gpt' column as a rich string (red color)
                    if cell_value:
                        cell_val.append(' ')
                        cell_val.extend([red, cell_value])
                        worksheet.write_rich_string(row_idx, col_idx, *cell_val)
                    else:
                        worksheet.write(row_idx, col_idx, cell_value, cell_format_wrap)
                else:
                    worksheet.write(row_idx, col_idx, cell_value, cell_format_wrap)

            row_idx += 1

# Example usage
# T = "This is a long text, with various punctuation! And some stemming."
# s = "stem"
# print(check_substring(T, s))  # Output: True