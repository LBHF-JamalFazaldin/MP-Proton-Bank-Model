import os
import pandas as pd
import numpy as np
import inspect as insp
from IPython.display import display as original_display

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------

def clean_label(label):
    """
    Cleans a label by replacing underscores with spaces and converting to title case.
    Args:
        label (str): The label string to clean.
    Returns:
        str: The cleaned label.
    """
    try:
        return label.replace('_', ' ').title()
    except AttributeError as e:
        print(f'Error cleaning label: {e}')
        return label
 
def get_var_name(var):
    """
    Attempts to retrieve the name of a variable from the global scope.
 
    Args:
        var (object): The object to look up.
 
    Returns:
        str or None: The variable name if found, else None.
    """
    try:
        for name, value in globals().items():
            if value is var:
                return name
    except Exception as e:
        print(f'Error getting variable name: {e}')
 
 
def header_list(df):
    """
    Extracts the header (first row) of a DataFrame and returns remaining rows as a new DataFrame.
 
    Args:
        df (pd.DataFrame): The DataFrame whose header and body are to be separated.
 
    Returns:
        pd.DataFrame: New DataFrame using first row as header.
    """
    try:
        df_list_ = df.copy()
        df_list = df_list_.columns.tolist()
        df_list = pd.DataFrame(df_list)
        new_header = df_list.iloc[0]
        df_list = df_list[1:]
        df_list.columns = new_header
        df_list.reset_index(drop=True, inplace=True)
        return df_list
    except Exception as e:
        print(f'Error creating header list: {e}')
        return pd.DataFrame()
 

def read_directory(directory=False):
    """
    Lists all files in a directory. Defaults to the current working directory if none provided.
 
    Args:
        directory (str or bool): Path to the directory or False to use current working directory.
    """
    if directory == False:
        directory = os.getcwd()
    files = os.listdir(directory)
 
    if directory == os.getcwd():
        print(f"Your Current Directory is: {directory}")
    else:
        print(f"Directory being read is: {directory}")
 
    print("Files in: %s\n" % (files))

def strip_dataframe(df):
    """
    Strips leading and trailing whitespace from all string cells and column headers in a DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to clean.
    Returns:
        pd.DataFrame: A new DataFrame with whitespace stripped from strings and headers.
    """
    # Strip whitespace from column headers
    df.columns = df.columns.str.strip()
 
    # Strip whitespace from string cells
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df 
 
def display(df, max_columns=True, max_rows=False, **kwargs):
    """
    Displays a DataFrame with its name and number of records.
    Args:
        df (pd.DataFrame): DataFrame to display.
        max_columns (bool): Show all columns if True.
        max_rows (bool): Show all rows if True.
    """
    try:
        frame = insp.currentframe().f_back
        name = "Unnamed DataFrame"
        for var_name, var_value in frame.f_locals.items():
            if var_value is df:
                name = var_name
                break
 
        if name not in {'df', 'Unnamed DataFrame', 'unique_counts'}:
            print(f"DataFrame: {name}")
        if name not in {'info_df', None}:
            number_of_records = df.shape[0]
            number_of_fields = df.shape[1]
            duplicate_count = df.duplicated(keep=False).sum()
            unique_duplicate_count = duplicate_count - df.duplicated(keep='first').sum()
            print(
                f"Number of records: {number_of_records:,}",
                " | ",
                f"Number of fields: {number_of_fields:,}\n",
                f"Number of unique duplicate records: {unique_duplicate_count}"
                " | ",
                f"Total number of duplicate records: {duplicate_count}"
            )
        elif name == 'info_df':
            print(f"Max number of non-null records: {kwargs.get('max_records', "N/A")}")
        
        if max_columns:
            pd.set_option('display.max_columns', None)
        if max_rows:
            pd.set_option('display.max_rows', None)
 
        original_display(df)
        pd.reset_option('display.max_columns')
        pd.reset_option('display.max_rows')
 
    except Exception as e:
        print(f'Error displaying DataFrame: {e}')

def unique_values(df, show_df=10, sort_values=True):
    """
    Displays unique values for each column in a DataFrame, with optional sorting.
 
    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        show_df (int): Number of rows to display from the resulting summary DataFrame.
        sort_values (bool): Whether to sort the unique values.
    """
    try:
        unique_df_data = {}
        max_length = 0
 
        for col in df.columns:
            values = df[col].dropna().unique()  # drop NaNs to avoid sorting issues
            if sort_values:
                try:
                    values = sorted(values)  # will work for most types
                except Exception:
                    values = list(values)  # fallback if sorting fails
 
            max_length = max(max_length, len(values))
            unique_df_data[col] = values
 
        # Pad shorter lists with None
        for col in unique_df_data:
            padding = [None] * (max_length - len(unique_df_data[col]))
            unique_df_data[col] += padding
 
        unique_df = pd.DataFrame(unique_df_data)
        unique_df.fillna('--', inplace=True)
 
        if show_df:
            display(unique_df.head(show_df), max_rows=True)
 
        return
 
    except Exception as e:
        print(f'Error extracting unique values: {e}')
        return
 
 
def validate_data(df, show_df=10):
    """
    Validates a DataFrame by displaying basic metrics such as unique/non-null values,
    data types, duplicates, and descriptive statistics.
 
    Args:
        df (pd.DataFrame): The DataFrame to validate.
    """
    try:
        print('#' * 165, end='')
 
        # Display the dataset
        print('\nValidation Dataframe')
        display(df)

        # Display the unique values of the dataset
        print('Unique Values')
        unique_values(df, show_df=show_df)
        print()
 
        # Unique values and non-null counts
        info_df = pd.DataFrame(df.nunique())
        non_null = pd.DataFrame(len(df) - df.notnull().sum())
        dtypes = pd.DataFrame(df.dtypes, columns=['Data Type'])
 
        # Merge all metrics into one DataFrame
        info_df = pd.merge(
            info_df, non_null,
            how='left', left_index=True, right_index=True,
            suffixes=['_unique', '_null']
        )
 
        info_df = pd.merge(
            info_df, dtypes,
            how='left', left_index=True, right_index=True
        )
 
        # Format and rename
        info_df.reset_index(inplace=True)
        info_df.rename(
            columns={
                '0_unique': 'No. of Unique Values',
                '0_null': 'No. of Null Values',
                'index': 'Field Name'
            },
            inplace=True
        )

        info_df[['No. of Unique Values', 'No. of Null Values']] = info_df[
            ['No. of Unique Values', 'No. of Null Values']
        ].map(lambda x: f"{x:,}")
        
        info_df['No. of Null Values'].replace({0:"Full"})

        # Check for duplicates
        duplicate_count = df.duplicated(keep=False).sum()

        display(info_df, max_rows=True, duplicate_count=duplicate_count, max_records=len(df))
 
        # Summary statistics
        print("\nSummary statistics:")
        display(df.describe())

        print('End of data validation')
        print('#' * 165)
    except Exception as e:
        print(f'Error validating data: {e}')

def export_to_csv(df, **kwargs):
    """
    Exports a DataFrame to a CSV file.
 
    Args:
        df (pd.DataFrame): The DataFrame to export.
        **kwargs: 
            - directory (str): The target directory path.
            - df_name (str): Optional name of the DataFrame to use in filename.
    """
    try:
        directory = kwargs.get('directory', r"path/to/downloads")
        df_name = kwargs.get('df_name', get_var_name(df))
 
        # Prompt for name if not found
        if not isinstance(df_name, str) or df_name == '_':
            df_name = input('Dataframe not found in global variables. Please enter a name for the DataFrame: ')
 
        file_path = f'{directory}\\{df_name}.csv'
 
        print(f'Exporting {df_name} to CSV...\n@ {file_path}\n')
        df.to_csv(file_path, index=False)
        print(f'Successfully exported {df_name} to CSV')
    except Exception as e:
        print(f'Error exporting to CSV: {e}')