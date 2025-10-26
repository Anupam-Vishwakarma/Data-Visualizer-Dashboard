# modules/utils.py
"""
Utility functions for Data Visualizer Dashboard
-----------------------------------------------
Handles dataset loading, available dataset listing, and user preferences.
"""

import os
import json
import pandas as pd
from configs.settings import DATA_FOLDER, STORAGE_FOLDER

# Path to user preferences JSON file
USER_PREFS_FILE = os.path.join(STORAGE_FOLDER, "user_prefs.json")

# ------------------ Function: Load Dataset ------------------ #
def load_data(dataset_name: str) -> pd.DataFrame:
    """
    Load a CSV dataset from the data folder.

    Parameters:
    -----------
    dataset_name : str
        Filename of the dataset (with .csv extension)

    Returns:
    --------
    pd.DataFrame
        Loaded dataset

    Raises:
    -------
    FileNotFoundError : if dataset file does not exist
    """
    file_path = os.path.join(DATA_FOLDER, dataset_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found in {DATA_FOLDER}")
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Error loading dataset '{dataset_name}': {e}")

# ------------------ Function: Get Available Datasets ------------------ #
def get_available_datasets() -> list:
    """
    List all CSV datasets available in the data folder.

    Returns:
    --------
    List[str] : List of dataset filenames
    """
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    datasets = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]
    return datasets

# ------------------ Function: Load User Preferences ------------------ #
def load_user_pref() -> dict:
    """
    Load user preferences (theme, chart colors, comments) from JSON storage.

    Returns:
    --------
    dict : User preferences
    """
    if not os.path.exists(STORAGE_FOLDER):
        os.makedirs(STORAGE_FOLDER)
    if not os.path.exists(USER_PREFS_FILE):
        return {}  # Return empty dict if file doesn't exist
    with open(USER_PREFS_FILE, "r") as f:
        prefs = json.load(f)
    return prefs

# ------------------ Function: Save User Preferences ------------------ #
def save_user_pref(prefs: dict):
    """
    Save user preferences to JSON storage.

    Parameters:
    -----------
    prefs : dict
        User preferences to save
    """
    if not os.path.exists(STORAGE_FOLDER):
        os.makedirs(STORAGE_FOLDER)
    with open(USER_PREFS_FILE, "w") as f:
        json.dump(prefs, f, indent=4)

# ------------------ Function: Update a Specific Preference ------------------ #
def update_user_pref(key: str, value):
    """
    Update a specific user preference key and save it.

    Parameters:
    -----------
    key : str
        Preference key (e.g., 'theme', 'chart_colors')
    value : any
        Value to update for the key
    """
    prefs = load_user_pref()
    prefs[key] = value
    save_user_pref(prefs)
