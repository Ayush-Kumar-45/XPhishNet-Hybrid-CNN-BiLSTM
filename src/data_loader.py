"""
Data loading utilities.
"""
import os
import glob
import pandas as pd

def detect_dataset_type(df):
    """Heuristic to detect URL vs Email dataset."""
    cols = df.columns.str.lower()
    if any('url' in c for c in cols) or 'domain' in cols:
        return 'url'
    elif any('email' in c for c in cols) or 'message' in cols or 'body' in cols:
        return 'email'
    return 'unknown'

def standardize_columns(df, content_candidates, label_candidates):
    """Find and rename content/label columns."""
    content_col, label_col = None, None
    for col in df.columns:
        if col.lower() in content_candidates or col.lower() in ('url', 'text'):
            content_col = col
        if col.lower() in label_candidates or col.lower() in ('target', 'label'):
            label_col = col
    if content_col is None:
        content_col = df.columns[0]
    if label_col is None:
        for col in df.columns:
            if df[col].nunique() <= 2:
                label_col = col
                break
        if label_col is None:
            label_col = df.columns[1] if len(df.columns) > 1 else None
    return content_col, label_col

def load_datasets(input_dir):
    """Load and combine all CSV files from input_dir."""
    csv_files = glob.glob(os.path.join(input_dir, '**', '*.csv'), recursive=True)
    url_dfs, email_dfs = [], []
    for file in csv_files:
        try:
            df = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')
            dtype = detect_dataset_type(df)
            if dtype == 'url':
                url_dfs.append(df)
                print(f"Loaded URL: {file} (shape: {df.shape})")
            elif dtype == 'email':
                email_dfs.append(df)
                print(f"Loaded Email: {file} (shape: {df.shape})")
            else:
                # fallback: if 'url' column exists, treat as URL
                if 'url' in df.columns.str.lower():
                    url_dfs.append(df)
                    print(f"Loaded as URL (by column): {file} (shape: {df.shape})")
                else:
                    print(f"Skipped unknown: {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    url_df = pd.concat(url_dfs, ignore_index=True) if url_dfs else pd.DataFrame()
    email_df = pd.concat(email_dfs, ignore_index=True) if email_dfs else pd.DataFrame()
    return url_df, email_df

def combine_and_clean(url_df, email_df):
    """Combine and standardise datasets."""
    combined = pd.concat([url_df, email_df], ignore_index=True)
    if combined.empty:
        raise RuntimeError("No data loaded. Check input directory.")
    # Ensure 'text' and 'label' columns exist
    if 'text' not in combined.columns:
        content_col, label_col = standardize_columns(
            combined,
            ['text', 'body', 'message', 'url'],
            ['label', 'target']
        )
        combined.rename(columns={content_col: 'text', label_col: 'label'}, inplace=True)
    # Convert label to int
    if combined['label'].dtype == object:
        combined['label'] = combined['label'].map(
            lambda x: 1 if str(x).lower() in ['phishing', '1', 'malicious', 'bad'] else 0
        )
    combined['label'] = combined['label'].astype(int)
    combined = combined.dropna(subset=['text', 'label']).drop_duplicates(subset=['text'])
    return combined