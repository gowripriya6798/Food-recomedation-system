"""
=============================================================================
STEP 2: DATA PREPROCESSING
=============================================================================

Why preprocessing matters:
    Raw data is rarely clean. Missing ratings, inconsistent formats, and
    unstructured text all degrade model quality. This module handles:

    1. Missing-value imputation – Replace NaN ratings with the per-user
       mean so that we don't lose entire rows. Per-user mean is preferred
       over the global mean because it preserves individual rating tendencies
       (some users are generous raters, others are harsh).

    2. Feature engineering for content-based filtering – Combine textual
       attributes (cuisine, category, ingredients) into a single string
       per food item so that TF-IDF can capture all of them in one vector
       space.

    3. Train/test split – Split the ratings into training (80 %) and test
       (20 %) sets for unbiased evaluation of collaborative filtering.
       The split is stratified by user to ensure every user appears in
       both sets.

=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def handle_missing_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing ratings with each user's mean rating.

    Strategy:
        For each user, compute their average rating across all non-NaN
        entries, then fill NaN values with that average. If a user has
        *all* NaN ratings (unlikely but possible), fall back to the
        global mean.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Raw ratings with possible NaN values in the ``rating`` column.

    Returns
    -------
    pd.DataFrame
        Cleaned ratings with no NaN values.
    """
    df = ratings_df.copy()
    n_missing = df["rating"].isna().sum()

    if n_missing == 0:
        print("  ✓ No missing ratings found.")
        return df

    print(f"  → Found {n_missing} missing ratings ({n_missing / len(df):.1%} of total).")

    # Per-user mean imputation
    user_means = df.groupby("user_id")["rating"].transform("mean")
    global_mean = df["rating"].mean()

    df["rating"] = df["rating"].fillna(user_means)
    # Fallback for users with all-NaN ratings
    df["rating"] = df["rating"].fillna(global_mean)

    print(f"  ✓ Imputed missing ratings using per-user means (global mean fallback: {global_mean:.2f}).")
    return df


def create_content_features(foods_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a combined text feature for each food item.

    We concatenate ``cuisine``, ``category``, and ``ingredients`` into a
    single string called ``content_features``. This allows TF-IDF to
    capture all relevant item attributes in one pass.

    Example:
        cuisine="Italian", category="Main Course", ingredients="tomato basil mozzarella"
        → content_features = "Italian Main Course tomato basil mozzarella"

    Parameters
    ----------
    foods_df : pd.DataFrame
        Food catalog with columns [cuisine, category, ingredients].

    Returns
    -------
    pd.DataFrame
        Copy of foods_df with an added ``content_features`` column.
    """
    df = foods_df.copy()

    df["content_features"] = (
        df["cuisine"].str.lower() + " "
        + df["category"].str.lower() + " "
        + df["ingredients"].str.lower()
    )

    print(f"  ✓ Created content features for {len(df)} food items.")
    print(f"    Sample: \"{df['content_features'].iloc[0]}\"")
    return df


def split_ratings(
    ratings_df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings into training and test sets.

    The split is stratified by ``user_id`` to ensure that every user has
    ratings in both the training and test sets. This is critical for
    collaborative filtering: if a user appears only in the test set, the
    model has never seen them and cannot make predictions.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Cleaned ratings (no NaN values).
    test_size : float
        Fraction of data reserved for testing (default 20 %).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    # Filter out users with fewer than 2 ratings (can't stratify with only 1)
    user_counts = ratings_df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= 2].index
    valid_df = ratings_df[ratings_df["user_id"].isin(valid_users)]

    train_df, test_df = train_test_split(
        valid_df,
        test_size=test_size,
        random_state=seed,
        stratify=valid_df["user_id"],
    )

    print(f"  ✓ Split ratings → Train: {len(train_df):,}  |  Test: {len(test_df):,}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def preprocess_pipeline(
    foods_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the full preprocessing pipeline.

    Steps:
        1. Handle missing ratings via per-user mean imputation.
        2. Create combined content features for the food catalog.
        3. Split ratings into train / test.

    Parameters
    ----------
    foods_df : pd.DataFrame
        Raw food catalog.
    ratings_df : pd.DataFrame
        Raw user-food ratings (may contain NaN).
    test_size : float
        Fraction of ratings to hold out for testing.
    seed : int
        Random seed.

    Returns
    -------
    tuple of 4 DataFrames
        (foods_with_features, cleaned_ratings, train_ratings, test_ratings)
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    print("\n[1/3] Handling missing ratings …")
    cleaned_ratings = handle_missing_ratings(ratings_df)

    print("\n[2/3] Creating content features …")
    foods_with_features = create_content_features(foods_df)

    print("\n[3/3] Splitting into train/test …")
    train_df, test_df = split_ratings(cleaned_ratings, test_size=test_size, seed=seed)

    # Summary statistics
    n_users = cleaned_ratings["user_id"].nunique()
    n_foods = foods_df["food_id"].nunique()
    sparsity = 1 - len(cleaned_ratings) / (n_users * n_foods)
    print(f"\n  Summary: {n_users} users × {n_foods} foods | Sparsity: {sparsity:.1%}")

    return foods_with_features, cleaned_ratings, train_df, test_df
