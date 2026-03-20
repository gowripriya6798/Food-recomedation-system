"""
=============================================================================
STEP 4: EVALUATION METRICS
=============================================================================

Why evaluation matters:
    A recommendation system is only as good as its ability to predict user
    preferences accurately. We use two complementary metrics:

    1. RMSE (Root Mean Squared Error)
       ─────────────────────────────
       Measures *rating prediction accuracy*. Lower is better.

       Formula: RMSE = sqrt( mean( (actual - predicted)² ) )

       - Penalises large errors more than small ones (due to squaring).
       - A RMSE of 0.8 on a 1-5 scale means predictions are off by
         about 0.8 stars on average.
       - Typical values for food/movie recommenders: 0.7 – 1.2.

    2. Precision@K
       ────────────
       Measures *recommendation relevance*. Higher is better.

       Formula: Precision@K = |relevant items in top-K| / K

       - A "relevant" item is one that the user rated ≥ threshold (default 3.5).
       - Precision@5 = 0.80 means 4 out of 5 recommended foods would be
         liked by the user.
       - More actionable than RMSE for top-N recommendation scenarios.

    Using both metrics together gives a fuller picture:
        - RMSE tells us how well we predict *exact ratings*.
        - Precision@K tells us how well we identify *what users will like*.

=============================================================================
"""

import numpy as np
import pandas as pd


def compute_rmse(actual: list[float], predicted: list[float]) -> float:
    """
    Compute Root Mean Squared Error between actual and predicted ratings.

    Parameters
    ----------
    actual : list[float]
        Ground-truth ratings.
    predicted : list[float]
        Model-predicted ratings.

    Returns
    -------
    float
        RMSE value.

    Raises
    ------
    ValueError
        If input lists have different lengths or are empty.
    """
    if len(actual) != len(predicted):
        raise ValueError(
            f"Length mismatch: actual has {len(actual)} values, "
            f"predicted has {len(predicted)}."
        )
    if len(actual) == 0:
        raise ValueError("Cannot compute RMSE on empty arrays.")

    actual_arr = np.array(actual, dtype=float)
    predicted_arr = np.array(predicted, dtype=float)

    mse = np.mean((actual_arr - predicted_arr) ** 2)
    rmse = float(np.sqrt(mse))
    return rmse


def compute_precision_at_k(
    user_id: int,
    recommended_food_ids: list[int],
    test_df: pd.DataFrame,
    relevance_threshold: float = 3.5,
) -> float:
    """
    Compute Precision@K for a single user.

    A recommended item is considered *relevant* if the user's actual
    rating for that item in the test set is ≥ relevance_threshold.

    Parameters
    ----------
    user_id : int
        Target user.
    recommended_food_ids : list[int]
        Ordered list of K recommended food IDs.
    test_df : pd.DataFrame
        Test ratings with columns [user_id, food_id, rating].
    relevance_threshold : float
        Minimum rating to consider an item "relevant" (default 3.5).

    Returns
    -------
    float
        Precision@K ∈ [0, 1].
    """
    if len(recommended_food_ids) == 0:
        return 0.0

    k = len(recommended_food_ids)

    # Get user's test ratings
    user_test = test_df[test_df["user_id"] == user_id]
    relevant_foods = set(
        user_test[user_test["rating"] >= relevance_threshold]["food_id"].values
    )

    # Count how many recommended items are relevant
    hits = sum(1 for fid in recommended_food_ids if fid in relevant_foods)

    return hits / k


def compute_mean_precision_at_k(
    recommendations: dict[int, list[int]],
    test_df: pd.DataFrame,
    relevance_threshold: float = 3.5,
) -> float:
    """
    Compute mean Precision@K across all users.

    Parameters
    ----------
    recommendations : dict[int, list[int]]
        Mapping of user_id → list of recommended food IDs.
    test_df : pd.DataFrame
        Test ratings.
    relevance_threshold : float
        Minimum rating for relevance.

    Returns
    -------
    float
        Mean Precision@K across all users.
    """
    precisions = []
    for user_id, rec_ids in recommendations.items():
        p = compute_precision_at_k(user_id, rec_ids, test_df, relevance_threshold)
        precisions.append(p)

    if len(precisions) == 0:
        return 0.0

    return float(np.mean(precisions))


def evaluate_collaborative_filtering(
    cf_model: "CollaborativeFilteringRecommender",
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    foods_df: pd.DataFrame,
    top_k: int = 10,
    relevance_threshold: float = 3.5,
) -> dict[str, float]:
    """
    Run a full evaluation of the collaborative filtering model.

    Computes:
        - RMSE on the test set (rating prediction accuracy).
        - Mean Precision@K across all test users (recommendation relevance).

    For Precision@K, we exclude only items in the *training* set so that
    test-set items are eligible for recommendation and can be evaluated.

    Parameters
    ----------
    cf_model : CollaborativeFilteringRecommender
        Fitted collaborative filtering model.
    train_df : pd.DataFrame
        Training ratings (used to exclude already-seen items).
    test_df : pd.DataFrame
        Test ratings (ground truth for evaluation).
    foods_df : pd.DataFrame
        Food catalog.
    top_k : int
        K for Precision@K (default 10 for meaningful evaluation).
    relevance_threshold : float
        Minimum rating for a relevant item.

    Returns
    -------
    dict[str, float]
        Dictionary with keys "rmse" and "mean_precision_at_k".
    """
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    # --- RMSE ---
    print(f"\n[1/2] Computing RMSE on {len(test_df):,} test ratings …")
    actual_ratings = test_df["rating"].tolist()
    predicted_ratings = cf_model.get_predictions_for_test(test_df)
    rmse = compute_rmse(actual_ratings, predicted_ratings)
    print(f"  ✓ RMSE = {rmse:.4f}")
    print(f"    Interpretation: predictions are off by ~{rmse:.2f} stars on a 1-5 scale.")

    # --- Precision@K ---
    print(f"\n[2/2] Computing Mean Precision@{top_k} …")
    test_users = test_df["user_id"].unique()
    recommendations: dict[int, list[int]] = {}

    for uid in test_users:
        if uid not in cf_model.predicted_ratings.index:
            continue
        try:
            # Exclude only training-set items so test items can be recommended
            recs = cf_model.recommend_for_user(
                uid, train_df, foods_df, top_n=top_k
            )
            recommendations[uid] = recs["food_id"].tolist()
        except (ValueError, KeyError):
            continue

    mean_prec = compute_mean_precision_at_k(
        recommendations, test_df, relevance_threshold
    )
    print(f"  ✓ Mean Precision@{top_k} = {mean_prec:.4f}")
    print(f"    Interpretation: on average, {mean_prec * top_k:.1f} out of {top_k} "
          f"recommended foods are relevant (rated ≥ {relevance_threshold}).")

    results = {
        "rmse": rmse,
        "mean_precision_at_k": mean_prec,
    }

    print(f"\n{'─' * 40}")
    print(f"  RMSE              : {rmse:.4f}")
    print(f"  Mean Precision@{top_k}  : {mean_prec:.4f}")
    print(f"{'─' * 40}")

    return results
