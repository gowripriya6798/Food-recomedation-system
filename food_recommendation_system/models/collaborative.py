"""
=============================================================================
STEP 3B: COLLABORATIVE FILTERING
=============================================================================

How it works:
    Collaborative filtering finds patterns in *user behaviour*. It doesn't
    need to know anything about the food items themselves — only who rated
    what and how highly.

    We implement a matrix-factorisation approach (truncated SVD):
        1. Build a user-item rating matrix (users × foods).
        2. Compute per-user mean ratings from *actual* ratings only.
        3. Fill missing entries with each user's mean (baseline).
        4. De-mean the matrix and decompose into latent factors via SVD.
        5. Reconstruct the matrix from the latent factors to predict
           ratings for unseen user-item pairs.

Why SVD (Singular Value Decomposition)?
    - SVD discovers hidden "taste dimensions" (latent factors). For
      example, one factor might capture a preference for spicy food,
      another for desserts.
    - By keeping only the top-k factors, we get a low-rank approximation
      that generalises better than the raw (sparse) matrix.
    - This is the same family of methods that powered the Netflix Prize
      winning solution.

Hyperparameters:
    - n_factors (k): Number of latent dimensions. Higher values capture
      more nuance but risk overfitting. We default to 20.

Strengths:
    ✓ Discovers non-obvious patterns (e.g., users who like sushi also
      tend to like Thai curries).
    ✓ No feature engineering required — works purely from interactions.

Limitations:
    ✗ Cold-start problem: cannot recommend for new users or new items
      with zero ratings.
    ✗ Requires sufficient rating density to learn meaningful factors.
    ✗ Less interpretable than content-based methods.

=============================================================================
"""

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds


class CollaborativeFilteringRecommender:
    """
    Collaborative filtering recommender using matrix factorisation (SVD).

    Attributes
    ----------
    user_item_matrix : pd.DataFrame
        Dense user-item rating matrix (users × foods).
    predicted_ratings : pd.DataFrame
        Reconstructed rating matrix after SVD.
    user_means : pd.Series
        Per-user mean ratings (used for de-meaning / re-meaning).
    """

    def __init__(
        self,
        ratings_df: pd.DataFrame,
        n_factors: int = 20,
    ) -> None:
        """
        Fit the SVD model on the training ratings.

        Steps:
            1. Pivot ratings into a user-item matrix.
            2. Subtract per-user means (de-mean) so that SVD focuses on
               deviations from each user's baseline.
            3. Apply truncated SVD with k factors.
            4. Reconstruct the matrix and add user means back.

        Parameters
        ----------
        ratings_df : pd.DataFrame
            Training ratings with columns [user_id, food_id, rating].
        n_factors : int
            Number of latent factors for SVD.
        """
        # --- Step 1: Build the user-item matrix ---
        self.user_item_matrix = ratings_df.pivot_table(
            index="user_id",
            columns="food_id",
            values="rating",
            aggfunc="mean",
        )

        self.user_ids = self.user_item_matrix.index.tolist()
        self.food_ids = self.user_item_matrix.columns.tolist()

        # --- Step 2: Compute per-user means from ACTUAL ratings only ---
        # Important: we compute means *before* filling NaN so that
        # unrated items don't bias the user baseline downward.
        self.user_means = self.user_item_matrix.mean(axis=1, skipna=True)

        # Fill missing entries with the user's mean rating (baseline)
        # so SVD sees a neutral value instead of a misleading zero.
        filled_matrix = self.user_item_matrix.T.fillna(self.user_means).T

        # --- Step 3: De-mean the matrix ---
        # Subtracting each user's mean ensures SVD captures *relative*
        # preferences rather than absolute rating tendencies.
        matrix_demeaned = filled_matrix.subtract(self.user_means, axis=0).values

        # --- Step 3: Truncated SVD ---
        # svds returns the top-k singular values/vectors efficiently.
        # We cap n_factors at min(matrix dimensions) - 1 for safety.
        k = min(n_factors, min(matrix_demeaned.shape) - 1)
        U, sigma, Vt = svds(matrix_demeaned.astype(float), k=k)

        # sigma is returned as a 1-D array; convert to diagonal matrix
        sigma_diag = np.diag(sigma)

        # --- Step 4: Reconstruct predicted ratings ---
        predicted = U @ sigma_diag @ Vt
        # Add user means back to get absolute predicted ratings
        predicted = predicted + self.user_means.values.reshape(-1, 1)

        self.predicted_ratings = pd.DataFrame(
            predicted,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.columns,
        )

        print(f"  ✓ Collaborative filtering model fitted: "
              f"{len(self.user_ids)} users × {len(self.food_ids)} foods, "
              f"k={k} latent factors.")

    def predict_rating(self, user_id: int, food_id: int) -> float:
        """
        Predict a single user-food rating.

        Parameters
        ----------
        user_id : int
            Target user.
        food_id : int
            Target food item.

        Returns
        -------
        float
            Predicted rating (clipped to [1, 5]).
        """
        if user_id not in self.predicted_ratings.index:
            return float(self.user_means.mean())  # fallback to global mean
        if food_id not in self.predicted_ratings.columns:
            return float(self.user_means.get(user_id, self.user_means.mean()))

        pred = self.predicted_ratings.loc[user_id, food_id]
        return float(np.clip(pred, 1.0, 5.0))

    def recommend_for_user(
        self,
        user_id: int,
        ratings_df: pd.DataFrame,
        foods_df: pd.DataFrame,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Recommend top-N foods for a user that they haven't rated yet.

        Parameters
        ----------
        user_id : int
            Target user.
        ratings_df : pd.DataFrame
            Full ratings (to exclude already-rated items).
        foods_df : pd.DataFrame
            Food catalog (to enrich results with names).
        top_n : int
            Number of recommendations.

        Returns
        -------
        pd.DataFrame
            Recommended foods with predicted ratings.
        """
        if user_id not in self.predicted_ratings.index:
            raise ValueError(f"User {user_id} not in training data.")

        # Get predicted ratings for this user
        user_preds = self.predicted_ratings.loc[user_id]

        # Exclude already-rated foods
        already_rated = set(
            ratings_df[ratings_df["user_id"] == user_id]["food_id"].values
        )
        candidates = user_preds.drop(
            labels=[fid for fid in already_rated if fid in user_preds.index],
            errors="ignore",
        )

        # Sort by predicted rating and take top-N
        top_food_ids = candidates.nlargest(top_n).index.tolist()
        top_scores = candidates.nlargest(top_n).values

        results = foods_df[foods_df["food_id"].isin(top_food_ids)][
            ["food_id", "name", "cuisine", "category"]
        ].copy()
        score_map = dict(zip(top_food_ids, top_scores))
        results["predicted_rating"] = results["food_id"].map(score_map)
        results = results.sort_values("predicted_rating", ascending=False).reset_index(drop=True)

        # Clip predictions to valid range
        results["predicted_rating"] = results["predicted_rating"].clip(1.0, 5.0).round(2)
        return results

    def get_predictions_for_test(
        self,
        test_df: pd.DataFrame,
    ) -> list[float]:
        """
        Generate predictions for all (user, food) pairs in the test set.

        Used by the evaluation module to compute RMSE.

        Parameters
        ----------
        test_df : pd.DataFrame
            Test ratings with columns [user_id, food_id, rating].

        Returns
        -------
        list[float]
            Predicted ratings in the same order as test_df rows.
        """
        predictions = []
        for _, row in test_df.iterrows():
            pred = self.predict_rating(int(row["user_id"]), int(row["food_id"]))
            predictions.append(pred)
        return predictions
