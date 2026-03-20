"""
=============================================================================
HYBRID RECOMMENDATION MODEL
=============================================================================

How it works:
    The hybrid recommender combines content-based and collaborative filtering
    to leverage the strengths of both approaches:

    - Content-based filtering excels at finding items with similar attributes
      and has no cold-start problem for items.
    - Collaborative filtering discovers latent user-taste patterns from
      interactions but struggles with new users/items.

    The hybrid model uses a weighted combination:
        hybrid_score = alpha * content_score + (1 - alpha) * collab_score

    where alpha controls the balance (default 0.5 = equal weight).

    It also includes a **popularity-based fallback** for cold-start users
    (users with no rating history), ensuring every user gets recommendations.

Strengths:
    + Better coverage than either method alone
    + Handles cold-start via popularity fallback
    + Tunable balance between content and collaborative signals

=============================================================================
"""

import numpy as np
import pandas as pd

from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeFilteringRecommender


class PopularityRecommender:
    """
    Popularity-based recommender for cold-start fallback.

    Recommends the most popular (highest average rating) foods that meet
    a minimum number of ratings to avoid recommending niche items.
    """

    def __init__(
        self,
        ratings_df: pd.DataFrame,
        foods_df: pd.DataFrame,
        min_ratings: int = 5,
    ) -> None:
        """
        Compute popularity scores for all food items.

        Parameters
        ----------
        ratings_df : pd.DataFrame
            Ratings with columns [user_id, food_id, rating].
        foods_df : pd.DataFrame
            Food catalog.
        min_ratings : int
            Minimum number of ratings for a food to be considered.
        """
        self.foods_df = foods_df

        # Compute average rating and count per food
        food_stats = ratings_df.groupby("food_id")["rating"].agg(
            ["mean", "count"]
        ).reset_index()
        food_stats.columns = ["food_id", "avg_rating", "num_ratings"]

        # Filter by minimum ratings
        popular = food_stats[food_stats["num_ratings"] >= min_ratings].copy()

        # Weighted score: balances average rating with number of ratings
        # using a Bayesian average (similar to IMDb formula)
        C = food_stats["avg_rating"].mean()  # global mean
        m = min_ratings
        popular["popularity_score"] = (
            (popular["num_ratings"] / (popular["num_ratings"] + m)) * popular["avg_rating"]
            + (m / (popular["num_ratings"] + m)) * C
        )

        self.popular_foods = popular.sort_values(
            "popularity_score", ascending=False
        ).reset_index(drop=True)

    def recommend(self, top_n: int = 10) -> pd.DataFrame:
        """
        Return the top-N most popular foods.

        Parameters
        ----------
        top_n : int
            Number of recommendations.

        Returns
        -------
        pd.DataFrame
            Popular foods with scores.
        """
        top_ids = self.popular_foods.head(top_n)["food_id"].tolist()
        results = self.foods_df[self.foods_df["food_id"].isin(top_ids)][
            ["food_id", "name", "cuisine", "category"]
        ].copy()

        score_map = dict(
            zip(
                self.popular_foods["food_id"],
                self.popular_foods["popularity_score"],
            )
        )
        results["popularity_score"] = results["food_id"].map(score_map)
        results = results.sort_values(
            "popularity_score", ascending=False
        ).reset_index(drop=True)
        return results


class HybridRecommender:
    """
    Hybrid recommender combining content-based and collaborative filtering
    with a popularity-based cold-start fallback.

    Parameters
    ----------
    cb_model : ContentBasedRecommender
        Fitted content-based model.
    cf_model : CollaborativeFilteringRecommender
        Fitted collaborative filtering model.
    popularity_model : PopularityRecommender
        Popularity-based fallback model.
    alpha : float
        Weight for content-based scores (0 to 1).
        alpha=0.5 means equal weight for both methods.
    """

    def __init__(
        self,
        cb_model: ContentBasedRecommender,
        cf_model: CollaborativeFilteringRecommender,
        popularity_model: PopularityRecommender,
        alpha: float = 0.5,
    ) -> None:
        self.cb_model = cb_model
        self.cf_model = cf_model
        self.popularity_model = popularity_model
        self.alpha = alpha

    def recommend_for_user(
        self,
        user_id: int,
        ratings_df: pd.DataFrame,
        foods_df: pd.DataFrame,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Generate hybrid recommendations for a user.

        Strategy:
            1. If the user has no ratings, fall back to popularity-based recs.
            2. Otherwise, get scores from both content-based and collaborative
               filtering, normalise them to [0, 1], and combine with weighted
               average.

        Parameters
        ----------
        user_id : int
            Target user.
        ratings_df : pd.DataFrame
            Full ratings data.
        foods_df : pd.DataFrame
            Food catalog with content_features.
        top_n : int
            Number of recommendations.

        Returns
        -------
        pd.DataFrame
            Hybrid recommendations with scores.
        """
        user_ratings = ratings_df[ratings_df["user_id"] == user_id]

        # Cold-start fallback: user has no ratings
        if user_ratings.empty:
            recs = self.popularity_model.recommend(top_n=top_n)
            recs["method"] = "popularity (cold-start)"
            return recs

        already_rated_ids = set(user_ratings["food_id"].values)

        # --- Content-based scores ---
        cb_scores = self._get_content_scores(user_id, ratings_df, already_rated_ids)

        # --- Collaborative filtering scores ---
        cf_scores = self._get_collab_scores(user_id, already_rated_ids)

        # --- Combine scores ---
        all_food_ids = set(cb_scores.keys()) | set(cf_scores.keys())
        hybrid_scores: dict[int, float] = {}

        for fid in all_food_ids:
            cb_val = cb_scores.get(fid, 0.0)
            cf_val = cf_scores.get(fid, 0.0)
            hybrid_scores[fid] = self.alpha * cb_val + (1 - self.alpha) * cf_val

        # Sort and take top-N
        sorted_items = sorted(
            hybrid_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        result_ids = [item[0] for item in sorted_items]
        result_scores = [item[1] for item in sorted_items]

        results = foods_df[foods_df["food_id"].isin(result_ids)][
            ["food_id", "name", "cuisine", "category"]
        ].copy()

        score_map = dict(zip(result_ids, result_scores))
        results["hybrid_score"] = results["food_id"].map(score_map)
        results = results.sort_values(
            "hybrid_score", ascending=False
        ).reset_index(drop=True)
        results["method"] = "hybrid"
        return results

    def _get_content_scores(
        self,
        user_id: int,
        ratings_df: pd.DataFrame,
        already_rated_ids: set[int],
    ) -> dict[int, float]:
        """Get normalised content-based scores for unrated foods."""
        user_ratings = ratings_df[ratings_df["user_id"] == user_id]
        top_rated = user_ratings.nlargest(3, "rating")

        scores: dict[int, float] = {}
        for _, row in top_rated.iterrows():
            fid = row["food_id"]
            matches = self.cb_model.foods_df[
                self.cb_model.foods_df["food_id"] == fid
            ]
            if matches.empty:
                continue
            idx = matches.index[0]
            sim_scores = self.cb_model.similarity_matrix[idx]

            for j, score in enumerate(sim_scores):
                cand_id = int(self.cb_model.foods_df.iloc[j]["food_id"])
                if cand_id not in already_rated_ids:
                    scores[cand_id] = scores.get(cand_id, 0.0) + score

        # Normalise to [0, 1]
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

        return scores

    def _get_collab_scores(
        self,
        user_id: int,
        already_rated_ids: set[int],
    ) -> dict[int, float]:
        """Get normalised collaborative filtering scores for unrated foods."""
        scores: dict[int, float] = {}

        if user_id not in self.cf_model.predicted_ratings.index:
            return scores

        user_preds = self.cf_model.predicted_ratings.loc[user_id]

        for food_id in user_preds.index:
            if food_id not in already_rated_ids:
                # Normalise predicted rating from [1, 5] to [0, 1]
                pred = float(np.clip(user_preds[food_id], 1.0, 5.0))
                scores[int(food_id)] = (pred - 1.0) / 4.0

        return scores
