"""
=============================================================================
STEP 5: HYBRID RECOMMENDATION MODEL
=============================================================================

Why hybrid?
    Content-based and collaborative filtering each have blind spots:

    - Content-based can only recommend items similar to what a user
      already liked (low serendipity, limited to item features).
    - Collaborative filtering fails for new users or items with no
      ratings (cold-start problem).

    A hybrid model combines both approaches to get the best of both
    worlds. We use a **weighted score fusion** strategy:

        hybrid_score = alpha * content_score + (1 - alpha) * collab_score

    where alpha controls the balance (default 0.5 = equal weight).

    This is one of the simplest yet most effective hybridisation
    strategies, used in production systems at Netflix, Spotify, and
    Amazon.

Strengths:
    - Mitigates cold-start: content-based scores are available even for
      items with no collaborative data.
    - Better accuracy: combining diverse signals reduces prediction error.
    - Tunable: alpha can be adjusted per user (e.g., more content-based
      for new users, more collaborative for active users).

=============================================================================
"""

import pandas as pd


class HybridRecommender:
    """
    Hybrid food recommender combining content-based and collaborative filtering.

    Uses weighted score fusion to blend recommendations from both models.

    Parameters
    ----------
    content_model : ContentBasedRecommender
        Fitted content-based model.
    collab_model : CollaborativeFilteringRecommender
        Fitted collaborative filtering model.
    alpha : float
        Weight for content-based scores (0-1). Higher = more content-based.
    """

    def __init__(
        self,
        content_model: "ContentBasedRecommender",
        collab_model: "CollaborativeFilteringRecommender",
        alpha: float = 0.5,
    ) -> None:
        self.content_model = content_model
        self.collab_model = collab_model
        self.alpha = alpha

        print(f"  Hybrid model initialised (alpha={alpha:.2f}: "
              f"{alpha:.0%} content-based + {1 - alpha:.0%} collaborative).")

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
            1. Get content-based recommendations (top 2*N candidates).
            2. Get collaborative filtering recommendations (top 2*N candidates).
            3. Normalise scores to [0, 1] for fair comparison.
            4. Compute weighted hybrid score.
            5. Return top-N by hybrid score.

        Parameters
        ----------
        user_id : int
            Target user.
        ratings_df : pd.DataFrame
            Full ratings (to exclude already-rated items).
        foods_df : pd.DataFrame
            Food catalog with content features.
        top_n : int
            Number of recommendations to return.

        Returns
        -------
        pd.DataFrame
            Recommendations with content, collaborative, and hybrid scores.
        """
        candidate_pool = top_n * 3

        # --- Content-based candidates ---
        try:
            cb_recs = self.content_model.recommend_for_user(
                user_id, ratings_df, top_n=candidate_pool
            )
            cb_scores = dict(zip(cb_recs["food_id"], cb_recs["aggregated_score"]))
        except (ValueError, KeyError):
            cb_scores = {}

        # --- Collaborative filtering candidates ---
        try:
            cf_recs = self.collab_model.recommend_for_user(
                user_id, ratings_df, foods_df, top_n=candidate_pool
            )
            cf_scores = dict(zip(cf_recs["food_id"], cf_recs["predicted_rating"]))
        except (ValueError, KeyError):
            cf_scores = {}

        # --- Merge all candidates ---
        all_food_ids = set(cb_scores.keys()) | set(cf_scores.keys())

        if not all_food_ids:
            return pd.DataFrame(columns=[
                "food_id", "name", "cuisine", "category",
                "content_score", "collab_score", "hybrid_score",
            ])

        # --- Normalise scores to [0, 1] ---
        cb_vals = list(cb_scores.values()) if cb_scores else [0]
        cf_vals = list(cf_scores.values()) if cf_scores else [0]

        cb_min, cb_max = min(cb_vals), max(cb_vals)
        cf_min, cf_max = min(cf_vals), max(cf_vals)

        cb_range = cb_max - cb_min if cb_max != cb_min else 1.0
        cf_range = cf_max - cf_min if cf_max != cf_min else 1.0

        records = []
        for fid in all_food_ids:
            cb_raw = cb_scores.get(fid, 0.0)
            cf_raw = cf_scores.get(fid, cf_min)  # default to min if missing

            cb_norm = (cb_raw - cb_min) / cb_range if cb_scores else 0.0
            cf_norm = (cf_raw - cf_min) / cf_range if cf_scores else 0.0

            hybrid = self.alpha * cb_norm + (1 - self.alpha) * cf_norm

            records.append({
                "food_id": fid,
                "content_score": round(cb_norm, 4),
                "collab_score": round(cf_norm, 4),
                "hybrid_score": round(hybrid, 4),
            })

        hybrid_df = pd.DataFrame(records)
        hybrid_df = hybrid_df.sort_values("hybrid_score", ascending=False).head(top_n)

        # Enrich with food metadata
        result = hybrid_df.merge(
            foods_df[["food_id", "name", "cuisine", "category"]],
            on="food_id",
            how="left",
        )
        result = result[
            ["food_id", "name", "cuisine", "category",
             "content_score", "collab_score", "hybrid_score"]
        ].reset_index(drop=True)

        return result
