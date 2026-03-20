"""
=============================================================================
POPULARITY-BASED FALLBACK (COLD START SOLUTION)
=============================================================================

The cold-start problem:
    When a new user joins the platform with zero ratings, both content-based
    and collaborative filtering fail — there's nothing to base recommendations
    on.

Solution: Popularity-based recommendations
    Recommend the most popular (highest average rating) foods as a sensible
    default. This is what platforms like Netflix, Spotify, and YouTube do
    for brand-new users — they show trending / top-rated content.

    We also incorporate a minimum-ratings threshold to avoid recommending
    niche items that have a high average from only 1-2 ratings (Bayesian
    prior / smoothing logic).

When to use:
    - New users with 0 ratings (pure cold start)
    - Users with very few ratings (< 3) where personalisation is unreliable
    - As a fallback when the primary model fails

=============================================================================
"""

import pandas as pd


class PopularityRecommender:
    """
    Popularity-based recommender for cold-start scenarios.

    Ranks foods by average rating, filtered to items with a minimum
    number of ratings to ensure statistical reliability.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Ratings with columns [user_id, food_id, rating].
    foods_df : pd.DataFrame
        Food catalog.
    min_ratings : int
        Minimum number of ratings a food must have to be considered.
    """

    def __init__(
        self,
        ratings_df: pd.DataFrame,
        foods_df: pd.DataFrame,
        min_ratings: int = 5,
    ) -> None:
        self.foods_df = foods_df

        # Compute average rating and count per food
        food_stats = ratings_df.groupby("food_id")["rating"].agg(
            ["mean", "count"]
        ).reset_index()
        food_stats.columns = ["food_id", "avg_rating", "num_ratings"]

        # Filter to foods with enough ratings
        food_stats = food_stats[food_stats["num_ratings"] >= min_ratings]

        # Sort by average rating (descending), then by count (tiebreaker)
        food_stats = food_stats.sort_values(
            ["avg_rating", "num_ratings"], ascending=[False, False]
        ).reset_index(drop=True)

        # Merge with food metadata
        self.popular_foods = food_stats.merge(
            foods_df[["food_id", "name", "cuisine", "category"]],
            on="food_id",
            how="left",
        )

        print(f"  Popularity model fitted: {len(self.popular_foods)} foods "
              f"with >= {min_ratings} ratings.")

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
            Popular foods with avg_rating and num_ratings.
        """
        result = self.popular_foods.head(top_n)[
            ["food_id", "name", "cuisine", "category", "avg_rating", "num_ratings"]
        ].copy()
        result["avg_rating"] = result["avg_rating"].round(2)
        return result.reset_index(drop=True)

    def recommend_by_cuisine(self, cuisine: str, top_n: int = 5) -> pd.DataFrame:
        """
        Return the most popular foods within a specific cuisine.

        Useful for new users who indicate a cuisine preference.

        Parameters
        ----------
        cuisine : str
            Target cuisine (e.g., "Italian").
        top_n : int
            Number of recommendations.

        Returns
        -------
        pd.DataFrame
            Popular foods from the specified cuisine.
        """
        filtered = self.popular_foods[
            self.popular_foods["cuisine"].str.lower() == cuisine.lower()
        ]
        result = filtered.head(top_n)[
            ["food_id", "name", "cuisine", "category", "avg_rating", "num_ratings"]
        ].copy()
        result["avg_rating"] = result["avg_rating"].round(2)
        return result.reset_index(drop=True)
