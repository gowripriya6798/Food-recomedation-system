"""
=============================================================================
STEP 3A: CONTENT-BASED FILTERING
=============================================================================

How it works:
    Content-based filtering recommends items that are *similar* to items a
    user has already liked. It relies entirely on item attributes — no
    information about other users is needed.

    Pipeline:
        1. Vectorise each food item's ``content_features`` (cuisine +
           category + ingredients) using TF-IDF.
        2. Compute pairwise cosine similarity between all item vectors.
        3. For a given food, return the top-N most similar items.

Why TF-IDF + Cosine Similarity?
    - TF-IDF (Term Frequency–Inverse Document Frequency) converts text
      into numerical vectors while down-weighting common words.
    - Cosine similarity measures the angle between two vectors, making it
      independent of vector magnitude — a food with many ingredient tags
      is treated fairly against one with fewer tags.

Strengths:
    ✓ No cold-start problem for items (works even for brand-new foods).
    ✓ Transparent: you can explain *why* a food was recommended by
      showing shared features.

Limitations:
    ✗ Cannot capture user taste that goes beyond item features.
    ✗ Tends to recommend items that are too similar (low serendipity).
    ✗ Quality depends on how well features describe the item.

=============================================================================
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """
    Content-based food recommender using TF-IDF + cosine similarity.

    Attributes
    ----------
    foods_df : pd.DataFrame
        Food catalog with ``content_features`` column.
    tfidf_matrix : sparse matrix
        TF-IDF vectors for every food item.
    similarity_matrix : np.ndarray
        Pairwise cosine similarity (n_foods × n_foods).
    """

    def __init__(self, foods_df: pd.DataFrame) -> None:
        """
        Initialise the recommender and fit the TF-IDF model.

        Parameters
        ----------
        foods_df : pd.DataFrame
            Must contain columns [food_id, name, content_features].
        """
        self.foods_df = foods_df.reset_index(drop=True)

        # --- Step 1: TF-IDF Vectorisation ---
        # max_features=5000 caps vocabulary size to avoid noise from very
        # rare terms. ngram_range=(1,2) captures bigrams like
        # "olive oil" or "main course".
        self.vectoriser = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self.tfidf_matrix = self.vectoriser.fit_transform(
            self.foods_df["content_features"]
        )

        # --- Step 2: Cosine Similarity Matrix ---
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

        print(f"  ✓ Content-based model fitted: {self.tfidf_matrix.shape[0]} items, "
              f"{self.tfidf_matrix.shape[1]} TF-IDF features.")

    def get_similar_foods(
        self,
        food_name: str,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Recommend foods similar to a given food item.

        Parameters
        ----------
        food_name : str
            Name of the reference food (must exist in the catalog).
        top_n : int
            Number of recommendations to return.

        Returns
        -------
        pd.DataFrame
            Top-N similar foods with columns [food_id, name, cuisine,
            category, similarity_score].
        """
        # Look up the index of the query food
        matches = self.foods_df[self.foods_df["name"] == food_name]
        if matches.empty:
            raise ValueError(f"Food '{food_name}' not found in catalog.")

        idx = matches.index[0]
        similarity_scores = self.similarity_matrix[idx]

        # Sort by similarity (descending), skip the item itself (score = 1.0)
        similar_indices = similarity_scores.argsort()[::-1][1: top_n + 1]

        results = self.foods_df.iloc[similar_indices][
            ["food_id", "name", "cuisine", "category"]
        ].copy()
        results["similarity_score"] = similarity_scores[similar_indices]
        return results.reset_index(drop=True)

    def recommend_for_user(
        self,
        user_id: int,
        ratings_df: pd.DataFrame,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Recommend foods for a user based on their highest-rated items.

        Strategy:
            1. Find the user's top 3 highest-rated foods.
            2. For each, retrieve the top similar foods.
            3. Aggregate similarity scores, remove already-rated items,
               and return the top-N.

        Parameters
        ----------
        user_id : int
            Target user.
        ratings_df : pd.DataFrame
            Ratings data with columns [user_id, food_id, rating].
        top_n : int
            Number of recommendations.

        Returns
        -------
        pd.DataFrame
            Recommended foods with aggregated similarity scores.
        """
        user_ratings = ratings_df[ratings_df["user_id"] == user_id]
        if user_ratings.empty:
            raise ValueError(f"No ratings found for user {user_id}.")

        # Get the user's top-3 highest-rated foods
        top_rated = user_ratings.nlargest(3, "rating")
        already_rated_ids = set(user_ratings["food_id"].values)

        # Aggregate similarity scores across the user's favourite foods
        candidates: dict[int, float] = {}
        for _, row in top_rated.iterrows():
            fid = row["food_id"]
            matches = self.foods_df[self.foods_df["food_id"] == fid]
            if matches.empty:
                continue
            idx = matches.index[0]
            scores = self.similarity_matrix[idx]

            for j, score in enumerate(scores):
                cand_id = int(self.foods_df.iloc[j]["food_id"])
                if cand_id not in already_rated_ids:
                    candidates[cand_id] = candidates.get(cand_id, 0.0) + score

        # Sort and return top-N
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_n]
        result_ids = [c[0] for c in sorted_candidates]
        result_scores = [c[1] for c in sorted_candidates]

        results = self.foods_df[self.foods_df["food_id"].isin(result_ids)][
            ["food_id", "name", "cuisine", "category"]
        ].copy()
        score_map = dict(zip(result_ids, result_scores))
        results["aggregated_score"] = results["food_id"].map(score_map)
        results = results.sort_values("aggregated_score", ascending=False).reset_index(drop=True)
        return results
