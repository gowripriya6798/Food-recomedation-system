#!/usr/bin/env python3
"""
=============================================================================
FOOD RECOMMENDATION SYSTEM — END-TO-END PIPELINE
=============================================================================

This script orchestrates the complete recommendation pipeline:

    Step 1 → Generate synthetic food & ratings data
    Step 2 → Preprocess: clean missing values, engineer features, train/test split
    Step 3a → Content-Based Filtering: TF-IDF + cosine similarity
    Step 3b → Collaborative Filtering: matrix factorisation via SVD
    Step 3c → Hybrid Filtering: weighted combination of both approaches
    Step 4 → Evaluate: RMSE (rating accuracy) + Precision@K (relevance)

Run:
    python main.py

Author:  Food Recommendation System Project
License: MIT
=============================================================================
"""

import warnings

import pandas as pd

from data.generate_dataset import load_datasets
from preprocessing.data_preprocessing import preprocess_pipeline
from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeFilteringRecommender
from models.hybrid import HybridRecommender, PopularityRecommender
from evaluation.metrics import evaluate_collaborative_filtering

warnings.filterwarnings("ignore")

# Set pandas display options for cleaner output
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 120)


def print_banner(title: str) -> None:
    """Print a formatted section banner."""
    print("\n")
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main() -> None:
    """Run the full food recommendation pipeline."""

    print_banner("FOOD RECOMMENDATION SYSTEM")
    print("  An end-to-end project demonstrating content-based and")
    print("  collaborative filtering with evaluation metrics.\n")

    # ==================================================================
    # STEP 1: DATA GENERATION
    # ==================================================================
    print_banner("STEP 1: DATA GENERATION")

    foods_df, ratings_df = load_datasets(seed=42)

    print(f"\n  Generated {len(foods_df)} food items across "
          f"{foods_df['cuisine'].nunique()} cuisines.")
    print(f"  Generated {len(ratings_df)} ratings from "
          f"{ratings_df['user_id'].nunique()} users.")

    print("\n  📋 Sample food items:")
    print(foods_df[["food_id", "name", "cuisine", "category"]].head(10).to_string(index=False))

    print("\n  📋 Sample ratings:")
    print(ratings_df.head(10).to_string(index=False))

    # ==================================================================
    # STEP 2: PREPROCESSING
    # ==================================================================
    print_banner("STEP 2: DATA PREPROCESSING")

    foods_processed, ratings_clean, train_df, test_df = preprocess_pipeline(
        foods_df, ratings_df, test_size=0.2, seed=42
    )

    # ==================================================================
    # STEP 3A: CONTENT-BASED FILTERING
    # ==================================================================
    print_banner("STEP 3A: CONTENT-BASED FILTERING")
    print("\n  Building content-based model (TF-IDF + cosine similarity) …\n")

    cb_model = ContentBasedRecommender(foods_processed)

    # Demo: find foods similar to a specific item
    demo_food = "Butter Chicken"
    print(f"\n  🔍 Foods similar to '{demo_food}':")
    similar = cb_model.get_similar_foods(demo_food, top_n=5)
    print(similar.to_string(index=False))

    # Demo: recommend for a specific user
    demo_user = 1
    print(f"\n  🎯 Content-based recommendations for User {demo_user}:")
    cb_recs = cb_model.recommend_for_user(demo_user, ratings_clean, top_n=5)
    print(cb_recs.to_string(index=False))

    # ==================================================================
    # STEP 3B: COLLABORATIVE FILTERING
    # ==================================================================
    print_banner("STEP 3B: COLLABORATIVE FILTERING")
    print("\n  Building collaborative filtering model (SVD matrix factorisation) …\n")

    cf_model = CollaborativeFilteringRecommender(train_df, n_factors=20)

    # Demo: recommend for the same user
    print(f"\n  🎯 Collaborative filtering recommendations for User {demo_user}:")
    cf_recs = cf_model.recommend_for_user(demo_user, ratings_clean, foods_processed, top_n=5)
    print(cf_recs.to_string(index=False))

    # Demo: predict a specific rating
    demo_food_id = 3
    pred = cf_model.predict_rating(demo_user, demo_food_id)
    food_name = foods_df[foods_df["food_id"] == demo_food_id]["name"].values[0]
    print(f"\n  📊 Predicted rating for User {demo_user} → '{food_name}': {pred:.2f}")

    # ==================================================================
    # STEP 3C: HYBRID RECOMMENDATION
    # ==================================================================
    print_banner("STEP 3C: HYBRID RECOMMENDATION")
    print("\n  Building hybrid model (content + collaborative + popularity fallback) ...\n")

    popularity_model = PopularityRecommender(ratings_clean, foods_processed, min_ratings=5)
    hybrid_model = HybridRecommender(
        cb_model=cb_model,
        cf_model=cf_model,
        popularity_model=popularity_model,
        alpha=0.5,
    )

    # Demo: hybrid recommendations for the same user
    print(f"\n  🎯 Hybrid recommendations for User {demo_user}:")
    hybrid_recs = hybrid_model.recommend_for_user(
        demo_user, ratings_clean, foods_processed, top_n=5
    )
    print(hybrid_recs.to_string(index=False))

    # Demo: cold-start user (no ratings)
    cold_start_user = 999
    print(f"\n  🆕 Cold-start recommendations for new User {cold_start_user} (popularity fallback):")
    cold_recs = hybrid_model.recommend_for_user(
        cold_start_user, ratings_clean, foods_processed, top_n=5
    )
    print(cold_recs.to_string(index=False))

    # ==================================================================
    # STEP 4: EVALUATION
    # ==================================================================
    print_banner("STEP 4: EVALUATION METRICS")

    results = evaluate_collaborative_filtering(
        cf_model=cf_model,
        train_df=train_df,
        test_df=test_df,
        foods_df=foods_processed,
        top_k=10,
        relevance_threshold=3.5,
    )

    # ==================================================================
    # COMPARISON SUMMARY
    # ==================================================================
    print_banner("SUMMARY & COMPARISON")

    print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │                    METHOD COMPARISON                            │
  ├─────────────────────┬───────────────────────────────────────────┤
  │  Content-Based      │  • Uses item features (cuisine,          │
  │  Filtering          │    category, ingredients)                 │
  │                     │  • TF-IDF vectorisation + cosine sim.     │
  │                     │  • No cold-start for items                │
  │                     │  • Explainable recommendations            │
  ├─────────────────────┼───────────────────────────────────────────┤
  │  Collaborative      │  • Uses user-item interactions only       │
  │  Filtering (SVD)    │  • Matrix factorisation discovers latent  │
  │                     │    taste dimensions                       │
  │                     │  • Captures cross-item patterns           │
  │                     │  • Cold-start for new users/items         │
  └─────────────────────┴───────────────────────────────────────────┘
    """)

    print(f"  Collaborative Filtering Evaluation:")
    print(f"    • RMSE              = {results['rmse']:.4f}")
    print(f"    • Mean Precision@10 = {results['mean_precision_at_k']:.4f}")

    print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │  IMPLEMENTED FEATURES:                                          │
  │    1. Content-based filtering (TF-IDF + cosine similarity)      │
  │    2. Collaborative filtering (SVD matrix factorisation)        │
  │    3. Hybrid model combining both approaches                    │
  │    4. Popularity-based cold-start fallback                      │
  │    5. Interactive Streamlit web application                     │
  ├─────────────────────────────────────────────────────────────────┤
  │  FUTURE IMPROVEMENTS:                                           │
  │    1. Deep learning (neural collaborative filtering)            │
  │    2. A/B testing with real user feedback                       │
  │    3. Real-time recommendation serving with caching             │
  │    4. Contextual features (time, location, dietary restrictions)│
  └─────────────────────────────────────────────────────────────────┘
    """)

    print("=" * 70)
    print("  Pipeline complete! All steps executed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
