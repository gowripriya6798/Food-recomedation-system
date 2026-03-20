"""
Streamlit Interactive Food Recommendation App

Run with:
    cd food_recommendation_system
    streamlit run app.py
"""

import sys
import os

import streamlit as st
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.generate_dataset import load_datasets
from preprocessing.data_preprocessing import preprocess_pipeline
from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeFilteringRecommender
from models.hybrid import HybridRecommender
from models.popularity import PopularityRecommender
from evaluation.metrics import evaluate_collaborative_filtering


@st.cache_resource
def load_models():
    """Load and cache all models (runs once on startup)."""
    foods_df, ratings_df = load_datasets(seed=42)
    foods_processed, ratings_clean, train_df, test_df = preprocess_pipeline(
        foods_df, ratings_df, test_size=0.2, seed=42
    )

    cb_model = ContentBasedRecommender(foods_processed)
    cf_model = CollaborativeFilteringRecommender(train_df, n_factors=20)
    hybrid_model = HybridRecommender(cb_model, cf_model, alpha=0.5)
    pop_model = PopularityRecommender(ratings_clean, foods_processed, min_ratings=5)

    results = evaluate_collaborative_filtering(
        cf_model=cf_model,
        train_df=train_df,
        test_df=test_df,
        foods_df=foods_processed,
        top_k=10,
        relevance_threshold=3.5,
    )

    return {
        "foods_df": foods_df,
        "foods_processed": foods_processed,
        "ratings_clean": ratings_clean,
        "train_df": train_df,
        "test_df": test_df,
        "cb_model": cb_model,
        "cf_model": cf_model,
        "hybrid_model": hybrid_model,
        "pop_model": pop_model,
        "eval_results": results,
    }


def main():
    st.set_page_config(
        page_title="Food Recommendation System",
        page_icon="🍽️",
        layout="wide",
    )

    st.title("🍽️ Food Recommendation System")
    st.markdown(
        "An end-to-end recommendation engine using **Content-Based**, "
        "**Collaborative**, and **Hybrid Filtering** with evaluation metrics."
    )

    # Load models
    with st.spinner("Loading models..."):
        data = load_models()

    foods_df = data["foods_df"]
    foods_processed = data["foods_processed"]
    ratings_clean = data["ratings_clean"]
    cb_model = data["cb_model"]
    cf_model = data["cf_model"]
    hybrid_model = data["hybrid_model"]
    pop_model = data["pop_model"]
    eval_results = data["eval_results"]

    # --- Sidebar ---
    st.sidebar.header("Configuration")
    user_ids = sorted(ratings_clean["user_id"].unique())
    selected_user = st.sidebar.selectbox("Select User ID", user_ids, index=0)
    top_n = st.sidebar.slider("Number of Recommendations", 3, 20, 5)
    food_names = sorted(foods_df["name"].tolist())
    selected_food = st.sidebar.selectbox("Select Food (for similarity)", food_names, index=food_names.index("Butter Chicken"))

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Similar Foods",
        "📝 Content-Based",
        "🤝 Collaborative",
        "⚡ Hybrid",
        "🔥 Popular (Cold Start)",
        "📊 Evaluation",
    ])

    # --- Tab 1: Similar Foods ---
    with tab1:
        st.subheader(f"Foods Similar to '{selected_food}'")
        st.markdown(
            "Uses **TF-IDF vectorisation** of food attributes (cuisine, category, "
            "ingredients) and **cosine similarity** to find the most similar items."
        )
        similar = cb_model.get_similar_foods(selected_food, top_n=top_n)
        st.dataframe(similar, use_container_width=True, hide_index=True)

        st.markdown("**How it works:**")
        st.markdown(
            "1. Each food's features are converted to a TF-IDF vector\n"
            "2. Cosine similarity measures the angle between vectors\n"
            "3. Higher similarity = more similar food attributes"
        )

    # --- Tab 2: Content-Based ---
    with tab2:
        st.subheader(f"Content-Based Recommendations for User {selected_user}")
        st.markdown(
            "Recommends foods similar to the user's **highest-rated items**. "
            "Uses the same TF-IDF + cosine similarity approach."
        )
        try:
            cb_recs = cb_model.recommend_for_user(selected_user, ratings_clean, top_n=top_n)
            st.dataframe(cb_recs, use_container_width=True, hide_index=True)
        except ValueError as e:
            st.warning(str(e))

        st.markdown("**Strengths:** No cold-start for items, explainable recommendations")
        st.markdown("**Limitations:** Limited to item feature similarity, low serendipity")

    # --- Tab 3: Collaborative ---
    with tab3:
        st.subheader(f"Collaborative Filtering for User {selected_user}")
        st.markdown(
            "Uses **SVD matrix factorisation** to discover latent taste dimensions "
            "from user-item interactions. Predicts ratings for unseen items."
        )
        try:
            cf_recs = cf_model.recommend_for_user(
                selected_user, ratings_clean, foods_processed, top_n=top_n
            )
            st.dataframe(cf_recs, use_container_width=True, hide_index=True)
        except ValueError as e:
            st.warning(str(e))

        st.markdown("**Strengths:** Discovers non-obvious patterns, no feature engineering needed")
        st.markdown("**Limitations:** Cold-start for new users/items")

    # --- Tab 4: Hybrid ---
    with tab4:
        st.subheader(f"Hybrid Recommendations for User {selected_user}")

        alpha = st.slider(
            "Content-Based Weight (alpha)",
            0.0, 1.0, 0.5, 0.1,
            help="Higher = more content-based, Lower = more collaborative"
        )

        st.markdown(
            f"**Weighted fusion:** {alpha:.0%} content-based + {1-alpha:.0%} collaborative"
        )

        temp_hybrid = HybridRecommender(cb_model, cf_model, alpha=alpha)
        try:
            hybrid_recs = temp_hybrid.recommend_for_user(
                selected_user, ratings_clean, foods_processed, top_n=top_n
            )
            st.dataframe(hybrid_recs, use_container_width=True, hide_index=True)
        except ValueError as e:
            st.warning(str(e))

        st.markdown(
            "**Why hybrid?** Combines the best of both approaches — "
            "content-based handles new items, collaborative captures taste patterns."
        )

    # --- Tab 5: Popularity ---
    with tab5:
        st.subheader("Most Popular Foods (Cold Start Fallback)")
        st.markdown(
            "For **new users with zero ratings**, we recommend the highest-rated "
            "foods as a sensible default. This is what Netflix, Spotify, and YouTube "
            "do for brand-new users."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Overall Top Foods**")
            pop_recs = pop_model.recommend(top_n=top_n)
            st.dataframe(pop_recs, use_container_width=True, hide_index=True)

        with col2:
            cuisines = sorted(foods_df["cuisine"].unique())
            selected_cuisine = st.selectbox("Filter by Cuisine", cuisines)
            st.markdown(f"**Top {selected_cuisine} Foods**")
            cuisine_recs = pop_model.recommend_by_cuisine(selected_cuisine, top_n=top_n)
            if cuisine_recs.empty:
                st.info(f"No popular {selected_cuisine} foods with enough ratings.")
            else:
                st.dataframe(cuisine_recs, use_container_width=True, hide_index=True)

    # --- Tab 6: Evaluation ---
    with tab6:
        st.subheader("Model Evaluation Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("RMSE", f"{eval_results['rmse']:.4f}")
            st.markdown(
                "**Root Mean Squared Error** measures rating prediction accuracy. "
                f"Predictions are off by ~{eval_results['rmse']:.2f} stars on a 1-5 scale. "
                "Typical range for food recommenders: 0.7 - 1.2."
            )

        with col2:
            st.metric("Mean Precision@10", f"{eval_results['mean_precision_at_k']:.4f}")
            st.markdown(
                "**Precision@K** measures recommendation relevance. "
                "A recommended item is relevant if the user rated it >= 3.5. "
                "Low values are expected with ~78% matrix sparsity."
            )

        st.markdown("---")
        st.subheader("Dataset Statistics")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Food Items", len(foods_df))
        col2.metric("Users", ratings_clean["user_id"].nunique())
        col3.metric("Ratings", len(ratings_clean))
        n_users = ratings_clean["user_id"].nunique()
        n_foods = foods_df["food_id"].nunique()
        sparsity = 1 - len(ratings_clean) / (n_users * n_foods)
        col4.metric("Matrix Sparsity", f"{sparsity:.1%}")

        st.markdown("---")
        st.subheader("Method Comparison")

        comparison_data = {
            "Method": [
                "Content-Based",
                "Collaborative (SVD)",
                "Hybrid",
                "Popularity Fallback"
            ],
            "Approach": [
                "TF-IDF + Cosine Similarity",
                "Matrix Factorisation (SVD)",
                "Weighted Score Fusion",
                "Average Rating Ranking"
            ],
            "Cold Start": [
                "No (for items)",
                "Yes (for users & items)",
                "Partial mitigation",
                "No (works for all)"
            ],
            "Strengths": [
                "Explainable, no item cold-start",
                "Discovers hidden patterns",
                "Best of both worlds",
                "Simple, always works"
            ],
        }
        st.table(pd.DataFrame(comparison_data))


if __name__ == "__main__":
    main()
