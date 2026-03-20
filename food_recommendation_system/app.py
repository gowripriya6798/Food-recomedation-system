"""
Streamlit Web Application for the Food Recommendation System.

Run with:
    cd food_recommendation_system
    streamlit run app.py
"""

import sys
import warnings

import pandas as pd
import streamlit as st

from data.generate_dataset import load_datasets
from preprocessing.data_preprocessing import preprocess_pipeline
from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeFilteringRecommender
from models.hybrid import HybridRecommender, PopularityRecommender
from evaluation.metrics import evaluate_collaborative_filtering

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Food Recommendation System",
    page_icon="🍽️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .food-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data & model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load data, train all models, and return everything needed."""
    # Suppress print output during model loading
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        foods_df, ratings_df = load_datasets(seed=42)
        foods_processed, ratings_clean, train_df, test_df = preprocess_pipeline(
            foods_df, ratings_df, test_size=0.2, seed=42
        )

        cb_model = ContentBasedRecommender(foods_processed)
        cf_model = CollaborativeFilteringRecommender(train_df, n_factors=20)
        popularity_model = PopularityRecommender(ratings_clean, foods_processed, min_ratings=5)
        hybrid_model = HybridRecommender(
            cb_model=cb_model,
            cf_model=cf_model,
            popularity_model=popularity_model,
            alpha=0.5,
        )

        eval_results = evaluate_collaborative_filtering(
            cf_model=cf_model,
            train_df=train_df,
            test_df=test_df,
            foods_df=foods_processed,
            top_k=10,
            relevance_threshold=3.5,
        )
    finally:
        sys.stdout = old_stdout

    return {
        "foods_df": foods_df,
        "foods_processed": foods_processed,
        "ratings_clean": ratings_clean,
        "train_df": train_df,
        "test_df": test_df,
        "cb_model": cb_model,
        "cf_model": cf_model,
        "hybrid_model": hybrid_model,
        "popularity_model": popularity_model,
        "eval_results": eval_results,
    }


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------
with st.spinner("Loading models..."):
    models = load_models()

foods_df = models["foods_df"]
foods_processed = models["foods_processed"]
ratings_clean = models["ratings_clean"]
cb_model = models["cb_model"]
cf_model = models["cf_model"]
hybrid_model = models["hybrid_model"]
popularity_model = models["popularity_model"]
eval_results = models["eval_results"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🍽️ Food Recommendation System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    "A hybrid recommendation engine combining content-based filtering, "
    "collaborative filtering, and popularity-based cold-start fallback"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Top metrics row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Foods", f"{len(foods_df)}")
with col2:
    st.metric("Total Users", f"{ratings_clean['user_id'].nunique()}")
with col3:
    st.metric("RMSE", f"{eval_results['rmse']:.4f}")
with col4:
    st.metric("Precision@10", f"{eval_results['mean_precision_at_k']:.4f}")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Get Recommendations",
    "🔍 Find Similar Foods",
    "📊 Model Evaluation",
    "📋 Food Catalog",
    "ℹ️ How It Works",
])

# ===================== TAB 1: Recommendations =============================
with tab1:
    st.header("Personalised Food Recommendations")

    rec_col1, rec_col2 = st.columns([1, 2])

    with rec_col1:
        method = st.selectbox(
            "Recommendation Method",
            ["Hybrid (Recommended)", "Content-Based", "Collaborative Filtering", "Popularity (Cold Start)"],
        )

        user_ids = sorted(ratings_clean["user_id"].unique())
        if method == "Popularity (Cold Start)":
            st.info("Popularity-based recommendations don't need a user — they show the most popular foods for new users.")
            selected_user = None
        else:
            selected_user = st.selectbox("Select User", user_ids)

        top_n = st.slider("Number of Recommendations", min_value=3, max_value=20, value=10)

        generate = st.button("Generate Recommendations", type="primary", use_container_width=True)

    with rec_col2:
        if generate:
            if method == "Hybrid (Recommended)":
                recs = hybrid_model.recommend_for_user(
                    selected_user, ratings_clean, foods_processed, top_n=top_n
                )
                score_col = "hybrid_score"
            elif method == "Content-Based":
                recs = cb_model.recommend_for_user(
                    selected_user, ratings_clean, top_n=top_n
                )
                score_col = "aggregated_score"
            elif method == "Collaborative Filtering":
                recs = cf_model.recommend_for_user(
                    selected_user, ratings_clean, foods_processed, top_n=top_n
                )
                score_col = "predicted_rating"
            else:
                recs = popularity_model.recommend(top_n=top_n)
                score_col = "popularity_score"

            st.subheader(f"Top {top_n} Recommendations")
            if "method" in recs.columns:
                recs = recs.drop(columns=["method"])

            for i, row in recs.iterrows():
                with st.container():
                    c1, c2, c3, c4 = st.columns([3, 2, 2, 1.5])
                    c1.markdown(f"**{row['name']}**")
                    c2.markdown(f"🌍 {row['cuisine']}")
                    c3.markdown(f"📂 {row['category']}")
                    c4.markdown(f"⭐ {row[score_col]:.2f}")
            st.divider()
            st.dataframe(recs, use_container_width=True, hide_index=True)

# ===================== TAB 2: Similar Foods ===============================
with tab2:
    st.header("Find Similar Foods")
    st.write("Select a food item to discover similar dishes based on cuisine, category, and ingredients.")

    sim_col1, sim_col2 = st.columns([1, 2])

    with sim_col1:
        food_names = sorted(foods_df["name"].tolist())
        selected_food = st.selectbox("Select a Food Item", food_names, index=food_names.index("Butter Chicken"))
        sim_top_n = st.slider("Number of Similar Items", min_value=3, max_value=15, value=5, key="sim_slider")
        find_similar = st.button("Find Similar Foods", type="primary", use_container_width=True)

    with sim_col2:
        if find_similar:
            similar = cb_model.get_similar_foods(selected_food, top_n=sim_top_n)
            st.subheader(f"Foods Similar to '{selected_food}'")

            for _, row in similar.iterrows():
                with st.container():
                    c1, c2, c3, c4 = st.columns([3, 2, 2, 1.5])
                    c1.markdown(f"**{row['name']}**")
                    c2.markdown(f"🌍 {row['cuisine']}")
                    c3.markdown(f"📂 {row['category']}")
                    c4.markdown(f"📏 {row['similarity_score']:.3f}")
            st.divider()
            st.dataframe(similar, use_container_width=True, hide_index=True)

# ===================== TAB 3: Evaluation ==================================
with tab3:
    st.header("Model Evaluation")

    eval_col1, eval_col2 = st.columns(2)

    with eval_col1:
        st.subheader("RMSE (Root Mean Squared Error)")
        st.metric("RMSE", f"{eval_results['rmse']:.4f}")
        st.write(
            f"Predictions are off by **~{eval_results['rmse']:.2f} stars** "
            f"on a 1-5 scale. Typical range for food recommenders: 0.7 - 1.2."
        )
        st.progress(max(0.0, min(1.0, 1.0 - (eval_results["rmse"] - 0.7) / 0.8)))

    with eval_col2:
        st.subheader("Precision@10")
        st.metric("Mean Precision@10", f"{eval_results['mean_precision_at_k']:.4f}")
        st.write(
            f"On average, **{eval_results['mean_precision_at_k'] * 10:.1f} out of 10** "
            f"recommended foods are relevant (rated >= 3.5)."
        )
        st.progress(min(1.0, eval_results["mean_precision_at_k"]))

    st.divider()
    st.subheader("Method Comparison")

    comparison_data = {
        "Aspect": [
            "Approach", "Key Technique", "Cold-Start Items",
            "Cold-Start Users", "Explainability", "Serendipity",
        ],
        "Content-Based": [
            "Item features (cuisine, ingredients)",
            "TF-IDF + Cosine Similarity",
            "No problem", "Needs some ratings",
            "High (shared features)", "Low",
        ],
        "Collaborative (SVD)": [
            "User-item interactions",
            "Matrix Factorisation (SVD)",
            "Problem", "Problem",
            "Low (latent factors)", "High",
        ],
        "Hybrid": [
            "Weighted combination of both",
            "alpha * CB + (1-alpha) * CF",
            "No problem", "Popularity fallback",
            "Medium", "Medium-High",
        ],
    }
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

# ===================== TAB 4: Food Catalog ================================
with tab4:
    st.header("Food Catalog")

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        cuisine_filter = st.multiselect(
            "Filter by Cuisine",
            options=sorted(foods_df["cuisine"].unique()),
            default=[],
        )
    with filter_col2:
        category_filter = st.multiselect(
            "Filter by Category",
            options=sorted(foods_df["category"].unique()),
            default=[],
        )

    filtered_df = foods_df.copy()
    if cuisine_filter:
        filtered_df = filtered_df[filtered_df["cuisine"].isin(cuisine_filter)]
    if category_filter:
        filtered_df = filtered_df[filtered_df["category"].isin(category_filter)]

    st.write(f"Showing **{len(filtered_df)}** of {len(foods_df)} food items")
    st.dataframe(
        filtered_df[["food_id", "name", "cuisine", "category", "ingredients"]],
        use_container_width=True,
        hide_index=True,
    )

    # Cuisine distribution chart
    st.subheader("Cuisine Distribution")
    cuisine_counts = foods_df["cuisine"].value_counts().reset_index()
    cuisine_counts.columns = ["Cuisine", "Count"]
    st.bar_chart(cuisine_counts, x="Cuisine", y="Count")

# ===================== TAB 5: How It Works ================================
with tab5:
    st.header("How It Works")

    st.subheader("System Architecture")
    st.markdown("""
    ```
    User Input
        |
        v
    ┌──────────────────────────────────────────────┐
    │           HYBRID RECOMMENDER                  │
    │                                               │
    │   ┌─────────────┐    ┌──────────────────┐    │
    │   │  Content-    │    │  Collaborative   │    │
    │   │  Based       │    │  Filtering       │    │
    │   │  (TF-IDF)    │    │  (SVD)           │    │
    │   └──────┬───────┘    └────────┬─────────┘    │
    │          │    Weighted Combine  │              │
    │          └────────┬────────────┘              │
    │                   │                           │
    │          ┌────────v─────────┐                 │
    │          │  Popularity      │                 │
    │          │  Fallback        │                 │
    │          │  (Cold Start)    │                 │
    │          └──────────────────┘                 │
    └──────────────────────────────────────────────┘
        |
        v
    Top-N Recommendations
    ```
    """)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Content-Based Filtering")
        st.markdown("""
        **How:** Uses item features (cuisine, category, ingredients)

        1. **TF-IDF Vectorisation** - converts food descriptions into numerical vectors
        2. **Cosine Similarity** - measures similarity between food vectors
        3. **Recommendation** - suggests foods similar to user's top-rated items

        **Strengths:** No cold-start for items, explainable results

        **Formula:** `similarity = cos(theta) = (A . B) / (||A|| * ||B||)`
        """)

    with col_b:
        st.subheader("Collaborative Filtering")
        st.markdown("""
        **How:** Uses user-item interaction patterns (ratings)

        1. **User-Item Matrix** - pivots ratings into users x foods matrix
        2. **SVD Decomposition** - discovers latent "taste dimensions"
        3. **Rating Prediction** - reconstructs matrix to predict unseen ratings

        **Strengths:** Discovers non-obvious patterns across users

        **Formula:** `R ≈ U * Σ * V^T` (low-rank approximation)
        """)

    st.subheader("Hybrid Model")
    st.markdown("""
    The hybrid model combines both approaches using a weighted formula:

    ```
    hybrid_score = alpha * content_score + (1 - alpha) * collab_score
    ```

    Where `alpha = 0.5` gives equal weight to both methods. For **cold-start users**
    (no rating history), the system automatically falls back to **popularity-based
    recommendations** using a Bayesian average score.
    """)

    st.subheader("Evaluation Metrics")
    st.markdown("""
    | Metric | What It Measures | Formula |
    |--------|-----------------|---------|
    | **RMSE** | Rating prediction accuracy | `sqrt(mean((actual - predicted)^2))` |
    | **Precision@K** | Recommendation relevance | `relevant_in_top_K / K` |
    """)
