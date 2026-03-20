# 🍽️ Food Recommendation System

A production-style **hybrid food recommendation engine** that combines content-based filtering, collaborative filtering (SVD), and popularity-based cold-start fallback — with an interactive Streamlit web app for live demos.

---

## 📌 Problem

Users are overwhelmed by too many food choices. Traditional recommendation systems either rely solely on item attributes (missing user-taste patterns) or solely on user behavior (failing for new users). A real-world system needs **both** — plus a fallback strategy for cold-start scenarios.

## 🎯 Solution

Built a **hybrid recommendation system** that intelligently combines three approaches:

| Method | Technique | Strengths |
|--------|-----------|-----------|
| **Content-Based** | TF-IDF vectorisation + cosine similarity on food attributes | No cold-start for items; explainable results |
| **Collaborative** | SVD matrix factorisation on user-item ratings | Discovers latent taste patterns across users |
| **Hybrid** | Weighted combination: `alpha * CB + (1-alpha) * CF` | Best of both worlds; tunable balance |
| **Popularity Fallback** | Bayesian average rating (IMDb-style formula) | Handles cold-start for brand-new users |

## 📊 Evaluation Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | 1.0124 | Predictions off by ~1.01 stars on a 1–5 scale |
| **Mean Precision@10** | 0.0220 | Baseline collaborative filtering relevance |

> RMSE is within the typical 0.7–1.2 range for food/movie recommenders. The hybrid model improves coverage and handles edge cases that pure collaborative filtering cannot.

## ⚙️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Core language |
| **Pandas** | Data manipulation & analysis |
| **NumPy** | Numerical computing |
| **Scikit-learn** | TF-IDF vectorisation, cosine similarity, train/test splitting |
| **SciPy** | Truncated SVD for matrix factorisation |
| **Streamlit** | Interactive web application |

## 🚀 Features

- **Personalised recommendations** — tailored to each user's rating history
- **Similar food discovery** — find dishes like your favourites using content similarity
- **Hybrid engine** — weighted blend of content + collaborative signals
- **Cold-start handling** — popularity-based fallback for new users with no history
- **Interactive web UI** — explore recommendations, compare methods, and browse the food catalog
- **Evaluation pipeline** — RMSE and Precision@K with interpretable output
- **Modular architecture** — clean separation of data, preprocessing, models, and evaluation

## 📁 Project Structure

```
food_recommendation_system/
├── data/
│   ├── __init__.py
│   └── generate_dataset.py          # Synthetic dataset generation (200 foods, 50 users)
├── preprocessing/
│   ├── __init__.py
│   └── data_preprocessing.py        # Missing value imputation, feature engineering, train/test split
├── models/
│   ├── __init__.py
│   ├── content_based.py             # Content-based filtering (TF-IDF + cosine similarity)
│   ├── collaborative.py             # Collaborative filtering (SVD matrix factorisation)
│   └── hybrid.py                    # Hybrid recommender + popularity-based cold-start fallback
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                   # RMSE & Precision@K evaluation
├── app.py                           # Streamlit web application
├── main.py                          # End-to-end CLI pipeline runner
├── requirements.txt                 # Python dependencies
└── README.md                        # Detailed technical documentation
```

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/gowripriya6798/Food-recomedation-system.git
cd Food-recomedation-system/food_recommendation_system

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Run the CLI Pipeline

```bash
cd food_recommendation_system
python main.py
```

This executes the full pipeline: data generation → preprocessing → content-based model → collaborative model → hybrid model → evaluation.

### Launch the Streamlit Web App

```bash
cd food_recommendation_system
streamlit run app.py
```

The web app provides:
- **Get Recommendations** — choose a user and method (hybrid, content-based, collaborative, or popularity)
- **Find Similar Foods** — discover dishes similar to any food item
- **Model Evaluation** — view RMSE, Precision@K, and method comparison
- **Food Catalog** — browse and filter all 200 food items
- **How It Works** — system architecture and algorithm explanations

## 🏗️ System Architecture

```
User Input
    │
    ▼
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
│          ┌────────▼─────────┐                 │
│          │  Popularity      │                 │
│          │  Fallback        │                 │
│          │  (Cold Start)    │                 │
│          └──────────────────┘                 │
└──────────────────────────────────────────────┘
    │
    ▼
Top-N Recommendations
```

## 🔬 Key Algorithms Explained

### TF-IDF (Term Frequency–Inverse Document Frequency)
Converts food descriptions into numerical vectors. Words frequent in one food but rare across all foods get higher weights — distinguishing "lemongrass" (Thai-specific) from "garlic" (universal).

### SVD (Singular Value Decomposition)
Decomposes the user-item matrix into latent factors. Each factor captures a hidden "taste dimension" — e.g., one might represent spicy food preference, another dessert affinity. Same family of methods behind the Netflix Prize solution.

### Bayesian Average (Popularity Score)
Balances average rating with number of ratings to avoid recommending niche items with few but high ratings:
```
score = (n / (n + m)) * avg_rating + (m / (n + m)) * C
```
where `m` = minimum ratings threshold and `C` = global mean rating.

## 📈 Future Improvements

1. **Neural Collaborative Filtering** — deep learning for more expressive user-item models
2. **A/B Testing Framework** — validate recommendations with real user feedback
3. **Real-Time Serving** — deploy with caching for low-latency recommendations
4. **Contextual Features** — incorporate time of day, location, and dietary restrictions
5. **Explainability Dashboard** — show users *why* each food was recommended

## 📄 License

MIT License
