# Food Recommendation System

An end-to-end food recommendation system built in Python, demonstrating **content-based filtering**, **collaborative filtering**, and proper **evaluation metrics**.

## Project Structure

```
food_recommendation_system/
├── data/
│   ├── __init__.py
│   └── generate_dataset.py        # Synthetic dataset generation
├── preprocessing/
│   ├── __init__.py
│   └── data_preprocessing.py      # Data cleaning & feature engineering
├── models/
│   ├── __init__.py
│   ├── content_based.py           # Content-based filtering (TF-IDF + cosine similarity)
│   └── collaborative.py           # Collaborative filtering (SVD matrix factorisation)
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                 # RMSE & Precision@K evaluation
├── main.py                        # End-to-end pipeline runner
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## How It Works

### Step 1: Data Generation (`data/generate_dataset.py`)
- Generates **200 food items** across 8 world cuisines (Italian, Mexican, Indian, Chinese, Japanese, American, Thai, Mediterranean)
- Creates **~1,500 user ratings** from 50 synthetic users on a 1-5 scale
- Intentionally introduces missing values (~5%) to demonstrate preprocessing

### Step 2: Data Preprocessing (`preprocessing/data_preprocessing.py`)
- **Missing value imputation**: Replaces NaN ratings with per-user means (preserves individual rating tendencies)
- **Feature engineering**: Combines cuisine, category, and ingredients into a single text feature for TF-IDF
- **Train/test split**: 80/20 stratified split ensuring every user appears in both sets

### Step 3A: Content-Based Filtering (`models/content_based.py`)
- **TF-IDF Vectorisation**: Converts food text features into numerical vectors
- **Cosine Similarity**: Computes pairwise similarity between all food items
- **Recommendation**: Suggests foods similar to a user's highest-rated items
- **Strengths**: No cold-start problem for items, explainable recommendations
- **Limitations**: Limited to item feature similarity, low serendipity

### Step 3B: Collaborative Filtering (`models/collaborative.py`)
- **User-Item Matrix**: Pivots ratings into a users × foods matrix
- **SVD Decomposition**: Discovers latent "taste dimensions" via matrix factorisation
- **Rating Prediction**: Reconstructs the matrix to predict unseen user-item ratings
- **Strengths**: Discovers non-obvious patterns, no feature engineering needed
- **Limitations**: Cold-start problem for new users/items

### Step 4: Evaluation (`evaluation/metrics.py`)
- **RMSE (Root Mean Squared Error)**: Measures rating prediction accuracy (lower is better)
  - Formula: `RMSE = sqrt(mean((actual - predicted)²))`
  - Typical range for food recommenders: 0.7 - 1.2
- **Precision@K**: Measures recommendation relevance (higher is better)
  - Formula: `Precision@K = |relevant items in top-K| / K`
  - An item is "relevant" if the user rated it ≥ 3.5

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd food_recommendation_system

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the full pipeline
cd food_recommendation_system
python main.py
```

This will:
1. Generate synthetic food and ratings data
2. Preprocess data (clean, engineer features, split)
3. Build and demo a content-based recommender
4. Build and demo a collaborative filtering recommender
5. Evaluate the models with RMSE and Precision@K
6. Print a comparison summary

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computing |
| `pandas` | Data manipulation |
| `scikit-learn` | TF-IDF, cosine similarity, train/test split |
| `scipy` | Truncated SVD for matrix factorisation |

## Key Concepts Explained

### TF-IDF (Term Frequency–Inverse Document Frequency)
Converts text into numerical vectors. Words that appear frequently in one document but rarely across all documents get higher weights. This helps distinguish "garlic" (common across cuisines) from "lemongrass" (distinctive to Thai cuisine).

### Cosine Similarity
Measures the angle between two vectors, ranging from 0 (completely different) to 1 (identical). It's magnitude-independent, so a food with many tags is compared fairly against one with fewer tags.

### SVD (Singular Value Decomposition)
Decomposes the user-item matrix into latent factors. Each factor captures a hidden "taste dimension" — for example, one factor might represent a preference for spicy food, another for desserts. This is the same family of methods that powered the Netflix Prize winning solution.

### RMSE vs Precision@K
- **RMSE** tells you how well you predict *exact ratings* (regression metric)
- **Precision@K** tells you how well you identify *what users will like* (ranking metric)
- Both together give a complete picture of recommendation quality

## Future Improvements

1. **Hybrid Model**: Combine content-based and collaborative filtering for better coverage
2. **Neural Collaborative Filtering**: Use deep learning for more expressive models
3. **A/B Testing**: Validate recommendations with real user feedback
4. **Real-Time Serving**: Deploy with caching for low-latency recommendations
5. **Contextual Features**: Incorporate time of day, location, dietary restrictions
