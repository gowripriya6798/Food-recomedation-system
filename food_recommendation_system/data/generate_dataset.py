"""
=============================================================================
STEP 1: SYNTHETIC FOOD DATASET GENERATION
=============================================================================

Why synthetic data?
    In a real-world scenario, you would pull data from sources like Yelp,
    Zomato, or internal databases. Here we generate realistic synthetic data
    so the project is fully self-contained and reproducible.

What we generate:
    1. foods_df   – A catalog of food items with attributes like cuisine,
                    category, and ingredient tags (used by content-based
                    filtering).
    2. ratings_df – A user-item interaction matrix where users rate food
                    items on a 1-5 scale (used by collaborative filtering).

Design choices:
    - 200 food items across 8 cuisines and 5 categories give enough variety
      for meaningful recommendations.
    - 50 users with ~20-40 ratings each create a sparse but usable matrix
      (sparsity ≈ 70-85 %), which is realistic for recommendation systems.
    - A small fraction of ratings are intentionally dropped to simulate
      missing data for the preprocessing step.
=============================================================================
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Food catalog constants
# ---------------------------------------------------------------------------
CUISINES = [
    "Italian", "Mexican", "Indian", "Chinese",
    "Japanese", "American", "Thai", "Mediterranean",
]

CATEGORIES = ["Appetizer", "Main Course", "Dessert", "Beverage", "Snack"]

INGREDIENT_POOLS = {
    "Italian": ["tomato", "basil", "mozzarella", "olive oil", "garlic", "parmesan", "pasta", "oregano"],
    "Mexican": ["tortilla", "beans", "avocado", "jalapeño", "cilantro", "lime", "cheese", "salsa"],
    "Indian": ["cumin", "turmeric", "garam masala", "onion", "ginger", "garlic", "chili", "yogurt"],
    "Chinese": ["soy sauce", "ginger", "garlic", "sesame oil", "rice", "tofu", "scallion", "chili"],
    "Japanese": ["soy sauce", "miso", "seaweed", "rice", "wasabi", "ginger", "sesame", "dashi"],
    "American": ["cheese", "beef", "lettuce", "tomato", "mustard", "bacon", "bread", "potato"],
    "Thai": ["lemongrass", "coconut milk", "basil", "chili", "fish sauce", "lime", "ginger", "garlic"],
    "Mediterranean": ["olive oil", "feta", "cucumber", "tomato", "hummus", "pita", "lemon", "oregano"],
}

FOOD_NAMES = {
    "Italian": [
        "Margherita Pizza", "Spaghetti Carbonara", "Bruschetta", "Tiramisu",
        "Risotto", "Lasagna", "Panna Cotta", "Minestrone Soup",
        "Caprese Salad", "Fettuccine Alfredo", "Gelato", "Arancini",
        "Osso Buco", "Focaccia", "Cannoli", "Gnocchi",
        "Prosciutto Wrapped Melon", "Ravioli", "Pesto Pasta", "Espresso",
        "Limoncello Sorbet", "Calzone", "Ciabatta", "Polenta",
        "Vitello Tonnato",
    ],
    "Mexican": [
        "Tacos al Pastor", "Guacamole", "Churros", "Burrito Bowl",
        "Enchiladas", "Quesadilla", "Elote", "Pozole",
        "Tamales", "Nachos Supreme", "Horchata", "Mole Poblano",
        "Chilaquiles", "Ceviche", "Tostadas", "Tres Leches Cake",
        "Mexican Street Corn", "Fajitas", "Sopapillas", "Agua Fresca",
        "Carnitas", "Chiles Rellenos", "Birria Tacos", "Sopes",
        "Paleta",
    ],
    "Indian": [
        "Butter Chicken", "Samosa", "Biryani", "Gulab Jamun",
        "Palak Paneer", "Masala Dosa", "Naan Bread", "Tandoori Chicken",
        "Chole Bhature", "Mango Lassi", "Raita", "Aloo Gobi",
        "Rogan Josh", "Pav Bhaji", "Jalebi", "Paneer Tikka",
        "Idli Sambhar", "Vindaloo", "Kulfi", "Chai",
        "Dal Makhani", "Malai Kofta", "Rasam", "Pani Puri",
        "Kheer",
    ],
    "Chinese": [
        "Kung Pao Chicken", "Dim Sum", "Fried Rice", "Peking Duck",
        "Spring Rolls", "Mapo Tofu", "Wonton Soup", "Chow Mein",
        "Sweet and Sour Pork", "Hot and Sour Soup", "Bubble Tea",
        "Dumplings", "General Tso Chicken", "Egg Drop Soup",
        "Char Siu Pork", "Dan Dan Noodles", "Congee", "Scallion Pancake",
        "Steamed Fish", "Tea Eggs", "Sesame Balls", "Bao Buns",
        "Ma La Xiang Guo", "Zongzi", "Mooncake",
    ],
    "Japanese": [
        "Sushi Roll", "Ramen", "Tempura", "Matcha Cake",
        "Miso Soup", "Tonkatsu", "Edamame", "Takoyaki",
        "Udon Noodles", "Onigiri", "Green Tea Ice Cream", "Gyoza",
        "Sashimi", "Okonomiyaki", "Yakitori", "Mochi",
        "Katsu Curry", "Dorayaki", "Taiyaki", "Sake",
        "Oyakodon", "Chawanmushi", "Karaage", "Nikujaga",
        "Anmitsu",
    ],
    "American": [
        "Classic Burger", "BBQ Ribs", "Mac and Cheese", "Apple Pie",
        "Caesar Salad", "Hot Dog", "Clam Chowder", "Buffalo Wings",
        "Philly Cheesesteak", "Pancakes", "Milkshake", "Corn Bread",
        "Fried Chicken", "BLT Sandwich", "Coleslaw", "Brownies",
        "Pulled Pork", "Biscuits and Gravy", "Key Lime Pie", "Root Beer Float",
        "Cobb Salad", "Meatloaf", "Grilled Cheese", "Onion Rings",
        "Banana Split",
    ],
    "Thai": [
        "Pad Thai", "Green Curry", "Tom Yum Soup", "Mango Sticky Rice",
        "Papaya Salad", "Massaman Curry", "Thai Iced Tea", "Satay Skewers",
        "Larb", "Pad See Ew", "Coconut Soup", "Thai Spring Rolls",
        "Red Curry", "Khao Pad", "Pineapple Fried Rice", "Sticky Rice",
        "Thai Basil Chicken", "Panang Curry", "Thai Fish Cakes", "Tom Kha Gai",
        "Boat Noodles", "Pad Kra Pao", "Crispy Pork Belly", "Som Tum",
        "Thai Milk Tea",
    ],
    "Mediterranean": [
        "Falafel", "Hummus Plate", "Greek Salad", "Baklava",
        "Shawarma", "Tabbouleh", "Stuffed Grape Leaves", "Kebab Platter",
        "Spanakopita", "Tzatziki", "Lemonade", "Pita Bread",
        "Moussaka", "Fattoush", "Baba Ganoush", "Kunafa",
        "Shakshuka", "Kibbeh", "Labneh", "Halloumi Fries",
        "Lamb Kofta", "Couscous Bowl", "Harissa Chicken", "Za'atar Bread",
        "Turkish Delight",
    ],
}


def generate_foods_dataframe(seed: int = 42) -> pd.DataFrame:
    """
    Generate a DataFrame of food items with attributes.

    Each food item has:
        - food_id   : unique integer identifier
        - name      : human-readable name
        - cuisine   : one of 8 world cuisines
        - category  : one of 5 meal categories
        - ingredients: space-separated ingredient tags (for TF-IDF)

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Food catalog with columns [food_id, name, cuisine, category, ingredients].
    """
    rng = np.random.RandomState(seed)
    records = []
    food_id = 1

    for cuisine in CUISINES:
        names = FOOD_NAMES[cuisine]
        pool = INGREDIENT_POOLS[cuisine]
        for name in names:
            # Randomly assign a category
            category = rng.choice(CATEGORIES)
            # Pick 3-5 random ingredients from the cuisine's pool
            n_ingredients = rng.randint(3, 6)
            chosen = rng.choice(pool, size=min(n_ingredients, len(pool)), replace=False)
            ingredients = " ".join(chosen)

            records.append({
                "food_id": food_id,
                "name": name,
                "cuisine": cuisine,
                "category": category,
                "ingredients": ingredients,
            })
            food_id += 1

    return pd.DataFrame(records)


def generate_ratings_dataframe(
    foods_df: pd.DataFrame,
    n_users: int = 50,
    min_ratings_per_user: int = 30,
    max_ratings_per_user: int = 60,
    missing_rate: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a DataFrame of user-food ratings with cuisine-preference bias.

    Simulates realistic rating behaviour:
        - Each user has 2-3 "favourite" cuisines and rates those higher.
        - Ratings for preferred cuisines are centred at 4.0, others at 2.5.
        - This creates meaningful user-preference patterns that SVD can
          discover (e.g., users who love Italian also tend to rate
          Mediterranean highly).
        - A small percentage of ratings are dropped (set to NaN) to
          simulate missing data for the preprocessing step.

    Parameters
    ----------
    foods_df : pd.DataFrame
        Food catalog (must contain ``food_id`` and ``cuisine`` columns).
    n_users : int
        Number of synthetic users.
    min_ratings_per_user : int
        Minimum foods each user rates.
    max_ratings_per_user : int
        Maximum foods each user rates.
    missing_rate : float
        Fraction of ratings to set to NaN (simulates missing values).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Ratings with columns [user_id, food_id, rating].
    """
    rng = np.random.RandomState(seed)
    all_cuisines = list(foods_df["cuisine"].unique())

    # Build a lookup: food_id → cuisine
    food_cuisine_map = dict(zip(foods_df["food_id"], foods_df["cuisine"]))
    food_ids = foods_df["food_id"].values
    records = []

    for user_id in range(1, n_users + 1):
        # Each user prefers 2-3 cuisines (creates collaborative signal)
        n_preferred = rng.randint(2, 4)
        preferred_cuisines = set(rng.choice(all_cuisines, size=n_preferred, replace=False))

        n_ratings = rng.randint(min_ratings_per_user, max_ratings_per_user + 1)
        rated_foods = rng.choice(food_ids, size=n_ratings, replace=False)

        for fid in rated_foods:
            cuisine = food_cuisine_map[fid]

            # Bias rating based on cuisine preference
            if cuisine in preferred_cuisines:
                base_rating = rng.normal(loc=4.0, scale=0.7)
            else:
                base_rating = rng.normal(loc=2.5, scale=0.8)

            rating = np.clip(base_rating, 1.0, 5.0)
            rating = round(rating * 2) / 2  # round to nearest 0.5

            records.append({
                "user_id": user_id,
                "food_id": int(fid),
                "rating": rating,
            })

    df = pd.DataFrame(records)

    # Introduce missing values to simulate real-world messiness
    n_missing = int(len(df) * missing_rate)
    missing_idx = rng.choice(df.index, size=n_missing, replace=False)
    df.loc[missing_idx, "rating"] = np.nan

    return df


def load_datasets(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function: generate and return both DataFrames.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (foods_df, ratings_df)
    """
    foods_df = generate_foods_dataframe(seed=seed)
    ratings_df = generate_ratings_dataframe(foods_df, seed=seed)
    return foods_df, ratings_df
