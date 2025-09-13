import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- One-time setup: Download the VADER lexicon for sentiment analysis ---
# This is only needed the first time you run the script.
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError: # <-- This line has been corrected
    print("Downloading VADER lexicon for sentiment analysis (one-time setup)...")
    nltk.download('vader_lexicon')
# -------------------------------------------------------------------------

def train_and_score_model():
    """
    This function simulates a model training process.
    It reads the combined product data, analyzes reviews for sentiment and keywords,
    generates ethical scores, and saves the result to a new file.
    """
    input_filepath = 'data/all_products.csv'
    output_filepath = 'data/products_with_scores.csv'

    try:
        df = pd.read_csv(input_filepath)
    except FileNotFoundError:
        print(f"Error: '{input_filepath}' not found.")
        print("Please run the 'prepare_data.py' script first.")
        return

    # Initialize the sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # --- Define keywords for each ethical pillar ---
    # This is the core of our "model's" logic.
    keywords = {
        'environmental_impact': ['eco-friendly', 'sustainable', 'recycled', 'green', 'organic', 'biodegradable', 'low-impact'],
        'labor_rights': ['fair-trade', 'ethical', 'handmade', 'local', 'artisan', 'union'],
        'animal_welfare': ['cruelty-free', 'vegan', 'no-animal-testing', 'plant-based'],
        'corporate_governance': ['durable', 'quality', 'long-lasting', 'reliable', 'transparent', 'b-corp']
    }

    results = []

    print("\nStarting scoring process...")
    # --- Scoring Logic ---
    for index, row in df.iterrows():
        review_text = str(row['reviews']).lower()
        
        # 1. Sentiment Score
        # The 'compound' score is a single metric from -1 (most negative) to +1 (most positive)
        sentiment_score = sia.polarity_scores(review_text)['compound']
        
        # We'll normalize this to a 1-10 scale. Let's make 5 the neutral point.
        # A score of 0 (neutral sentiment) will become a 5.
        # A score of 1 (max positive) will become a 10.
        # A score of -1 (max negative) will become a 0.
        base_score = 5 * (sentiment_score + 1)

        # 2. Keyword Score for each pillar
        scores = {
            'environmental_impact': base_score,
            'labor_rights': base_score,
            'animal_welfare': base_score,
            'corporate_governance': base_score
        }

        # Add bonus points for each keyword found
        for pillar, key_list in keywords.items():
            for keyword in key_list:
                if keyword in review_text:
                    scores[pillar] += 1.5 # Add a bonus for finding a keyword
        
        # Cap scores at 10
        for pillar in scores:
            if scores[pillar] > 10:
                scores[pillar] = 10.0
            # Round to one decimal place
            scores[pillar] = round(scores[pillar], 1)

        # Append all data to our results list
        processed_row = {
            'product_id': row['product_id'],
            'product_name': row['product_name'],
            'product_price': row['product_price'],
            'category': row['category'],
            'reviews': row['reviews'],
            'environmental_impact_score': scores['environmental_impact'],
            'labor_rights_score': scores['labor_rights'],
            'animal_welfare_score': scores['animal_welfare'],
            'corporate_governance_score': scores['corporate_governance']
        }
        results.append(processed_row)
        
        # Print progress
        if (index + 1) % 500 == 0:
            print(f" - Processed {index + 1}/{len(df)} products...")

    # Create a new DataFrame with the results
    scored_df = pd.DataFrame(results)

    # Save the final dataset
    scored_df.to_csv(output_filepath, index=False)
    
    print("\nScoring process complete!")
    print(f"Saved scored data to '{output_filepath}'. You can inspect this file to see the results.")
    print(f"Total products scored: {len(scored_df)}")


if __name__ == '__main__':
    train_and_score_model()

