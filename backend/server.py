import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import os
import google.generativeai as genai
import time
import lime
from lime.lime_text import LimeTextExplainer
import json


GEMINI_API_KEY = "AIzaSyDBN4rYQA-T5_-X7JwU0OSJKAg6LPMgvbg"

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) 

# --- Global Variables for Models & Data ---
products_df = None
vectorizer = None
model = None
gemini_model = None
explainer = None

# --- Model & Data Loading ---
def load_resources():
    global products_df, vectorizer, model, gemini_model, explainer
    try:
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'products_with_scores.csv')
        products_df = pd.read_csv(data_path)
        products_df['product_price'] = pd.to_numeric(products_df['product_price'], errors='coerce')
        print("Successfully loaded product data.")
    except Exception as e:
        print(f"Error loading product CSV: {e}")
        return False

    try:
        vectorizer_path = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        print("Successfully loaded TF-IDF vectorizer.")
    except Exception as e:
        print(f"Error loading vectorizer: {e}")
        return False

    try:
        model_path = os.path.join(os.path.dirname(__file__), 'ethical_model.keras')
        model = load_model(model_path)
        print("Successfully loaded custom-trained neural network.")
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        return False

    try:
        if not GEMINI_API_KEY:
            print("WARNING: Gemini API Key not found in .env file. Chatbot will be disabled.")
            gemini_model = None
        else:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            print("Successfully configured Gemini API.")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        gemini_model = None
    
    explainer = LimeTextExplainer(class_names=['environmental_impact', 'labor_rights', 'animal_welfare', 'corporate_governance'])
    print("Successfully initialized LIME explainer.")
    return True

# --- Expanded Toolkit for the Chatbot ---

def get_product_details(product_name):
    product_series = products_df[products_df['product_name'].str.lower() == product_name.lower()]
    if product_series.empty: return {"error": "Product not found."}
    product_data = product_series.iloc[0].to_dict()
    review_text = product_data.get('reviews', '')
    vectorized_text = vectorizer.transform([review_text]).toarray()
    prediction = model.predict(vectorized_text)
    scores = np.clip(prediction[0], 0, 10)
    product_data['scores'] = { "Environmental": f"{scores[0]:.1f}", "Labor Rights": f"{scores[1]:.1f}", "Animal Welfare": f"{scores[2]:.1f}", "Governance": f"{scores[3]:.1f}" }
    return product_data

def get_recommendations(category, top_n=5):
    category_df = products_df[products_df['category'].str.lower() == category.lower()].copy()
    if category_df.empty: return {"error": f"Sorry, I don't have a '{category}' category."}
    reviews = category_df['reviews'].tolist()
    vectorized_reviews = vectorizer.transform(reviews).toarray()
    predictions = model.predict(vectorized_reviews)
    category_df['avg_ethical_score'] = np.mean(predictions, axis=1)
    top_products = category_df.nlargest(top_n, 'avg_ethical_score')
    return top_products[['product_name', 'avg_ethical_score']].to_dict(orient='records')

def compare_products(product_name_a, product_name_b):
    product_a = get_product_details(product_name_a)
    product_b = get_product_details(product_name_b)
    if "error" in product_a or "error" in product_b:
        return {"error": "One or both products could not be found. Please check the names."}
    return {
        "product_a": {"name": product_a['product_name'], "scores": product_a['scores']},
        "product_b": {"name": product_b['product_name'], "scores": product_b['scores']}
    }

def query_products_by_price(category, order='cheapest', top_n=3):
    category_df = products_df[products_df['category'].str.lower() == category.lower()]
    if category_df.empty: return {"error": f"Category '{category}' not found."}
    
    if order == 'cheapest':
        products = category_df.nsmallest(top_n, 'product_price')
    elif order == 'most_expensive':
        products = category_df.nlargest(top_n, 'product_price')
    else:
        return {"error": "Invalid order criteria. Please use 'cheapest' or 'most_expensive'."}
    
    return products[['product_name', 'product_price']].to_dict(orient='records')

def find_similar_products(product_name, top_n=3):
    target_product_series = products_df[products_df['product_name'].str.lower() == product_name.lower()]
    if target_product_series.empty: return {"error": "Target product not found."}
    target_product = target_product_series.iloc[0]
    target_category = target_product['category']
    category_df = products_df[products_df['category'] == target_category].copy()
    reviews = category_df['reviews'].tolist()
    vectorized_reviews = vectorizer.transform(reviews).toarray()
    predictions = model.predict(vectorized_reviews)
    target_prediction = predictions[category_df.index.get_loc(target_product.name)]
    distances = np.linalg.norm(predictions - target_prediction, axis=1)
    category_df['similarity_distance'] = distances
    similar_products = category_df.nsmallest(top_n + 1, 'similarity_distance')
    similar_products = similar_products[similar_products['product_name'] != product_name]
    return similar_products.head(top_n)[['product_name']].to_dict(orient='records')

def get_brand_info(brand_name):
    if 'created_by' not in products_df.columns:
        return {"error": "Brand information is not available in the current dataset."}
    brand_df = products_df[products_df['created_by'].str.lower() == brand_name.lower()].copy()
    if brand_df.empty: return {"error": f"Brand '{brand_name}' not found in my dataset."}
    reviews = brand_df['reviews'].tolist()
    vectorized_reviews = vectorizer.transform(reviews).toarray()
    predictions = model.predict(vectorized_reviews)
    avg_scores = np.mean(predictions, axis=0)
    return { "brand_name": brand_name, "product_count": len(brand_df), "average_scores": { "Environmental": f"{avg_scores[0]:.1f}", "Labor Rights": f"{avg_scores[1]:.1f}", "Animal Welfare": f"{avg_scores[2]:.1f}", "Governance": f"{avg_scores[3]:.1f}" } }

# --- API Endpoints ---
@app.route('/api/products', methods=['GET'])
def get_products_endpoint():
    if products_df is not None:
        explorer_df = products_df[['product_id', 'product_name', 'product_price', 'category', 'reviews']]
        return jsonify(explorer_df.to_dict(orient='records'))
    else: return jsonify({"error": "Products not loaded"}), 500

@app.route('/api/predict', methods=['POST'])
def predict_scores_endpoint():
    data = request.get_json()
    review_text = data.get('reviews', '')
    vectorized_text = vectorizer.transform([review_text]).toarray()
    prediction = model.predict(vectorized_text)
    scores = np.clip(prediction[0], 0, 10)
    result = { "environmental_impact_score": float(scores[0]), "labor_rights_score": float(scores[1]), "animal_welfare_score": float(scores[2]), "corporate_governance_score": float(scores[3]) }
    return jsonify(result)

@app.route('/api/snapshot', methods=['POST'])
def get_snapshot_endpoint():
    if not gemini_model: return jsonify({"summary": "Ethical Snapshot feature is currently disabled."})
    data = request.get_json()
    time.sleep(2)
    product_name = data.get("product_name")
    scores = data.get("scores")
    prompt = f"You are an ethical shopping assistant. Write a short, 2-3 sentence summary for the product '{product_name}' based on these scores (out of 10): Env: {scores['environmental_impact_score']:.1f}, Labor: {scores['labor_rights_score']:.1f}, Animal Welfare: {scores['animal_welfare_score']:.1f}, Governance: {scores['corporate_governance_score']:.1f}. Praise high scores (8+) and mention low scores (below 5) as concerns."
    response = gemini_model.generate_content(prompt)
    return jsonify({"summary": response.text})

@app.route('/api/explain', methods=['POST'])
def explain_prediction_endpoint():
    data = request.get_json()
    review_text = data.get('reviews', '')
    explanation = explainer.explain_instance(review_text, lime_predict_function, num_features=5, labels=(0, 1, 2, 3))
    explanation_data = {
        'environmental': [word for word, weight in explanation.as_list(label=0) if weight > 0],
        'labor': [word for word, weight in explanation.as_list(label=1) if weight > 0],
        'animal_welfare': [word for word, weight in explanation.as_list(label=2) if weight > 0],
        'governance': [word for word, weight in explanation.as_list(label=3) if weight > 0],
    }
    return jsonify(explanation_data)

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    if not gemini_model: return jsonify({"reply": "I'm sorry, my connection to the AI assistant is currently offline."})
    
    data = request.get_json()
    if not data or 'history' not in data: return jsonify({"reply": "Invalid request format."})
    
    history = data.get('history')
    user_query = history[-1]['parts'][0]['text'] if history else ""

    system_prompt = """
    You are an AI orchestrator. Your job is to determine which tool to call based on the user's query and the conversation history.
    You must respond ONLY with a JSON object containing a "tool" and "parameters".
    If the user's query does not specify a number for `top_n`, you should default to `top_n=3`.
    If the user is just greeting, making small talk, or the query doesn't fit any other tool, you MUST use the `general_knowledge` tool.

    Your available tools are:
    1. `get_product_details(product_name)`
    2. `get_recommendations(category, top_n=5)`
    3. `compare_products(product_name_a, product_name_b)`
    4. `query_products_by_price(category, order='cheapest', top_n=3)`
    5. `find_similar_products(product_name, top_n=3)`
    6. `get_brand_info(brand_name)`
    7. `general_knowledge()`
    """
    try:
        chat_session = gemini_model.start_chat(history=history[:-1])
        
        intent_response = chat_session.send_message(f"User Query: \"{user_query}\"\n\n{system_prompt}")
        
        # FIXED: Complete and robust JSON parsing logic
        raw_text = intent_response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:-3].strip()
        
        try:
            intent_data = json.loads(raw_text)
        except json.JSONDecodeError:
            print(f"Warning: Gemini did not return valid JSON for intent. Defaulting to general_knowledge. Response was: {raw_text}")
            intent_data = {"tool": "general_knowledge", "parameters": {}}

        tool_to_call = intent_data.get("tool")
        parameters = intent_data.get("parameters", {})

        tool_result = None
        if tool_to_call == "get_product_details": tool_result = get_product_details(**parameters)
        elif tool_to_call == "get_recommendations": tool_result = get_recommendations(**parameters)
        elif tool_to_call == "compare_products": tool_result = compare_products(**parameters)
        elif tool_to_call == "query_products_by_price": tool_result = query_products_by_price(**parameters)
        elif tool_to_call == "find_similar_products": tool_result = find_similar_products(**parameters)
        elif tool_to_call == "get_brand_info": tool_result = get_brand_info(**parameters)
        else: tool_result = {"query": user_query}

        response_prompt = f"""
        You are Conscia, a friendly AI assistant. Formulate a natural, conversational response based on the tool's result, keeping the entire chat history in mind.
        - NEVER just output raw JSON.
        - For lists of products, format them clearly with bullet points.
        - Be helpful and clear.

        FULL CONVERSATION HISTORY: {json.dumps(history, indent=2)}
        TOOL RESULT FOR LATEST QUERY: {json.dumps(tool_result, indent=2)}

        Now, write your friendly reply to the user.
        """
        final_response = chat_session.send_message(response_prompt)
        return jsonify({"reply": final_response.text})

    except Exception as e:
        if "429" in str(e) and "quota" in str(e).lower():
            print("Chatbot error: Gemini API daily quota exceeded.")
            return jsonify({"reply": "I'm sorry, but I've reached my daily request limit for today. My advanced features will be available again tomorrow. Thank you for your understanding!"})
        else:
            print(f"Chatbot error: {e}")
            return jsonify({"reply": "I'm sorry, I had a little trouble processing that. Could you please rephrase your question?"})

def lime_predict_function(texts):
    """Helper function for LIME."""
    vectorized_texts = vectorizer.transform(texts).toarray()
    return model.predict(vectorized_texts)

if __name__ == '__main__':
    if load_resources():
        print("All resources loaded. Starting Conscia Universal Server...")
        app.run(debug=True, port=5000)
    else:
        print("Failed to load resources. Server will not start.")




