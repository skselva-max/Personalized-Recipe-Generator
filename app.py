from flask import Flask, request, jsonify, render_template, send_from_directory
import google.generativeai as genai
import os
from dotenv import load_dotenv
import requests
import base64
from io import BytesIO
from PIL import Image
import uuid
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure APIs
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.0-flash', generation_config={
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
})

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

def generate_image(prompt):
    """Generate image using DreamStudio API"""
    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "text_prompts": [{"text": prompt, "weight": 1}],
        "width": 1024,
        "height": 1024,
        "samples": 1,
        "style_preset": "photographic"
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["artifacts"][0]["base64"]
    except Exception as e:
        logger.error(f"Image generation failed: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/saved-recipes')
def saved_recipes():
    return render_template('saved-recipes.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/style.css')
def serve_css():
    return send_from_directory('.', 'style.css')

@app.route('/generate_recipe', methods=['POST'])
def generate_recipe():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        # Extract parameters with language support
        ingredients = data.get('ingredients', '')
        language = data.get('language', 'English')
        prompt = data.get('prompt', '')
        strict_mode = data.get('strictMode', True)
        cuisine = data.get('cuisine', '')
        dietary_preferences = data.get('dietary_preferences', '')
        
        if not ingredients:
            return jsonify({"success": False, "error": "Please provide at least one ingredient"}), 400

        # Generate recipe text with strict/flexible mode
        logger.info(f"Generating recipe in {language} with strict mode: {strict_mode}")
        
        # Create the prompt based on strict/flexible mode
        if strict_mode:
            mode_instruction = f"""
            Create a recipe using ONLY these ingredients: {ingredients}.
            DO NOT add any extra ingredients beyond what the user has provided.
            If the recipe requires ingredients not listed, suggest alternatives from the provided ingredients.
            """
        else:
            mode_instruction = f"""
            Create a recipe using these primary ingredients: {ingredients}.
            You may add additional common ingredients to make the recipe complete.
            Keep the provided ingredients as the main focus of the dish.
            """
        
        full_prompt = f"""
        You are creating a recipe in {language} language.
        {mode_instruction}
        
        Additional requirements:
        - Cuisine type: {cuisine}
        - Dietary preferences: {dietary_preferences}
        - User notes: {prompt}
        
        The recipe should include:
        1. A creative recipe name
        2. Clear ingredients list with measurements
        3. Step-by-step instructions
        4. Cooking time and serving size
        5. Optional tips or variations
        
        IMPORTANT: Respond entirely in {language} language.
        """
        
        response = model.generate_content(full_prompt)
        recipe_text = response.text

        # Generate and save image
        image_prompt = (
            f"Professional food photo of {cuisine} cuisine dish "
            f"made with {ingredients}, {dietary_preferences} dietary, "
            "high quality, appetizing, well-lit"
        )
        
        image_data = generate_image(image_prompt)
        image_filename = f"recipe_{uuid.uuid4().hex}.png"
        image_path = os.path.join("static", image_filename)
        
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(image_data))

        return jsonify({
            "success": True,
            "recipe": recipe_text,
            "image_url": f"/static/{image_filename}"
        })

    except Exception as e:
        logger.error(f"Recipe generation error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)