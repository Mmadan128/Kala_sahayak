# app.py
import streamlit as st
import os
import json
import requests
from typing import Dict, Any
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import time

# --- 1. CONFIGURATION & SETUP ---
load_dotenv()

st.set_page_config(page_title="Kala Sahayak", layout="wide", initial_sidebar_state="expanded")

def setup_api_keys():
    """Handles the setup of API keys via the Streamlit sidebar."""
    with st.sidebar:
        st.header("🔑 API Keys")
        st.markdown("Enter your API keys below to activate the application.")
        google_api_key = st.text_input("Google Gemini API Key", type="password", help="Get yours from Google AI Studio.")
        clipdrop_api_key = st.text_input("Clipdrop API Key", type="password", help="Get yours from clipdrop.co.")

        # Strip whitespace from keys to prevent API header errors.
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key.strip()
        if clipdrop_api_key:
            os.environ["CLIPDROP_API_KEY"] = clipdrop_api_key.strip()
        
        st.info("Your keys are used for this session only and are not stored.")

# --- 2. REAL-WORLD AI TOOLS ---

@tool("Background Removal Tool")
def remove_background(image_path: str) -> str:
    """
    Removes the background from a product image using the Clipdrop API.
    Returns a JSON string with 'processed_image_path' on success or 'error' on failure.
    """
    api_key = os.getenv("CLIPDROP_API_KEY")
    if not api_key:
        return json.dumps({"error": "Clipdrop API Key is missing. Please add it in the sidebar."})
    
    url = "https://clipdrop-api.co/remove-background/v1"
    try:
        if not os.path.exists(image_path):
             return json.dumps({"error": f"The file path '{image_path}' does not exist."})

        with open(image_path, 'rb') as f:
            files = {'image_file': (os.path.basename(image_path), f)}
            headers = {'x-api-key': api_key}
            response = requests.post(url, files=files, headers=headers)
        
        response.raise_for_status() # Raises an exception for bad status codes (4xx or 5xx)

        processed_filename = f"processed_{os.path.basename(image_path)}"
        processed_path = os.path.join("temp_uploads", processed_filename)
        with open(processed_path, 'wb') as f:
            f.write(response.content)
        return json.dumps({"processed_image_path": processed_path})
    except requests.exceptions.HTTPError as e:
        # Provide a user-friendly message for the most common error (invalid key)
        if e.response.status_code == 401 or e.response.status_code == 403:
            return json.dumps({"error": "Clipdrop API Error: Invalid API Key. Please re-copy the key from the official website and paste it in the sidebar."})
        return json.dumps({"error": f"Clipdrop API Error: {e.response.status_code} - {e.response.text}"})
    except Exception as e:
        return json.dumps({"error": f"An unexpected error occurred during background removal: {str(e)}"})

@tool("Creative Content Tool")
def generate_creative_content(artisan_note: str) -> str:
    """
    Generates a product description, hashtags, and price from an artisan's note using a powerful LLM.
    Returns the data as a single JSON string.
    """
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7, google_api_key=os.getenv("GOOGLE_API_KEY"))
        
        prompt = PromptTemplate.from_template(
            """As a world-class e-commerce marketer for handmade crafts, analyze this artisan's note: "{note}"
            
            Your tasks:
            1. Write a compelling, evocative product description.
            2. Suggest 5 relevant and effective social media hashtags.
            3. Recommend a fair market price in USD (e.g., 49.99).
            
            Return a single, valid JSON object with keys: "description", "hashtags", "price".
            """
        )
        chain = prompt | llm
        result = chain.invoke({"note": artisan_note})
        cleaned_result = result.content.strip().replace("```json", "").replace("```", "")
        return cleaned_result
    except Exception as e:
        return json.dumps({"error": f"Error generating creative content: {str(e)}"})

# --- 3. STREAMLIT UI LAYOUT & ORCHESTRATION ---

def display_results(results: Dict[str, Any], user_price: float):
    """Renders the final product listing in the UI."""
    st.subheader("✅ Your Professional Listing is Ready!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Enhanced Photo")
        if "error_bg_removal" in results:
            st.error(f"Background Removal Failed: {results['error_bg_removal']}")
            st.image(st.session_state.original_image_path, caption="Original Image (Unprocessed)", use_container_width=True)
        else:
            processed_path = results.get('processed_image_path')
            if processed_path and os.path.exists(processed_path):
                 st.image(processed_path, caption="Background Removed", use_container_width=True)
            else:
                 st.warning("Could not locate the processed image.")
                 st.image(st.session_state.original_image_path, caption="Original Image", use_container_width=True)
    
    with col2:
        st.markdown("##### Pricing")
        ai_price = results.get('price')

        if user_price > 0:
            st.metric(label="Your Selling Price", value=f"Rs {user_price:.2f}")
            if isinstance(ai_price, (int, float)):
                 st.info(f"AI Suggestion: Rs {ai_price:.2f}")
        elif isinstance(ai_price, (int, float)):
            st.metric(label="AI Recommended Price", value=f"Rs{ai_price:.2f}")
        else:
            st.metric(label="Recommended Price", value="N/A")
        
        st.markdown("---")
        
        with st.expander("Product Story", expanded=True):
            st.write(results.get('description', 'Creative content generation failed.'))
        
        hashtags = results.get('hashtags')
        if hashtags and isinstance(hashtags, list):
            st.markdown("##### Social Media Hashtags")
            st.info(' '.join(hashtags))
        
        st.markdown("##### Mock URL")
        st.code(results.get('mock_url', 'N/A'), language='text')

def main():
    """The main function that runs the Streamlit application."""
    st.title("🎨 Kala Sahayak - The Artisan's AI Co-pilot")
    st.markdown("Turn a simple photo and a note into a professional, market-ready product listing.")

    setup_api_keys()

    if not (os.getenv("GOOGLE_API_KEY") and os.getenv("CLIPDROP_API_KEY")):
        st.warning("Please enter your API keys in the sidebar to begin.")
        st.stop()
    
    st.success("API keys detected. You can now generate a listing.")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("1. Your Product Details")
        uploaded_file = st.file_uploader("Upload a photo of your product:", type=["jpg", "jpeg", "png"])
        
        placeholder_text = (
            "e.g., This is a hand-painted necklace made from terracotta clay, "
            "featuring a traditional 'Warli' art motif. The black beads are made of glass. "
            "It took me two days to paint the intricate details."
        )
        artisan_note = st.text_area("Describe your product (like a voice note):", height=150, placeholder=placeholder_text)
        st.caption("Pro-Tip: More details will result in a better story and price!")

        user_price = st.number_input("Your Selling Price (USD, Optional)", min_value=0.0, format="%.2f")
        
        if uploaded_file:
            if "original_image_path" not in st.session_state or st.session_state.get("uploaded_filename") != uploaded_file.name:
                temp_dir = "temp_uploads"
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                
                image_path = os.path.join(temp_dir, uploaded_file.name)
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.session_state.original_image_path = image_path
                st.session_state.uploaded_filename = uploaded_file.name

            st.image(st.session_state.original_image_path, caption="Your Uploaded Image", use_container_width=True)

            if artisan_note:
                if st.button("🚀 Generate Listing", type="primary", use_container_width=True):
                    final_results = {}
                    
                    with col2:
                        st.subheader("2. AI Co-pilot at Work")
                        with st.status("🤖 AI Co-pilot is initializing...", expanded=True) as status:
                            try:
                                # This part is now a direct function call, not an agent
                                status.write("Task 1: Enhancing product photo...")
                                bg_result_str = remove_background(st.session_state.original_image_path)
                                bg_result = json.loads(bg_result_str)
                                time.sleep(1)
                                if "error" in bg_result:
                                    status.warning(f"Photo enhancement failed.")
                                    final_results["processed_image_path"] = st.session_state.original_image_path
                                    final_results["error_bg_removal"] = bg_result['error']
                                else:
                                    status.write("✅ Photo enhanced successfully.")
                                    final_results["processed_image_path"] = bg_result['processed_image_path']
                                
                                status.write("Task 2: Crafting product story, hashtags, and price...")
                                content_result_str = generate_creative_content(artisan_note)
                                content_result = json.loads(content_result_str)
                                time.sleep(1)
                                if "error" in content_result:
                                     status.warning(f"Content generation failed.")
                                     final_results.update({"description": "N/A", "hashtags": [], "price": 0})
                                else:
                                    status.write("✅ Creative content generated.")
                                    final_results.update(content_result)
                                
                                status.write("Task 3: Finalizing listing...")
                                final_results["mock_url"] = f"https://kalasahayk.com/gallery/artisan123/product_{str(hash(artisan_note))[:6]}"
                                time.sleep(1)
                                
                                status.update(label="✅ Orchestration Complete!", state="complete", expanded=False)

                            except Exception as e:
                                status.update(label="🚨 Workflow Failed!", state="error")
                                st.error(f"A critical error occurred: {e}")
                        
                        st.markdown("---")
                        if final_results:
                            display_results(final_results, user_price)

if __name__ == "__main__":
    main()