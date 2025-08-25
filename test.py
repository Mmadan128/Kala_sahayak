# test_script.py
import os
from crew_mvp import run_workflow

# --- Define Test Inputs ---
ARTISAN_ID = "artisan_lc8765"
IMAGE_PATH = "sample_data/product_image.jpg"
NOTE_PATH = "sample_data/artisan_note.txt"

def setup_test_files():
    """Ensures that dummy files exist for the test to run."""
    if not os.path.exists("sample_data"):
        os.makedirs("sample_data")
    
    if not os.path.exists(NOTE_PATH):
        print(f"Creating dummy note file at: {NOTE_PATH}")
        with open(NOTE_PATH, 'w') as f:
            f.write("This is a test note for a beautiful, handmade craft from our village.")
            
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Sample image not found at '{IMAGE_PATH}'.")
        print("Please place an image file at that location to run the test.")
        return False
    return True

def test_langchain_workflow():
    """
    A simple test function to run the entire LangChain workflow and validate the result.
    """
    print("--- Initializing Test for Kala Sahayak LangChain Workflow ---")
    
    if not setup_test_files():
        return

    # Read the content from the artisan's note file
    with open(NOTE_PATH, 'r') as f:
        artisan_note_content = f.read()

    print(f"\nArtisan ID: {ARTISAN_ID}")
    print(f"Input Image: {IMAGE_PATH}")
    print(f"Input Note: '{artisan_note_content}'")
    
    # Execute the workflow
    final_result = run_workflow(
        image_path=IMAGE_PATH,
        artisan_note=artisan_note_content,
        artisan_id=ARTISAN_ID
    )

    # Print and validate the final output
    print("\n--- Final Test Result ---")
    print(final_result)
    
    assert final_result is not None, "The result should not be None."
    assert "SUCCESS" in final_result, "The final output must be a success message."
    assert "https://www.kalasahayk.com" in final_result, "The final output must contain a public URL."
    assert ARTISAN_ID in final_result, "The artisan's ID should be in the final URL."
    
    print("\n✅ Test Completed Successfully!")
    print("The agentic workflow using only LangChain produced the expected final output.")

if __name__ == "__main__":
    test_langchain_workflow()