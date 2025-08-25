# main.py
import os
import json
import ast
from typing import Dict, Any
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# --- Step 1: Load Environment Variables and Initialize LLM ---
load_dotenv()

try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        verbose=True,
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
except Exception as e:
    print(f"Error initializing Gemini LLM: {e}")
    llm = None

# --- Step 2: Define All Tools ---

@tool("Image Enhancing Tool")
def enhance_image(image_path: str) -> str:
    """Processes a raw product image and returns the path to the enhanced version."""
    print(f"--- (Tool Used) Enhancing image: {image_path} ---")
    if not os.path.exists(image_path):
        return "Error: Image path does not exist."
    filename = os.path.basename(image_path)
    processed_path = f"processed_{filename}"
    return processed_path

@tool("Storytelling & Hashtag Tool")
def generate_narrative(artisan_note: str) -> str:
    """Generates a product story and hashtags from an artisan's note."""
    print(f"--- (Tool Used) Generating narrative from note: '{artisan_note[:30]}...' ---")
    story = f"Inspired by the artisan's words: '{artisan_note}', this piece is a testament to traditional craftsmanship."
    hashtags = ["#Handmade", "#ArtisanCraft", "#IndianHeritage", "#SupportLocal"]
    return json.dumps({"product_description": story, "hashtags": hashtags})

@tool("Market Pricing Tool")
def recommend_price(product_description: str) -> float:
    """Recommends a market price based on the product description."""
    print(f"--- (Tool Used) Recommending price for description: '{product_description[:30]}...' ---")
    price = 75.0 + (len(product_description) / 25.0)
    return round(price, 2)

@tool("Data Consolidation Tool")
def consolidate_data(data_string: str) -> str:
    """
    Takes a string representation of a Python dictionary containing all product details and consolidates them into a final JSON string.
    The input string MUST be a valid Python dictionary format and contain the keys: 'artisan_id', 'enhanced_image_path', 'description', 'hashtags', and 'price'.
    This is the last step before publishing.
    """
    print("--- (Tool Used) Consolidating all data into final JSON ---")
    try:
        data = ast.literal_eval(data_string)
        if not isinstance(data, dict):
            raise TypeError("Input is not a dictionary.")
            
    except (ValueError, SyntaxError, TypeError) as e:
        return f"Error: Input is not a valid Python dictionary string. Details: {e}"

    required_keys = ['artisan_id', 'enhanced_image_path', 'description', 'hashtags', 'price']
    if not all(key in data for key in required_keys):
        return f"Error: The parsed dictionary is missing one of the required keys: {required_keys}"
    
    return json.dumps(data)

@tool("Web Publishing Tool")
def publish_to_web_gallery(final_product_json: str) -> str:
    """Publishes the final product data to a web gallery, returning a public URL."""
    print("--- (Tool Used) Publishing to Web Gallery ---")
    try:
        data = json.loads(final_product_json)
        artisan_id = data.get("artisan_id", "unknown_artisan")
        product_id = hash(data.get('description', '')) % 10000
        public_url = f"https://www.kalasahayk.com/gallery/{artisan_id}/product{product_id}"
        print(f"--- API CALL SIMULATED: {json.dumps(data, indent=2)} ---")
        return f"SUCCESS: Product live at {public_url}"
    except Exception as e:
        return f"Error publishing: {e}"

# --- Step 3: Create a Standard ReAct Prompt Template ---
REACT_PROMPT_TEMPLATE = """
You are an expert assistant. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Now, begin!

{system_message}
Question: {input}
{agent_scratchpad}
"""

# --- Step 4: Create Specialized Agent Chains ---
darpan_system_message = "You are a specialized agent that operates a single tool. Your ONLY job is to take the user's input (an image path) and pass it to the 'Image Enhancing Tool'. You MUST return the raw, unmodified output from this tool as your final answer."
darpan_prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE.format(system_message=darpan_system_message, tools="{tools}", tool_names="{tool_names}", input="{input}", agent_scratchpad="{agent_scratchpad}"))
darpan_tools = [enhance_image]
darpan_agent = create_react_agent(llm, darpan_tools, darpan_prompt)
darpan_executor = AgentExecutor(agent=darpan_agent, tools=darpan_tools, verbose=True, handle_parsing_errors=True)

katha_system_message = "You are a specialized agent that operates a single tool. Your ONLY job is to take the user's input (an artisan's note) and pass it to the 'Storytelling & Hashtag Tool'. You MUST return the raw, unmodified JSON output from this tool as your final answer. Do not add any extra text or explanations."
katha_prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE.format(system_message=katha_system_message, tools="{tools}", tool_names="{tool_names}", input="{input}", agent_scratchpad="{agent_scratchpad}"))
katha_tools = [generate_narrative]
katha_agent = create_react_agent(llm, katha_tools, katha_prompt)
katha_executor = AgentExecutor(agent=katha_agent, tools=katha_tools, verbose=True, handle_parsing_errors=True)

# AGENT 3: PRASAR (Publishing Agent)
# FIX: Added a final sentence to make the output predictable.
prasar_system_message = """You are Prasar, a digital publishing manager. Your job is to take product information and publish it.
Your input is a JSON string containing the initial product details.

Follow these steps STRICTLY:
1. Parse the input JSON string to understand the product details.
2. Use the 'Market Pricing Tool' on the product's 'description' to get a price.
3. In your thought process, construct a complete Python dictionary that contains all the original details PLUS the new price.
4. Use the 'Data Consolidation Tool'. The Action Input for this tool MUST be the string representation of the complete Python dictionary you just created.
5. Use the 'Web Publishing Tool' with the final JSON string output from the consolidation tool to publish the product.
6. Your Final Answer MUST be the complete, raw, unmodified string output from the 'Web Publishing Tool'.
"""
prasar_prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE.format(system_message=prasar_system_message, tools="{tools}", tool_names="{tool_names}", input="{input}", agent_scratchpad="{agent_scratchpad}"))
prasar_tools = [recommend_price, consolidate_data, publish_to_web_gallery]
prasar_agent = create_react_agent(llm, prasar_tools, prasar_prompt)
prasar_executor = AgentExecutor(agent=prasar_agent, tools=prasar_tools, verbose=True, handle_parsing_errors=True)


# --- Step 5: Chain the Agents Together into a Sequential Workflow ---
def prepare_for_publishing(results: dict) -> dict:
    print("--- Preparing data for final publishing step ---")
    visual_output = results['visual_output']['output']
    narrative_output = json.loads(results['narrative_output']['output'])
    
    final_input_data = {
        "artisan_id": results['original_inputs']['artisan_id'],
        "enhanced_image_path": visual_output,
        "description": narrative_output['product_description'],
        "hashtags": narrative_output['hashtags']
    }
    return {"input": json.dumps(final_input_data)}

workflow = (
    RunnablePassthrough()
    .assign(
        visual_output=RunnableLambda(lambda x: {"input": x['raw_image_path']}) | darpan_executor,
        narrative_output=RunnableLambda(lambda x: {"input": x['artisan_note']}) | katha_executor,
        original_inputs=RunnableLambda(lambda x: x)
    )
    | RunnableLambda(prepare_for_publishing)
    | prasar_executor
)

def run_workflow(image_path: str, artisan_note: str, artisan_id: str):
    """Initializes and runs the full LangChain workflow."""
    if not llm:
        return "ERROR: LLM not initialized. Please check your API key."

    inputs = {
        "raw_image_path": image_path,
        "artisan_note": artisan_note,
        "artisan_id": artisan_id
    }
    
    print("\n--- Starting Kala Sahayak LangChain Workflow ---")
    result = workflow.invoke(inputs)
    print("\n--- LangChain Workflow Finished ---")
    
    return result['output']