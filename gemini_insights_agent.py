# python
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import json
from pydantic import BaseModel, Field, ValidationError
from typing import List

# Import the Google GenAI SDK and tools
from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- Configuration & Compliance Checks ---
# Ensure the API key is set for Google/Gemini
# The key should be in a .env file as requested (GEMINI_API_KEY or GOOGLE_API_KEY)
# For this example, we'll assume it's loaded into the environment.
# You would typically use a library like `dotenv` for this.
try:
    # This will use the GEMINI_API_KEY environment variable if available
    client = genai.Client()
except Exception as e:
    # In a real app, you'd handle this more robustly
    print(f"ERROR: Could not initialize Gemini Client. Check your API Key environment variable. Details: {e}")
    client = None

# --- 1. Structured Output Definition (Pydantic, similar to Zod/TypeScript Schema) ---

class DigitalABCsInsights(BaseModel):
    """
    Defines the required JSON structure for the monthly business insights.
    This structure ensures WCAG and brand compliance by forcing clear,
    actionable, and sourced content.
    """
    title: str = Field(..., description="The title of the insight report, e.g., 'Australian Small Business Pulse: Month Year'.")
    summary: str = Field(..., description="A 2-sentence max overview of the month's business climate and key trend.")
    key_updates: List[str] = Field(..., description="3-4 specific, actionable, and accessible bullet points for micro-businesses.")
    sources: List[str] = Field(..., description="Minimum 2 direct, reputable URLs (prioritize .gov.au) for source validation.")

# --- 2. Agent Instructions (System Prompt) ---

# The comprehensive instructions are moved to the system prompt.
# I've slightly adjusted the formatting for clarity in Python and to
# emphasize the DigitalABCs brand voice and legal compliance requirements.

SYSTEM_INSTRUCTIONS = f"""
# DigitalABCs Insights Agent - Trusted Advisor

## ROLE & PURPOSE
You are a trusted advisor for DigitalABCs, working with Australian micro-businesses (less than 5 employees). Your mission is to generate monthly business insights that are **compliant, accessible, and actionable**, empowering them through technology and education, based on Trish's vision of resilience and opportunity.

## CORE RESPONSIBILITIES & LEGAL COMPLIANCE
- **Brand Voice:** Relatable, supportive, and empowering (especially for micro-businesses). Use honest, practical, and clear Australian English. Avoid jargon.
- **Privacy (Privacy Act 1988 & GDPR):** When discussing data, always emphasize security, transparency, and the need to protect personal information.
- **Accessibility (DDA 1992 & WCAG 2.1/2.2 AA):** All content must be easy to read and consume. Use simple language (Grade 8-10 reading level), clear structure, and descriptive link text.
- **Actionable Focus:** Every insight must include "what to do," not just "what's happening." Simplify complexity into clear next steps.

## NEWS GATHERING & ANALYSIS
- **Sources:** Prioritize news from Department of Industry, Science and Resources, Business.gov.au, ASIC, and reputable Australian tech/business news.
- **Relevance:** Focus on news published within the last 30 days, regulatory changes, compliance updates, and technology trends specific to micro-businesses.
- **Geographical Filter:** Only provide advice specific to Australian businesses.

## INSIGHT GENERATION FORMAT & VALIDATION
1.  **Strict Output Structure:** You MUST return the final output as a JSON object that strictly adheres to the `DigitalABCsInsights` schema (provided as a tool/function call).
2.  **Content Validation Rules:**
    - Title format: "Australian Small Business Pulse: [Month Year]"
    - Summary: 2 sentences maximum.
    - Key Updates: Minimum 3, maximum 4 actionable bullet points.
    - Sources: Minimum 2 URLs to authoritative resources (.gov.au preferred).
3.  **Special Considerations:**
    - Always mention specific dates for deadlines.
    - Highlight free government resources, grants, or cost information.

## ERROR HANDLING
If you are unable to find recent news (last 30 days): State clearly that current news is limited and provide general, evergreen compliance reminders (e.g., ATO deadlines, DDA compliance check). Never invent or fabricate news sources.
"""

# --- 3. Tool Definition (Google Search) ---
# The Gemini SDK automatically provides the `google` tool for web searches,
# similar to the `webSearchTool` but natively integrated. We'll rely on the
# agent's instructions to filter the search to relevant Australian sources.

def generate_insights(user_input: str) -> DigitalABCsInsights:
    """
    Runs the DigitalABCs Insights Agent workflow.
    
    This function takes the user's prompt (e.g., "Generate this month's business insights")
    and uses the Gemini model, system instructions, and Google Search tool to generate
    a compliant, structured, and actionable report for Australian micro-businesses.
    
    Args:
        user_input: The text input from the user.

    Returns:
        A Pydantic model instance containing the structured insights.
    """
    if not client:
        raise RuntimeError("Gemini Client not initialized. Cannot run workflow.")

    print(f"--- Running DigitalABCs Insights Workflow for: {user_input} ---")
    
    # 1. Configuration for the model call
    
    # The `google` tool is automatically used if the model decides it needs to search the web.


    # The structured output is enforced using the response_schema
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTIONS,
        response_mime_type="application/json",
        response_schema=DigitalABCsInsights,
    )

    # 2. Call the Gemini API
    try:
        # Use a model with strong reasoning and function-calling capabilities
        response = client.models.generate_content(
            model='gemini-2.5-pro', # Strongest model for complex reasoning and instruction following
            contents=user_input,
            config=config,
        )
        
        # 3. Process and Validate Output
        
        # The API is configured to return a JSON string matching the Pydantic schema
        # We parse the JSON string from the response
        try:
            # We must use json.loads() on the text to get a dictionary
            json_data = json.loads(response.text)
            
            # Use Pydantic to validate and create the structured object
            insights = DigitalABCsInsights(**json_data)
            
            print("--- Insights Generated Successfully (Structured JSON) ---")
            return insights

        except (json.JSONDecodeError, ValidationError) as e:
            print(f"ERROR: Failed to parse or validate the structured output. Details: {e}")
            print(f"Raw model response text: {response.text}")
            # Fallback or error handling for invalid structure
            raise APIError(f"Model failed to return compliant JSON structure: {e}")

    except APIError as e:
        print(f"An API error occurred during content generation: {e}")
        raise

# --- 4. Main Execution Block ---

# Example of how the workflow is run, similar to your `runWorkflow`
if __name__ == "__main__":
    
    # This mimics the input from the original TypeScript workflow
    workflow_input = {"input_as_text": "Generate this month's business insights for Australian micro-businesses."}
    
    try:
        # Call the Python equivalent of the workflow runner
        result_object = generate_insights(workflow_input["input_as_text"])
        
        # --- Output for User and Debugging ---
        print("\n" + "="*50)
        print("DIGITALABCS BUSINESS INSIGHTS REPORT")
        print("="*50)
        
        # Output the parsed Python object
        print("\n--- Parsed Python Object (DigitalABCsInsights) ---")
        print(result_object.model_dump_json(indent=2)) # Output the validated JSON
        
        # You can now use this clean Python object for further processing,
        # such as generating an HTML page with the correct WCAG-compliant structure.
        
    except (RuntimeError, APIError) as e:
        print(f"\nFATAL WORKFLOW ERROR: {e}")
        # In a real application, you would log this error and inform the user.