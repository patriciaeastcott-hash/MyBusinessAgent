# insights_agent.py
# DigitalABCs Compliant AI Agent — Gemini API Version
# Empowerment | Compliance | Practicality | Privacy

import os
# 🟢 --- FIX: Set RAYON_NUM_THREADS ---
# This MUST be set before chromadb is imported to prevent a
# ThreadPoolBuildError in low-resource environments.
os.environ["RAYON_NUM_THREADS"] = "1"

import sys
from pathlib import Path

# --- ▼▼▼ START: SERVICE ACCOUNT AUTHENTICATION ▼▼▼ ---
# This block MUST come before you import 'vertexai'
SCRIPT_DIR = Path(__file__).resolve().parent

# This should be the name of the JSON key file you uploaded
KEY_FILE_PATH = SCRIPT_DIR / "gen-lang-client-0944750405-a5911516ca25.json"

if not KEY_FILE_PATH.exists():
    print(f"❌ Error: Authentication key file not found at {KEY_FILE_PATH}")
    print("❌ Please download your Google Cloud service account JSON key,")
    print("❌ upload it to this directory, and ensure the filename matches.")
    sys.exit(1)
    
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(KEY_FILE_PATH)
# --- ▲▲▲ END: SERVICE ACCOUNT AUTHENTICATION ▲▲▲ ---


import json
from datetime import datetime, timedelta
from typing import List
import shutil  # To copy the final HTML file
import base64  # To create the placeholder image on failure

# 🟢 ADDED: Vertex AI Imports for Image Generation
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

# 🟢 FIX: Attempt to import orjson for robust JSON decoding
try:
    import orjson
    JSON_LOADER = orjson.loads
except ImportError:
    JSON_LOADER = json.loads


# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Structured output for trust & simplicity
from pydantic import BaseModel, Field
from typing import Annotated 

# Gemini SDK
import google.generativeai as genai
# 🐛 FIX: Import the types module for the correct safety_settings syntax
from google.generativeai import types 

# Pure Chroma for private RAG (no LangChain)
import chromadb
from chromadb.utils import embedding_functions

# Document & news handling
from dateutil import parser as date_parser
import feedparser
from docx import Document as DocxDocument

# Practical outputs
import pandas as pd
# 🟢 FIX: openpyxl is implicitly required by pandas.to_excel

# Accessibility audit
from selenium import webdriver
from axe_selenium_python import Axe

# Ensure clean environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 🟢 START: NEW PATH CONFIGURATION ---
# (SCRIPT_DIR is already defined at the top)

# Paths for local files (brand voice, vector DB)
BRAND_VOICE_FILE = SCRIPT_DIR / "brand_voice.docx"
CHROMA_DB_PATH = SCRIPT_DIR / "chroma_db"

# Path for your private ARCHIVES (docx, xlsx, and a copy of the html)
ARCHIVE_OUTPUT_DIR = Path("/home/digitala/my_scripts/agents/my_business_insights/outputs")
ARCHIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Path for your PUBLIC website
PUBLIC_HTML_DIR = Path("/home/digitala/public_html/insights")
PUBLIC_HTML_DIR.mkdir(parents=True, exist_ok=True)

# Generate timestamp like: 2025-10-21_14-30
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

# Define filenames
HTML_FILENAME = f"{timestamp}_insights.html"
DOCX_FILENAME = f"{timestamp}_insights.docx"
EXCEL_FILENAME = f"{timestamp}_action_plan.xlsx"

# Define full ARCHIVE paths
ARCHIVE_HTML_FILE = ARCHIVE_OUTPUT_DIR / HTML_FILENAME
ARCHIVE_DOCX_FILE = ARCHIVE_OUTPUT_DIR / DOCX_FILENAME
ARCHIVE_EXCEL_FILE = ARCHIVE_OUTPUT_DIR / EXCEL_FILENAME

# Define full PUBLIC path
PUBLIC_HTML_FILE = PUBLIC_HTML_DIR / HTML_FILENAME
# --- 🟢 END: NEW PATH CONFIGURATION ---


NEWS_SOURCES = [
    "https://www.smartcompany.com.au/feed/",
    "https://businessnewsaustralia.com/feed/",
    "https://asic.gov.au/about-asic/media-centre/news-releases/media-releases-2025/feed/",
    "https://ausbiz.com.au/feeds/rss",
    "https://www.itnews.com.au/rss",
    "https://www.crn.com.au/rss",
    "https://www.themandarin.com.au/feed/",
    "https://www.businessnewsaustralia.com/articles/rss/startup-daily",
    "https://www.business.gov.au/news.rss",
    "https://www.dynamicbusiness.com.au/feed",
    "https://www.smartcompany.com.au/feed/",
    "https://www.businessnewsaustralia.com/articles/rss/sme",
    "https://www.asbfeo.gov.au/news/rss.xml"
]

# Get Gemini API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("❌ Error: GEMINI_API_KEY not found in .env file")
    sys.exit(1)

# Configure Gemini
# --- ▼▼▼ START: REST TRANSPORT FIX ▼▼▼ ---
# This forces the library to use HTTPS, saving memory.
genai.configure(
    api_key=GEMINI_API_KEY,
    transport="rest"
)
# --- ▲▲▲ END: REST TRANSPORT FIX ▲▲▲ ---

# 2. Create the Google embedding function
# --- ▼▼▼ START: ERROR FIX ▼▼▼ ---
# Removed the 'transport="rest"' argument, as it's not supported
# by this specific chromadb wrapper.
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=GEMINI_API_KEY,
    model_name="models/text-embedding-004",
    task_type="RETRIEVAL_DOCUMENT"
)
# --- ▲▲▲ END: ERROR FIX ▲▲▲ ---

# === (NEW & MOVED) STEP 1.5: INITIALIZE VERTEX AI ===
try:
    # Your Google Cloud Project ID
    GCP_PROJECT_ID = "gen-lang-client-0944750405" 
    GCP_REGION = "australia-southeast1" # Use a supported region
    
    # Auth is handled by the GOOGLE_APPLICATION_CREDENTIALS env var we set at the top
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION)
    print(f"✅ Vertex AI initialized for project: {GCP_PROJECT_ID}")
    
except Exception as e:
    print(f"❌ FAILED to initialize Vertex AI: {e}")
    print("❌ Please ensure you have set your GCP_PROJECT_ID and that your")
    print(f"❌ JSON key file '{KEY_FILE_PATH.name}' is valid and has the 'Vertex AI User' role.")
    sys.exit(1)
# === END STEP 1.5 ===


# === STEP 1: LOAD BRAND VOICE ===
def load_brand_voice() -> str:
    brand_file = BRAND_VOICE_FILE
    if not brand_file.exists():
        print(f"❌ Error: '{BRAND_VOICE_FILE}' not found.")
        sys.exit(1)
    try:
        doc = DocxDocument(brand_file)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()]) or "You are a trusted Australian small business advisor."
    except Exception as e:
        print(f"❌ Error reading brand_voice.docx: {e}")
        sys.exit(1)

# === STEP 2: PRIVATE KNOWLEDGE BASE (PURE CHROMA) ===
def create_compliance_kb():
    """Create a local, private knowledge base using ChromaDB."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    
    collection = client.get_or_create_collection(
        name="compliance_kb",
        embedding_function=google_ef, # Uses the corrected function
        metadata={"hnsw:space": "cosine"}
    )
    
    # Build knowledge base (only once)
    if collection.count() == 0:
        print("🔄 Building new knowledge base... (This may take a moment for API embeddings)")
        brand_text = load_brand_voice()
        compliance_text = """
        Australian Privacy Principle 1: Open and transparent management of personal information.
        Disability Discrimination Act 1992: Digital services must be accessible.
        Small businesses must protect personal information from misuse.
        Always use plain Australian English. Avoid jargon. Focus on actionable steps.
        """
        all_text = brand_text + "\n\n" + compliance_text
        
        # Simple chunking
        chunks = [all_text[i:i+500] for i in range(0, len(all_text), 450)]
        
        try:
            collection.add(
                documents=chunks,
                ids=[f"chunk_{i}" for i in range(len(chunks))]
            )
            print("✅ Knowledge base created successfully via API.")
        except Exception as e:
            print(f"❌ API Error during collection.add(): {e}")
            print("❌ Please check your GEMINI_API_KEY and ensure the 'Generative Language API' is enabled in your Google Cloud project.")
            sys.exit(1)
    else:
        print("✅ Knowledge base loaded from disk.")
    
    return collection

def retrieve_context(collection, query: str, k: int = 2) -> str:
    """Retrieve relevant brand/compliance context."""
    results = collection.query(query_texts=[query], n_results=k)
    return "\n".join(results['documents'][0])

# === STEP 3: FETCH AUSTRALIAN NEWS (UPDATED FOR 7 DAYS) ===
def fetch_recent_news():
    cutoff = datetime.now() - timedelta(days=7) 
    articles = []
    for url in NEWS_SOURCES:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:4]:
                try:
                    pub_date_str = getattr(entry, 'published', None) or getattr(entry, 'updated', None)
                    if not pub_date_str:
                        continue
                    
                    pub_date = date_parser.parse(pub_date_str)
                    if pub_date.tzinfo:
                        pub_date = pub_date.replace(tzinfo=None)
                    if pub_date >= cutoff:
                        articles.append({
                            "title": entry.title,
                            "summary": entry.summary,
                            "link": entry.link,
                            "date": pub_date.strftime("%Y-%m-%d")
                        })
                except:
                    continue
        except Exception as e:
            print(f"⚠️  Warning: {url} - {e}")
            continue
    return articles[:6]

# === (NEW) STEP 3.5: TEXT-TO-IMAGE FUNCTION ===
# === (REVISED) STEP 3.5: TEXT-TO-IMAGE FUNCTION ===
def generate_ai_image(prompt: str, output_filepath: Path) -> bool:
    """
    Generates an image using the Gemini 2.5 Flash model via the genai.Client
    and saves it to a file.
    """
    try:
        print(f"🎨 Calling Gemini API for: {output_filepath.name}")
        print(f"   Prompt: {prompt}")

        # 1. Initialize the client (as per your example and saved context)
        client = genai.Client(
            vertexai=True,
            # Uses the API key from your environment, as per your saved instructions
            api_key=os.environ.get("GEMINI_API_KEY"), 
        )

        # 2. Define the model
        model = "gemini-2.5-flash-image" 

        # 3. Create the content (TEXT ONLY for text-to-image)
        text_part = types.Part.from_text(prompt)
        contents = [
            types.Content(
                role="user",
                parts=[text_part]
            )
        ]

        # 4. Configure the request
        generate_content_config = types.GenerateContentConfig(
            temperature = 0.8, # Lowered for more predictable image output
            top_p = 0.9,
            response_modalities = ["IMAGE"], # CRITICAL: Ask for an image back
            safety_settings = [ # Copied from your example
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
            ],
        )

        # 5. Call the API (using non-streaming for a single image)
        response = client.models.generate_content(
            model = model,
            contents = contents,
            config = generate_content_config,
        )

        # 6. Check for and extract the image data
        if not response.parts or not response.parts[0].image:
             raise ValueError("API response did not contain image data.")

        # Get the raw image bytes
        image_data = response.parts[0].image.data 

        # 7. Save the image bytes to the file
        with open(output_filepath, "wb") as f:
            f.write(image_data)
            
        print(f"✅ Image saved: {output_filepath}")
        return True
        
    except Exception as e:
        print(f"❌ Vertex AI Image Generation FAILED: {e}")
        # --- FALLBACK: Create placeholder ---
        try:
            print("   Falling back to placeholder image...")
            placeholder_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAA+gAAAJYCAIAAABCN03IAAAABGdBTUEAALGPC/xhBQAAADhlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAAD6KADAAQAAAABAAACYwAAAAAAvVMvSwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAlNJREFUeF7t0AEBAAAAIKiu+P+nBwcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADYB03cAAFq0LTIAAAAASUVORK5CYII="
            
            with open(output_filepath, "wb") as f:
                f.write(base64.b64decode(placeholder_png_b64))
            print(f"✅ Placeholder image saved: {output_filepath}")
            
        except Exception as e2:
            print(f"❌ Placeholder save FAILED: {e2}")
            pass 
        return False
# === END STEP 3.5 ===


# === (NEW) STEP 4: STRUCTURED OUTPUT MODEL ===
class BusinessInsight(BaseModel):
    title: str = Field(..., description="e.g., 'Australian Small Business Weekly: 21 Oct - 28 Oct 2025'")
    summary: str = Field(..., description="3-4 paragraphs (no bullet points) focusing on THIS WEEK's specific developments, framed around AI, automation, and immediate opportunities for micro-businesses.")
    key_updates: Annotated[List[str], Field(min_length=3, max_length=4, description="Specific actionable, time-sensitive tips for micro-businesses, written in plain language, emphasizing AI, automation, or software agents based on THIS WEEK's news.")]
    
    # --- NEW FIELDS ---
    image_generation_prompt: str = Field(..., description="A detailed, descriptive prompt for an AI image generator (like Imagen) to create a conceptual image for this article. Style: clean, modern, approachable, inclusive, soft shadows. e.g., 'A medium shot of a calm, neurodivergent founder (woman, 40s, glasses) in a simple home office, smiling at a clean Trello board on her laptop.'")
    image_alt_text: str = Field(..., description="A detailed, WCAG 2.1 AA compliant alt text for the image described in the prompt. This text should describe the scene for someone who cannot see it. e.g., 'A focused small business owner sits at a clean desk, smiling at their laptop.'")
    
    sources: Annotated[List[str], Field(min_length=2, description="Source URLs")]
# === END STEP 4 ===

# === (NEW) STEP 5: GENERATE INSIGHTS (DUAL API CALL) ===
def generate_insights(news_articles, kb_collection):
    """
    Generates structured insights using Gemini 1.5 Pro and tool-based
    Pydantic output.
    """
    # Prepare news
    news_text = "\n\n".join([
        f"Title: {a['title']}\nDate: {a['date']}\nSummary: {a['summary']}\nSource: {a['link']}"
        for a in news_articles
    ]) if news_articles else "No recent updates."

    # Get brand/compliance context
    context = retrieve_context(kb_collection, "Australian small business compliance and brand voice")
    
# 🟢 --- START FIX: Manually define the schema ---
    # The Pydantic 'model_json_schema()' generates fields (like 'title')
    # that the 'google-generativeai' library's 'Schema' object does not recognize.
    # We must define a schema using only the allowed fields.
    json_schema = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "e.g., 'Australian Small Business Weekly: 21 Oct - 28 Oct 2025'"
            },
            "summary": {
                "type": "string",
                "description": "3-4 paragraphs (no bullet points) focusing on THIS WEEK's specific developments, framed around AI, automation, and immediate opportunities for micro-businesses."
            },
            "key_updates": {
                "type": "array",
                "description": "Specific actionable, time-sensitive tips for micro-businesses, written in plain language, emphasizing AI, automation, or software agents based on THIS WEEK's news.",
                "items": {
                    "type": "string"
                }
            },
            "image_generation_prompt": {
                "type": "string",
                "description": "A detailed, descriptive prompt for an AI image generator (like Imagen) to create a conceptual image for this article. Style: clean, modern, approachable, inclusive, soft shadows. e.g., 'A medium shot of a calm, neurodivergent founder (woman, 40s, glasses) in a simple home office, smiling at a clean Trello board on her laptop.'"
            },
            "image_alt_text": {
                "type": "string",
                "description": "A detailed, WCAG 2.1 AA compliant alt text for the image described in the prompt. This text should describe the scene for someone who cannot see it. e.g., 'A focused small business owner sits at a clean desk, smiling at their laptop.'"
            },
            "sources": {
                "type": "array",
                "description": "Source URLs",
                "items": {
                    "type": "string"
                }
            }
        },
        "required": ["title", "summary", "key_updates", "image_generation_prompt", "image_alt_text", "sources"]
    }
    # 🟢 --- END FIX ---
    
    # We will pass the Pydantic model directly.

    hook_and_cta_text = (
        "Don't let the technical talk drown you out! Just like Trish fought to find her feet, "
        "DigitalABCs is here to simplify these Key Updates. We show you exactly how to use "
        "simple agents and automation to turn this week's challenges into your next big opportunity."
    )

    # 🟢 --- START FIX 2: UPDATE GENERATION & TOOL CONFIG ---
    # We remove 'response_mime_type'
    generation_config = genai.types.GenerationConfig(
        temperature=0.7,
        max_output_tokens=4096,
        response_mime_type="application/json",
        response_schema=json_schema  # <-- This is the correct way
    )
    
    # 🟢 --- START FIX 1: SIMPLIFY PROMPT ---
    # We remove ALL instructions about JSON, as the 'tools' config handles it.
    prompt = f"""You are a trusted advisor to Australian micro-businesses (<5 employees) for DigitalABCs.

[Your Guidelines]
{context}
- Your voice must be relatable, supportive, and empowering (Brand Voice: 'The Chaos-to-Clarity Architect').
- Frame all insights through the lens of AI, automation, and 'asynchronous' systems.
- Use 'Lived Experience' and 'Plain Language' tones.

[This Week's News - Last 7 Days]
{news_text}

IMPORTANT CONTEXT: This is a WEEKLY briefing. Focus ONLY on developments from the past 7 days.

Your task: Analyze THIS WEEK's specific news and identify:
1.  New developments that happened THIS WEEK
2.  Emerging trends visible in this week's news cycle
3.  Time-sensitive opportunities or risks micro-businesses should act on NOW
4.  A detailed image generation prompt (for Imagen) matching the brand's 'approachable, inclusive, clean, soft shadow' aesthetic.
5.  A descriptive, WCAG-compliant alt text for that conceptual image.

CRITICAL: Be specific and timely. Avoid generic advice. Reference actual news from this week. 
Now, generate the business insight."""
    # 🟢 --- END FIX 1 ---

    safety_settings = {
        types.HarmCategory.HARM_CATEGORY_HARASSMENT: types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }


    
#    # We ADD a 'tool_config' to force the model to use our Pydantic class
#    tool_config = genai.ToolConfig(
#        function_calling_config=genai.FunctionCallingConfig(
#            # Mode.ANY is fine, but we restrict it to ONLY our class
#            mode=genai.FunctionCallingConfig.Mode.ANY,
#            allowed_function_names=["BusinessInsight"] 
#        )
#    )
    # 🟢 --- END FIX 2 ---


    # 🟢 --- START FIX 3: PASS PYDANTIC CLASS TO MODEL ---
    model = genai.GenerativeModel(
        model_name='models/gemini-2.5-pro',
        #tools=[BusinessInsight]  # <-- This is the key change
    )
    # 🟢 --- END FIX 3 ---
    
    try:
        print("🔄 Calling Gemini API (text)...")
        
        # 1. REMOVE 'tool_config' from the call
        response = model.generate_content(
            prompt,
            generation_config=generation_config, 
            safety_settings=safety_settings
            # tool_config=tool_config  <-- REMOVED
        )

        # 2. USE PARSING LOGIC FOR JSON MODE (response.text)
        if not response.candidates or not response.candidates[0].content.parts:
            finish_reason = response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN"
            raise ValueError(
                f"Model blocked generation. Finish Reason: {finish_reason}. "
                "Review safety settings or simplify the prompt."
            )
        
        response_text = response.text.strip()
        
        # Handle potential markdown ```json ... ``` wrapper
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        # Parse the plain JSON text
        data = JSON_LOADER(response_text)
        insight = BusinessInsight(**data)
        # 🟢 --- END FIX 2 ---
        
        # --- NEW: GENERATE IMAGE ---
        # (This part of your code was correct)
        image_filename = HTML_FILENAME.replace('.html', '.jpg')
        PUBLIC_IMAGE_FILE = PUBLIC_HTML_DIR / image_filename
        
        generate_ai_image(
            prompt=insight.image_generation_prompt,
            output_filepath=PUBLIC_IMAGE_FILE
        )
        # --- END IMAGE GENERATION ---
        
        return insight, hook_and_cta_text
        
    except json.JSONDecodeError as e:
        # This error is now relevant again
        print(f"❌ JSON parsing error: {e}")
        try:
            print(f"Response snippet: {response.text[:500]}")
        except:
            pass
        raise
    except Exception as e:
        if isinstance(e, ValueError) and "Model blocked generation" in str(e):
            raise
        print(f"❌ Gemini API error: {e}")
        raise
# === END STEP 5 ===


# === (NEW) STEP 6: SAVE OUTPUTS (FULL TEMPLATE) ===
def save_outputs(insight: BusinessInsight, hook_and_cta_text: str):
    date_str = datetime.now().strftime("%d %B %Y")
    
    # Define filenames for image and HTML
    image_filename = HTML_FILENAME.replace('.html', '.jpg') # e.g., "2025-10-28_15-00_insights.jpg"

    # --- NEW: FULL HTML PAGE TEMPLATE ---
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{insight.title}</title>
    
    <meta name="description" content="{insight.summary[:160]}...">
    <meta property="og:title" content="{insight.title}">
    <meta property="og:description" content="{insight.summary[:160]}...">
    <meta property="og:type" content="article">
    <meta property="og:image" content="{image_filename}">

    <meta name="image-filename" content="{image_filename}">
    <meta name="image-alt" content="{insight.image_alt_text}">

    <link rel="icon" href="../assets/favicon.ico" type="image/x-icon">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&family=Roboto+Mono&display=swap" rel="stylesheet">
    
    <link rel="stylesheet" href="../style.css">

    <style>
        .main-article-image {{
            width: 100%;
            height: auto;
            max-height: 400px;
            object-fit: cover;
            border-radius: var(--border-radius);
            margin-bottom: 2rem;
            border: 1px solid #E5E7EB;
        }}
        .blog-post-content p {{ margin-bottom: 1.5rem; }}
        .blog-post-content ul {{ list-style-type: disc; padding-left: 1.5rem; margin-bottom: 1.5rem; }}
        .blog-post-content li {{ margin-bottom: 0.5rem; }}
    </style>
</head>
<body>
    <a href="#main-content" class="skip-link">Skip to main content</a>
    
    <header class="site-header">
        <div class="container">
            <a href="../index.html" class="logo" aria-label="DigitalABCs Home">
                <img src="../assets/logo.png" alt="DigitalABCs Logo" class="logo-img">
            </a>
            <button class="nav-toggle" aria-label="Open navigation menu" aria-expanded="false" aria-controls="main-nav-menu">
                <span class="hamburger-line"></span>
                <span class="hamburger-line"></span>
                <span class="hamburger-line"></span>
            </button>
            <nav class="main-nav" id="main-nav-menu">
                <ul>
                    <li><a href="../index.html">Home</a></li>
                    <li><a href="../about.html">About</a></li>
                    <li><a href="../services.html">Services</a></li>
                    <li><a href="insights.php" class="active">Insights</a></li>
                    <li><a href="../contact.html">Contact</a></li>
                    <li><a href="https://services.digitalabcs.com.au" target="_blank">Client Login</a></li>
                </ul>
            </nav>
        </div>
    </header>

    <main id="main-content">
        <section class="hero-small">
            <div class="container">
                <h1>{insight.title}</h1>
                <p class="subtitle">Published: {date_str}</p>
            </div>
        </section>

        <section class="insights-grid section-padding">
            <div class="container" style="max-width: 800px;">
                <article class="blog-post-content">
                    <img src="{image_filename}" alt="{insight.image_alt_text}" class="main-article-image">
                    
                    <h2>The Big Picture: AI, Automation, and Your Power</h2>
                    <p style="white-space: pre-wrap;">{insight.summary}</p>
                    
                    <h2>Your Action Plan: Practical AI & Automation Takeaways</h2>
                    <ul>{''.join(f'<li><strong>Time-Sensitive Action:</strong> {u}</li>' for u in insight.key_updates)}</ul>
                    
                    <div class="cta-box" style="border-color: var(--color-green-cta);">
                        <h2>Ready to Take Back Control?</h2>
                        <p class="cta-text" style="color: var(--color-green-cta); font-weight: bold;">{hook_and_cta_text}</p>
                        <button class="cta" style="background-color: var(--color-green-cta); color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-weight: bold; margin-top: 10px;" onclick="window.open('/contact.html', '_self')">Start Simplifying Your Business</button>
                    </div>
                    
                    <h2 style="margin-top: 2rem;">Sources</h2>
                    <ul>{''.join(f'<li><a href="{s}" target="_blank" rel="noopener noreferrer">{s}</a></li>' for s in insight.sources)}</ul>
                </article>
            </div>
        </section>
    </main>

    <footer class="site-footer">
        <div class="container">
            <div class="footer-grid">
                <div class="footer-col">
                    <h4>DigitalABCs</h4>
                    <p>Empowering Australian businesses through practical technology. Built on resilience, for resilience.</p>
                </div>
                <div class="footer-col">
                    <h4>Navigate</h4>
                    <ul>
                        <li><a href="../about.html">About Us</a></li>
                        <li><a href="../services.html">Our Services</a></li>
                        <li><a href="insights.php">Insights</a></li>
                        <li><a href="../contact.html">Contact</a></li>
                    </ul>
                </div>
                <div class="footer-col">
                    <h4>Legal</h4>
                    <ul>
                        <li><a href="../privacy.html">Privacy Policy</a></li>
                        <li><a href="../terms.html">Terms of Service</a></li>
                    </ul>
                </div>
                <div class="footer-col">
                    <h4>Connect</h4>
                    <p>info@digitalabcs.com.au<br>Toongabbie, NSW, Australia</p>
                </div>
            </div>
            <div class="footer-bottom">
                <p>&copy; 2025 DigitalABCs. All rights reserved.</p>
            </div>
        </div>
    </footer>

    <script>
        document.addEventListener('DOMContentLoaded', function () {{
            const navToggle = document.querySelector('.nav-toggle');
            const mainNav = document.querySelector('.main-nav');
            if (navToggle && mainNav) {{
                navToggle.addEventListener('click', function () {{
                    const isActive = mainNav.classList.toggle('is-active');
                    navToggle.setAttribute('aria-expanded', isActive);
                    navToggle.setAttribute('aria-label', isActive ? 'Close navigation menu' : 'Open navigation menu');
                }});
            }}
        }});
    </script>
</body>
</html>"""
    
    # --- Save HTML file ---
    with open(ARCHIVE_HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ HTML archived: {ARCHIVE_HTML_FILE}")

    # Copy the HTML file to the PUBLIC web directory
    try:
        shutil.copyfile(ARCHIVE_HTML_FILE, PUBLIC_HTML_FILE)
        print(f"✅ HTML published: {PUBLIC_HTML_FILE}")
    except Exception as e:
        print(f"❌ FAILED to copy HTML to public directory: {e}")

    # --- Save Word Doc ---
    doc = DocxDocument()
    doc.add_heading(insight.title, 0)
    doc.add_paragraph(f"Published: {date_str}")
    
    doc.add_heading("Image Generation Prompt", level=2)
    doc.add_paragraph(insight.image_generation_prompt)
    doc.add_heading("Image Alt Text (WCAG)", level=2)
    doc.add_paragraph(insight.image_alt_text)
    
    doc.add_heading("The Big Picture: AI, Automation, and Your Power", level=1)
    doc.add_paragraph(insight.summary)
    doc.add_heading("Your Action Plan: Practical AI & Automation Takeaways", level=1)
    for update in insight.key_updates:
        doc.add_paragraph(f"Time-Sensitive Action: {update}", style='List Bullet')
    doc.add_heading("Ready to Take Back Control?", level=1)
    doc.add_paragraph(hook_and_cta_text)
    doc.add_heading("Sources", level=1)
    for src in insight.sources:
        doc.add_paragraph(src, style='List Bullet')
    doc.save(ARCHIVE_DOCX_FILE)
    print(f"✅ Word doc saved: {ARCHIVE_DOCX_FILE}")

    # --- Save Excel Sheet ---
    df = pd.DataFrame({
        "AI/Automation Action Item": insight.key_updates,
        "Image Prompt": [insight.image_generation_prompt] + [""] * (len(insight.key_updates) - 1),
        "Alt Text": [insight.image_alt_text] + [""] * (len(insight.key_updates) - 1)
    })
    df.to_excel(ARCHIVE_EXCEL_FILE, index=False)
    print(f"✅ Excel saved: {ARCHIVE_EXCEL_FILE}")
# === END STEP 6 ===


# === STEP 7: ACCESSIBILITY AUDIT ===
def audit_accessibility():
    print("🔍 Running WCAG audit...")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    try:
        driver = webdriver.Chrome(options=options)
        driver.get(f"file:///{ARCHIVE_HTML_FILE.resolve()}")
        axe = Axe(driver)
        axe.inject()
        results = axe.run(options={'runOnly': {'type': 'tag', 'values': ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']}})
        driver.quit()
        
        if results["violations"]:
            print("⚠️  WCAG issues found:")
            for v in results["violations"]:
                print(f" - {v['description']}")
                print(f"   Affected nodes: {[n['html'] for n in v['nodes'][:2]]}...")
        else:
            print("✅ WCAG 2.1 AA compliant!")
    except Exception as e:
        print(f"⚠️  Accessibility Audit skipped (WebDriver issue in environment?): {e}")

# === MAIN ===
def main():
    print("🇦🇺 DigitalABCs Insights Agent (Gemini Native SDK)")
    
    # Setup
    kb = create_compliance_kb()
    news = fetch_recent_news()
    print(f"📰 Found {len(news)} articles in the last 7 days.")
    
    # Generate
    try:
        insight, hook_text = generate_insights(news, kb)
        print(f"🧠 Generated: {insight.title}")
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        print("❌ Full traceback:")
        traceback.print_exc()
        sys.exit(1)
    
    # Output
    save_outputs(insight, hook_text)
    #audit_accessibility()
    
    print(f"\n🎉 Success! Outputs archived in: {ARCHIVE_OUTPUT_DIR.resolve()}")
    print(f"🎉 Public HTML published to: {PUBLIC_HTML_FILE.resolve()}")


if __name__ == "__main__":
    main()