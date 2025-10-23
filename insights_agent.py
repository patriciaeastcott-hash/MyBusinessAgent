# insights_agent.py
# DigitalABCs Compliant AI Agent — Gemini API Version
# Empowerment | Compliance | Practicality | Privacy

import os
# 🟢 --- FIX: Set RAYON_NUM_THREADS ---
# This MUST be set before chromadb is imported to prevent a
# ThreadPoolBuildError in low-resource environments.
os.environ["RAYON_NUM_THREADS"] = "1"

import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import shutil  # 🟢 ADDED: To copy the final HTML file

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
# 🟢 FIX: openpyxl is implicitly required by pandas.to_excel, but not directly imported. 
# We need to ensure it's in the installation steps.

# Accessibility audit
from selenium import webdriver
from axe_selenium_python import Axe

# Ensure clean environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 🟢 START: NEW PATH CONFIGURATION ---
# This makes the script work from any location (like a cron job)
# SCRIPT_DIR is the absolute path to this script: /home/digitala/my_scripts/agents/my_business_insights
SCRIPT_DIR = Path(__file__).resolve().parent

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
genai.configure(api_key=GEMINI_API_KEY)

# 2. Create the Google embedding function
#    We use "models/embedding-001" which is Google's embedding model
# 🟢 --- FIX: Explicitly set model to text-embedding-004 ---
# This model is more likely to be enabled on your API key
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=GEMINI_API_KEY,
    model_name="models/text-embedding-004" 
)

# === STEP 1: LOAD BRAND VOICE ===
def load_brand_voice() -> str:
    # 🟢 CHANGED: Use absolute path
    brand_file = BRAND_VOICE_FILE
    if not brand_file.exists():
        # 🟢 CHANGED: Use absolute path in error
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
    """Create a local, private knowledge base using ChromaDB with default embeddings."""
    # 🟢 CHANGED: Use absolute path (and convert Path object to string for chromadb)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    
    # Use default sentence transformers for embeddings (local & private)
    # 3. When you create your collection, TELL Chroma to use this function
    #    (Instead of its default, memory-hungry one)
    
    collection = client.get_or_create_collection(
        name="compliance_kb",
        embedding_function=google_ef,
        # this is the key change
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
    # ⚙️ CHANGE: Set cutoff to 7 days for the weekly report
    cutoff = datetime.now() - timedelta(days=7) 
    articles = []
    for url in NEWS_SOURCES:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:4]:
                try:
                    # Handle potentially missing or poorly formatted dates
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

# === STEP 4: STRUCTURED OUTPUT MODEL (UPDATED for Weekly Prompt) ===
class BusinessInsight(BaseModel):
    # Adjusted field name to match the new prompt output structure
    title: str = Field(..., description="e.g., 'Australian Small Business Weekly: 21 Oct - 28 Oct 2025'")
    # Increased summary length and updated focus for weekly report
    summary: str = Field(..., description="3-4 paragraphs (no bullet points) focusing on THIS WEEK's specific developments, framed around AI, automation, and immediate opportunities for micro-businesses.")
    # Adjusted field name to match the new prompt output structure
    key_updates: Annotated[List[str], Field(min_length=3, max_length=4, description="Specific actionable, time-sensitive tips for micro-businesses, written in plain language, emphasizing AI, automation, or software agents based on THIS WEEK's news.")]
    sources: Annotated[List[str], Field(min_length=2, description="Source URLs")]

# === STEP 5: GENERATE INSIGHTS (NATIVE GEMINI SDK) (FIXED) ===
def generate_insights(news_articles, kb_collection):
    # Prepare news
    news_text = "\n\n".join([
        f"Title: {a['title']}\nDate: {a['date']}\nSummary: {a['summary']}\nSource: {a['link']}"
        for a in news_articles
    ]) if news_articles else "No recent updates."

    # Get brand/compliance context
    context = retrieve_context(kb_collection, "Australian small business compliance and brand voice")
    
    # Create JSON schema from Pydantic model
    schema_data = BusinessInsight.model_json_schema()
    json_schema = {
        "type": "object",
        "properties": {
            "title": schema_data["properties"]["title"],
            "summary": schema_data["properties"]["summary"],
            "key_updates": schema_data["properties"]["key_updates"],
            "sources": schema_data["properties"]["sources"]
        },
        "required": ["title", "summary", "key_updates", "sources"]
    }
    
    # Define the content of the emotional hook/CTA which is now inserted into the HTML, 
    # not generated by the model (as the model's output schema changed).
    hook_and_cta_text = (
        "Don't let the technical talk drown you out! Just like Trish fought to find her feet, "
        "DigitalABCs is here to simplify these Key Updates. We show you exactly how to use "
        "simple agents and automation to turn this week's challenges into your next big opportunity."
    )


    prompt = f"""You are a trusted advisor to Australian micro-businesses (<5 employees) for DigitalABCs.

[Your Guidelines]
{context}
- Your voice must be relatable, supportive, and empowering.
- Crucially, frame all insights and key takeaways through the lens of AI, automation, and the use of simple software agents to simplify life and give power back to employees.

[This Week's News - Last 7 Days]
{news_text}

IMPORTANT CONTEXT: This is a WEEKLY briefing. Focus ONLY on developments from the past 7 days.

Your task: Analyze THIS WEEK's specific news and identify:
1. New developments that happened THIS WEEK
2. Emerging trends visible in this week's news cycle
3. Time-sensitive opportunities or risks micro-businesses should act on NOW

Respond with ONLY valid JSON that matches this exact structure:
{json.dumps(json_schema, indent=2)}

CRITICAL: Be specific and timely. Avoid generic advice. Reference actual news from this week. 
**RIGOROUSLY ENSURE all strings within the JSON are valid: escape all double quotes (\") and newlines (\n).**
DO NOT include any text before or after the JSON. Just output the raw JSON.""" # 🟢 FIX: Added strict JSON prompt instruction

    # Define Safety Settings using the recommended moderate block threshold 
    # (BLOCK_NONE was causing the safety filter to reject the request, Finish Reason 2)
    # 🟢 FIX: Changed BLOCK_NONE to BLOCK_MEDIUM_AND_ABOVE for stability
    safety_settings = {
        types.HarmCategory.HARM_CATEGORY_HARASSMENT: types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    # Create Generation Config 
    # 🟢 FIX: Increased max_output_tokens to prevent truncation of the long summary string
    generation_config = genai.types.GenerationConfig(
        temperature=0.7,
        max_output_tokens=4096, # Increased from 2000
        response_mime_type="application/json"
    )

    # Use Gemini with native JSON mode
    model = genai.GenerativeModel(
        'models/gemini-2.5-pro',
    )
    
    try:
        print("🔄 Calling Gemini API with JSON mode...")
        response = model.generate_content(
            prompt,
            generation_config=generation_config, 
            safety_settings=safety_settings
        )

        # 🟢 FIX: Check the candidate's content existence before accessing .text
        if not response.candidates or not response.candidates[0].content.parts:
            # Get the finish reason if available
            finish_reason = response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN"
            raise ValueError(
                f"Model blocked generation. Finish Reason: {finish_reason}. "
                "Review safety settings or simplify the prompt."
            )
        
        # 🟢 FIX: Use the robust JSON_LOADER imported at the top of the file
        response_text = response.text.strip()
        # Clean up any stray markdown blocks just in case
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        data = JSON_LOADER(response_text)
        insight = BusinessInsight(**data)
        return insight, hook_and_cta_text # Return hook text separately for HTML rendering
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        # 🟢 FIX: Ensure response_text is assigned before the raise block in case of parsing failure
        try:
            print(f"Response snippet: {response.text[:500]}")
        except:
            pass
        raise
    except Exception as e:
        # Catch the new ValueError we introduced for blocked responses
        if isinstance(e, ValueError) and "Model blocked generation" in str(e):
            raise
        print(f"❌ Gemini API error: {e}")
        raise

# === STEP 6: SAVE OUTPUTS (UPDATED) ===
def save_outputs(insight: BusinessInsight, hook_and_cta_text: str):
    date_str = datetime.now().strftime("%d %B %Y")
    
    # HTML (WCAG-ready)
    html = f"""<!DOCTYPE html>
<html lang="en-AU">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{insight.title}</title>
    <style>
        /* DigitalABCs Brand Style & WCAG Compliance */
        :root {{
            --color-navy: #1E3A8A; 
            --color-purple: #7C3AED; 
            --color-green-cta: #10B981; 
            --color-text: #1F2937; /* Darker grey for better contrast */
        }}
        body {{ font-family: Inter, Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; color: var(--color-text); background-color: white; }}
        h1 {{ color: var(--color-navy); border-bottom: 2px solid var(--color-purple); padding-bottom: 0.2em; }}
        h2 {{ color: var(--color-purple); margin-top: 1.5em; }}
        .date {{ color: #6B7280; font-style: italic; }}
        ul {{ padding-left: 1.2em; }}
        a {{ color: var(--color-purple); text-decoration: underline; }}
        .cta-box {{ border: 1px solid var(--color-green-cta); background-color: #ecfdf5; padding: 15px; margin-top: 20px; border-radius: 8px; }}
        .cta-text {{ font-weight: bold; color: var(--color-green-cta); }}
        button.cta {{ background-color: var(--color-green-cta); color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-weight: bold; margin-top: 10px; transition: background-color: 0.3s; }}
        button.cta:hover {{ background-color: #0c9b6f; }}
        /* Ensure paragraphs (summary) have line breaks for readability */
        .summary-text {{ white-space: pre-wrap; }}
    </style>
</head>
<body>
    <h1>{insight.title}</h1>
    <p class="date">Published: {date_str}</p>
    
    <h2>The Big Picture: AI, Automation, and Your Power</h2>
    <p class="summary-text">{insight.summary}</p>
    
    <h2>Your Action Plan: Practical AI & Automation Takeaways</h2>
    <ul>{''.join(f'<li><strong>Time-Sensitive Action:</strong> {u}</li>' for u in insight.key_updates)}</ul>
    
    <div class="cta-box">
        <h2>Ready to Take Back Control?</h2>
        <p class="cta-text">{hook_and_cta_text}</p>
        <button class="cta" onclick="window.open('/start-your-automation-journey', '_self')">Start Simplifying Your Business Today</button>
    </div>
    
    <h2>Sources</h2>
    <ul>{''.join(f'<li><a href="{s}" target="_blank" rel="noopener noreferrer">{s}</a></li>' for s in insight.sources)}</ul>
</body>
</html>"""
    
    # 🟢 CHANGED: Save to ARCHIVE path
    with open(ARCHIVE_HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ HTML archived: {ARCHIVE_HTML_FILE}")

    # 🟢 ADDED: Copy the HTML file to the PUBLIC web directory
    try:
        shutil.copyfile(ARCHIVE_HTML_FILE, PUBLIC_HTML_FILE)
        print(f"✅ HTML published: {PUBLIC_HTML_FILE}")
    except Exception as e:
        print(f"❌ FAILED to copy HTML to public directory: {e}")


    # 🟢 CHANGED: Save to ARCHIVE path
    doc = DocxDocument()
    doc.add_heading(insight.title, 0)
    doc.add_paragraph(f"Published: {date_str}")
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

    # 🟢 CHANGED: Save to ARCHIVE path
    df = pd.DataFrame({"AI/Automation Action Item (Weekly Focus)": insight.key_updates})
    df.to_excel(ARCHIVE_EXCEL_FILE, index=False)
    print(f"✅ Excel saved: {ARCHIVE_EXCEL_FILE}")

# === STEP 7: ACCESSIBILITY AUDIT ===
def audit_accessibility():
    print("🔍 Running WCAG audit...")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    try:
        driver = webdriver.Chrome(options=options)
        # 🟢 CHANGED: Audit the ARCHIVED file
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
        # The generate_insights function now returns the insight object AND the static hook text
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
    
    # 🟢 CHANGED: Update success message
    print(f"\n🎉 Success! Outputs archived in: {ARCHIVE_OUTPUT_DIR.resolve()}")
    print(f"🎉 Public HTML published to: {PUBLIC_HTML_FILE.resolve()}")


if __name__ == "__main__":
    main()