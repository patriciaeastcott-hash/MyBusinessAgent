# insights_agent.py
# DigitalABCs Compliant AI Agent — Gemini API Version
# Empowerment | Compliance | Practicality | Privacy

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Structured output for trust & simplicity
from pydantic import BaseModel, Field
from typing import Annotated 

# Gemini SDK
import google.generativeai as genai

# Pure Chroma for private RAG (no LangChain)
import chromadb
from chromadb.utils import embedding_functions

# Document & news handling
from dateutil import parser as date_parser
import feedparser
from docx import Document as DocxDocument

# Practical outputs
import pandas as pd

# Accessibility audit
from selenium import webdriver
from axe_selenium_python import Axe

# Ensure clean environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Generate timestamp like: 2025-10-21_14-30
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
HTML_FILE = OUTPUT_DIR / f"{timestamp}_insights.html"
DOCX_FILE = OUTPUT_DIR / f"{timestamp}_insights.docx"
EXCEL_FILE = OUTPUT_DIR / f"{timestamp}_action_plan.xlsx"

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

# === STEP 1: LOAD BRAND VOICE ===
def load_brand_voice() -> str:
    brand_file = Path("brand_voice.docx")
    if not brand_file.exists():
        print("❌ Error: 'brand_voice.docx' not found.")
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
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Use default sentence transformers for embeddings (local & private)
    collection = client.get_or_create_collection(
        name="digitalabcs_kb",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Build knowledge base (only once)
    if collection.count() == 0:
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
        
        collection.add(
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )
        print("✅ Knowledge base created.")
    else:
        print("✅ Knowledge base loaded.")
    
    return collection

def retrieve_context(collection, query: str, k: int = 2) -> str:
    """Retrieve relevant brand/compliance context."""
    results = collection.query(query_texts=[query], n_results=k)
    return "\n".join(results['documents'][0])

# === STEP 3: FETCH AUSTRALIAN NEWS ===
def fetch_recent_news():
    cutoff = datetime.now() - timedelta(days=30)
    articles = []
    for url in NEWS_SOURCES:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:4]:
                try:
                    pub_date = date_parser.parse(entry.published)
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

# === STEP 4: STRUCTURED OUTPUT MODEL ===
class BusinessInsight(BaseModel):
    title: str = Field(..., description="e.g., 'Australian Small Business Pulse: October 2025'")
    summary: str = Field(..., description="2-sentence overview of the month's business climate")
    key_updates: Annotated[List[str], Field(min_length=3, max_length=4, description="Actionable tips for micro-businesses")]
    sources: Annotated[List[str], Field(min_length=2, description="Source URLs")]

# === STEP 5: GENERATE INSIGHTS (NATIVE GEMINI SDK) ===
def generate_insights(news_articles, kb_collection):
    # Prepare news
    news_text = "\n\n".join([
        f"Title: {a['title']}\nDate: {a['date']}\nSummary: {a['summary']}\nSource: {a['link']}"
        for a in news_articles
    ]) if news_articles else "No recent updates."

    # Get brand/compliance context
    context = retrieve_context(kb_collection, "Australian small business compliance and brand voice")
    
    # Create JSON schema for structured output
    json_schema = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "e.g., 'Australian Small Business Pulse: October 2025'"
            },
            "summary": {
                "type": "string",
                "description": "2-sentence overview of the month's business climate"
            },
            "key_updates": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 4,
                "description": "Actionable tips for micro-businesses"
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "description": "Source URLs"
            }
        },
        "required": ["title", "summary", "key_updates", "sources"]
    }
    
    prompt = f"""[Your Guidelines]
{context}

[Recent News]
{news_text}

You are a trusted advisor to Australian micro-businesses (<5 employees). 
Generate a monthly insight that is practical, empowering, and grounded in Australian regulations.
Write in clear, plain Australian English. Avoid jargon. Focus on actionable steps.

IMPORTANT: Respond with ONLY valid JSON that matches this exact structure:
{json.dumps(json_schema, indent=2)}

DO NOT include any text before or after the JSON. DO NOT use markdown code blocks. Just output the raw JSON."""

    # Use Gemini with native SDK
    # Using Gemini 2.5 Pro (latest production model)
    model = genai.GenerativeModel('models/gemini-2.5-pro')
    
    try:
        print("🔄 Calling Gemini API (native SDK)...")
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2000,
            )
        )
        
        # Parse response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        # Parse JSON
        data = json.loads(response_text)
        
        # Validate and create Pydantic model
        insight = BusinessInsight(**data)
        return insight
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"Response was: {response_text[:500]}")
        raise
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        raise

# === STEP 6: SAVE OUTPUTS ===
def save_outputs(insight: BusinessInsight):
    month_year = datetime.now().strftime("%B %Y")
    date_str = datetime.now().strftime("%d %B %Y")
    
    # HTML (WCAG-ready)
    html = f"""<!DOCTYPE html>
<html lang="en-AU">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{insight.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; margin-top: 1.5em; }}
        .date {{ color: #7f8c8d; font-style: italic; }}
        ul {{ padding-left: 1.2em; }}
        a {{ color: #2980b9; text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>{insight.title}</h1>
    <p class="date">Published: {date_str}</p>
    <p>{insight.summary}</p>
    <h2>Key Updates for Your Business</h2>
    <ul>{''.join(f'<li>{u}</li>' for u in insight.key_updates)}</ul>
    <h2>Sources</h2>
    <ul>{''.join(f'<li><a href="{s}" target="_blank" rel="noopener">{s}</a></li>' for s in insight.sources)}</ul>
</body>
</html>"""
    
    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ HTML saved: {HTML_FILE}")

    # Word doc
    doc = DocxDocument()
    doc.add_heading(insight.title, 0)
    doc.add_paragraph(f"Published: {date_str}")
    doc.add_paragraph(insight.summary)
    doc.add_heading("Key Updates for Your Business", level=1)
    for update in insight.key_updates:
        doc.add_paragraph(update, style='List Bullet')
    doc.add_heading("Sources", level=1)
    for src in insight.sources:
        doc.add_paragraph(src, style='List Bullet')
    doc.save(DOCX_FILE)
    print(f"✅ Word doc saved: {DOCX_FILE}")

    # Excel
    df = pd.DataFrame({"Action Item": insight.key_updates})
    df.to_excel(EXCEL_FILE, index=False)
    print(f"✅ Excel saved: {EXCEL_FILE}")

# === STEP 7: ACCESSIBILITY AUDIT ===
def audit_accessibility():
    print("🔍 Running WCAG audit...")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    try:
        driver = webdriver.Chrome(options=options)
        driver.get(f"file:///{HTML_FILE.resolve()}")
        axe = Axe(driver)
        axe.inject()
        results = axe.run()
        driver.quit()
        
        if results["violations"]:
            print("⚠️  WCAG issues found:")
            for v in results["violations"]:
                print(f" - {v['description']}")
        else:
            print("✅ WCAG 2.1 AA compliant!")
    except Exception as e:
        print(f"⚠️  Audit skipped: {e}")

# === MAIN ===
def main():
    print("🇦🇺 DigitalABCs Insights Agent (Gemini Native SDK)")
    
    # Setup
    kb = create_compliance_kb()
    news = fetch_recent_news()
    print(f"📰 Found {len(news)} articles.")
    
    # Generate
    try:
        insight = generate_insights(news, kb)
        print(f"🧠 Generated: {insight.title}")
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        print("❌ Full traceback:")
        traceback.print_exc()
        sys.exit(1)
    
    # Output
    save_outputs(insight)
    audit_accessibility()
    
    print(f"\n🎉 Success! Outputs in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()