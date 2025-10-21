# insights_agent.py
# DigitalABCs Compliant AI Agent — Pure Local Stack
# Empowerment | Compliance | Practicality | Privacy

import os
import sys
import contextlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from openai import OpenAI
from ollama import Client as OllamaClient
import json
from ollama import chat

import chromadb
from chromadb.utils import embedding_functions

from dateutil import parser as date_parser
import feedparser
from docx import Document as DocxDocument
import pandas as pd

from selenium import webdriver
from axe_selenium_python import Axe

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === CONFIG ===
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
HTML_FILE = OUTPUT_DIR / "insights.html"
DOCX_FILE = OUTPUT_DIR / "insights.docx"
EXCEL_FILE = OUTPUT_DIR / "action_plan.xlsx"

NEWS_SOURCES = [
    "https://www.industry.gov.au/contact-us/rss-feeds",
    "https://insidesmallbusiness.com.au/category/technology",
    "https://business.gov.au/news",
    "https://businessnewsaustralia.com/feed/",
    "https://asic.gov.au/about-asic/media-centre/news-releases/media-releases-2025/feed/",
    "https://itbrief.com.au/feed"
]

# === BRAND VOICE ===
def load_brand_voice() -> str:
    brand_file = Path("brand_voice.docx")
    if not brand_file.exists():
        raise FileNotFoundError("Missing 'brand_voice.docx'. Please place it in the working directory.")
    doc = DocxDocument(brand_file)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return text or "You are a trusted Australian small business advisor."

# === KNOWLEDGE BASE ===
def create_compliance_kb():
    client = chromadb.PersistentClient(path="./chroma_db")

    embedding_fn = embedding_functions.OllamaEmbeddingFunction(
        model_name="nomic-embed-text",
        url="http://localhost:11434/api/embed"  # updated endpoint
    )

    collection = client.get_or_create_collection(
        name="digitalabcs_kb",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    if collection.count() == 0:
        brand_text = load_brand_voice()
        compliance_text = (
            "Australian Privacy Principle 1: Open and transparent management of personal information.\n"
            "Disability Discrimination Act 1992: Digital services must be accessible.\n"
            "Small businesses must protect personal information from misuse.\n"
            "Always use plain Australian English. Avoid jargon. Focus on actionable steps."
        )
        combined = brand_text + "\n\n" + compliance_text
        chunks = [combined[i:i+500] for i in range(0, len(combined), 450)]
        collection.add(documents=chunks, ids=[f"chunk_{i}" for i in range(len(chunks))])
        print("✅ Knowledge base created.")
    else:
        print("✅ Knowledge base loaded.")
    return collection

def retrieve_context(collection, query: str, k: int = 2) -> str:
    results = collection.query(query_texts=[query], n_results=k)
    return "\n".join(results["documents"][0])

# === NEWS ===
def fetch_recent_news():
    cutoff = datetime.now() - timedelta(days=30)
    articles = []
    for url in NEWS_SOURCES:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:4]:
                pub_date_str = entry.get("published", None)
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
        except Exception as e:
            print(f"⚠️ Warning parsing {url}: {e}")
    return articles[:6]

# === MODEL ===
class BusinessInsight(BaseModel):
    title: str = Field(..., description="e.g., 'Australian Small Business Pulse: October 2025'")
    summary: str = Field(..., description="2-sentence overview of the month's business climate")
    key_updates: List[str] = Field(..., min_length=3, max_length=4, description="Actionable tips for micro-businesses")
    sources: List[str] = Field(..., min_length=2, description="Source URLs")

# === INSIGHTS ===
def generate_insights(news_articles, kb_collection):
    news_text = "\n\n".join([
        f"Title: {a['title']}\nDate: {a['date']}\nSummary: {a['summary']}\nSource: {a['link']}"
        for a in news_articles
    ]) if news_articles else "No recent updates."

    context = retrieve_context(kb_collection, "Australian small business compliance and brand voice")
    
    # Clear, explicit prompt that forces JSON output
    prompt = f"""[Your Guidelines]
{context}

[Recent News]
{news_text}

You are a trusted advisor to Australian micro-businesses (<5 employees). 
Generate a monthly insight in **valid JSON format only**, with this structure:
{{
  "title": "Australian Small Business Pulse: October 2025",
  "summary": "2-sentence overview...",
  "key_updates": ["Update 1", "Update 2", "Update 3"],
  "sources": ["https://example.com/1", "https://example.com/2"]
}}

Rules:
- Output ONLY the JSON object. No markdown, no text before or after.
- key_updates must have 3–4 items.
- sources must have at least 2 URLs.
- Use plain Australian English.
"""

    print("🧠 Generating with Ollama (OpenAI-compatible call)...")
    client = OpenAI(base_url="http://localhost:11434/v1",  # Ollama's local API
    api_key="ollama",  
    # dummy key; required by client but not used
    )

    response = client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
    )

    raw_output = response.choices[0].message.content.strip()


    # Clean common LLM artifacts
    if raw_output.startswith("```json"):
        raw_output = raw_output[7:]
    if raw_output.startswith("```"):
        raw_output = raw_output[3:]
    if raw_output.endswith("```"):
        raw_output = raw_output[:-3]

    # Parse and validate with Pydantic
    try:
        data = json.loads(raw_output)
        insight = BusinessInsight(**data)
        return insight
    except (json.JSONDecodeError, ValueError) as e:
        print(f"❌ Failed to parse JSON:\n{raw_output}")
        raise Exception(f"Invalid JSON or schema: {e}")

# === OUTPUTS ===
def save_outputs(insight: BusinessInsight):
    date_str = datetime.now().strftime("%d %B %Y")

    html = f"""<!DOCTYPE html>
<html lang="en-AU">
<head><meta charset="utf-8"><title>{insight.title}</title></head>
<body>
<h1>{insight.title}</h1>
<p><em>Published: {date_str}</em></p>
<p>{insight.summary}</p>
<h2>Key Updates</h2>
<ul>{''.join(f'<li>{u}</li>' for u in insight.key_updates)}</ul>
<h2>Sources</h2>
<ul>{''.join(f'<li><a href="{s}" target="_blank">{s}</a></li>' for s in insight.sources)}</ul>
</body>
</html>"""

    HTML_FILE.write_text(html, encoding="utf-8")
    print(f"✅ HTML saved: {HTML_FILE}")

    doc = DocxDocument()
    doc.add_heading(insight.title, 0)
    doc.add_paragraph(f"Published: {date_str}")
    doc.add_paragraph(insight.summary)
    doc.add_heading("Key Updates", 1)
    for update in insight.key_updates:
        doc.add_paragraph(update)
    doc.add_heading("Sources", 1)
    for src in insight.sources:
        doc.add_paragraph(src)
    doc.save(DOCX_FILE)
    print(f"✅ Word saved: {DOCX_FILE}")

    pd.DataFrame({"Action Item": insight.key_updates}).to_excel(EXCEL_FILE, index=False)
    print(f"✅ Excel saved: {EXCEL_FILE}")

# === ACCESSIBILITY ===
def audit_accessibility():
    print("🔍 Running WCAG audit...")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    try:
        driver = webdriver.Chrome(options=options)
        driver.get(f"file:///{HTML_FILE.resolve()}")
        axe = Axe(driver)
        axe.inject()
        results = axe.run()
        if results.get("violations"):
            print("⚠️ WCAG issues found:")
            for v in results["violations"]:
                print(f" - {v['description']}")
        else:
            print("✅ WCAG 2.1 AA compliant!")
    except Exception as e:
        print(f"⚠️ Audit skipped: {e}")
    finally:
        with contextlib.suppress(Exception):
            driver.quit()

# === MAIN ===
def main():
    print("🇦🇺 DigitalABCs Insights Agent (Local Stack)")
    kb = create_compliance_kb()
    news = fetch_recent_news()
    print(f"📰 Found {len(news)} articles.")
    try:
        insight = generate_insights(news, kb)
        print(f"🧠 Generated: {insight.title}")
        save_outputs(insight)
        audit_accessibility()
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        sys.exit(1)
    print(f"\n🎉 Success! Outputs in {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
