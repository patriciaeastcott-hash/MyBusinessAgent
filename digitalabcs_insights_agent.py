import os
import feedparser
from datetime import datetime, timedelta, timezone
import json
import logging
import smtplib
import requests  # Added for PHP upload
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import List, Dict, Set

# Google AI
import google.generativeai as genai
from vertexai.preview.vision_models import ImageGenerationModel
from google.cloud import aiplatform

from dotenv import load_dotenv

# ------------------ CONFIG ------------------
load_dotenv()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./blog_posts"))
OUTPUT_DIR.mkdir(exist_ok=True)

KB_FILE = OUTPUT_DIR / "knowledge_base.json"

# Email Config
EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", 587))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO = os.getenv("EMAIL_TO")

# Website Upload Config
# NOTE: Ensure UPLOAD_URL and UPLOAD_SECRET are set in your .env file
UPLOAD_URL = os.getenv("UPLOAD_URL")  
UPLOAD_SECRET = os.getenv("UPLOAD_SECRET") 

# Google AI & Project Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY missing in .env")
genai.configure(api_key=GEMINI_API_KEY)

PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
LOCATION = os.getenv("GOOGLE_LOCATION", "australia-southeast1")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not GOOGLE_APPLICATION_CREDENTIALS:
    raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS missing in .env")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

# Initialize AI Platform for Imagen (Vertex AI)
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Logging (WCAG compliant by logging errors and providing alerts via email)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "insights.log"),
        logging.StreamHandler()
    ]
)

# ------------------ SOURCES ------------------
FEED_URLS = [
    "https://www.accc.gov.au/media.rss",
    "https://www.oaic.gov.au/news-and-events.rss",
    "https://asic.gov.au/about-asic/media-centre/news-releases/media-releases-2025/feed/",
    "https://www.itnews.com.au/RSS/rss.ashx",
    "https://www.smartcompany.com.au/feed/",
    "https://www.innovationaus.com/feed/",
    "https://theconversation.com/au/topics/artificial-intelligence/feed",
]

# ------------------ CONTENT FILTERING ------------------
def is_business_article(title: str, summary: str, source: str) -> bool:
    """
    Filters for Australian business relevance, focusing on compliance, tech, and policy.
    Looser filtering to allow for 'chaos-to-clarity' narrative framing.
    """
    text = f"{title} {summary}".lower()
    
    # Block generic promos
    if any(promo in text for promo in ["pricing", "lifetime price", "buy now", "limited time offer"]):
        return False

    # Compliance & Business Signals
    business_signals = [
        'accc', 'asic', 'oaic', 'compliance', 'regulation', 'law', 'policy',
        'small business', 'sme', 'startup', 'funding', 'grant', 'scam', 'cyber',
        'data privacy', 'automation', 'governance', 'penalty', 'digital', 'ai', 'inclusion'
    ]
    
    # Require at least one strong signal
    return any(signal in text for signal in business_signals)

# ------------------ HELPERS ------------------
def load_knowledge_base() -> List[Dict]:
    if KB_FILE.exists():
        try:
            with open(KB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                cutoff = datetime.now(timezone.utc) - timedelta(days=7)
                
                valid_data = []
                for a in data:
                    try:
                        # Ensure we only keep articles with valid, recent dates
                        if datetime.fromisoformat(a["date"].replace("Z", "+00:00")) >= cutoff:
                            valid_data.append(a)
                    except ValueError:
                        continue
                return valid_data
        except Exception as e:
            logging.warning(f"Failed to load KB: {e}")
    return []

def save_knowledge_base(articles: List[Dict]):
    with open(KB_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

def is_recent(entry, days=7) -> bool:
    # Check published OR updated date
    for key in ['published_parsed', 'updated_parsed']:
        if hasattr(entry, key) and getattr(entry, key):
            pub = datetime(*getattr(entry, key)[:6], tzinfo=timezone.utc)
            if pub >= datetime.now(timezone.utc) - timedelta(days=days):
                return True
    return False

def get_seen_keys(articles: List[Dict]) -> Set[str]:
    # Use URL as primary unique key to avoid duplicate processing
    return {a["url"].lower() for a in articles}

def parse_feeds(seen_keys: Set[str]) -> List[Dict]:
    new_articles = []
    for url in FEED_URLS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if not is_recent(entry):
                    continue
                
                title = getattr(entry, 'title', '').strip()
                summary = (getattr(entry, 'summary', '') + ' ' + getattr(entry, 'description', ''))[:1500].strip()
                link = getattr(entry, 'link', '#').strip()
                source = getattr(feed.feed, 'title', 'Unknown')

                if not (title and link):
                    continue
                    
                unique_key = link.lower()
                if unique_key in seen_keys:
                    continue

                if is_business_article(title, summary, source):
                    pub_date = datetime.now(timezone.utc).isoformat()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                         pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()

                    new_articles.append({
                        "title": title,
                        "summary": summary,
                        "url": link,
                        "source": source,
                        "date": pub_date
                    })
                    seen_keys.add(unique_key)
        except Exception as e:
            logging.error(f"Parse error {url}: {e}")
    return new_articles

def generate_detailed_article(articles: List[Dict]) -> str:
    """
    Generates a timely article based on NEW articles using Brand Voice & WCAG compliance.
    """
    if not articles:
        return ""
    
    sources = "\n".join([
        f"- {a['title']} ({a['source']}): {a['summary'][:400]}..."
        for a in articles
    ])

    prompt = f"""
    You are Trish, the 'Chaos-to-Clarity Architect'. You help overwhelmed Australian small business owners build systems that work with their brains, not against them, promoting 'Asynchronous Automation'.

    **Task:** Write a detailed, timely blog post (~1000 words) summarizing the immediate impact of the following recent news items. Anchor the advice to the principle: "Stop being the 'rescuer' in your own business. Let the systems do the work for you."

    **Sources:**
    {sources}

    **Tone Guidelines (WCAG Compliant):**
    1.  **Authentic & Inclusive:** Use plain language. No corporate fluff. Be approachable.
    2.  **Neurodivergent Friendly:** Short paragraphs, clear <h2> headers, and bullet points.
    3.  **Focus:** Compliance, AI safety, scams, and opportunity.
    
    **Output Requirements:**
    -   Format: Clean HTML (h2, p, ul, li, strong).
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 3000}
        )
        return response.text.strip()
    except Exception as e:
        logging.error(f"Gemini Detailed Generation failed: {e}")
        return ""

def generate_synthetic_article(full_kb: List[Dict]) -> str:
    """
    Synthesizes a high-level, evergreen article when no new content is found.
    """
    if not full_kb:
        return ""
    
    all_sources = "\n".join([
        f"- {a['title']} ({a['source']}): {a['summary'][:400]}..."
        for a in full_kb
    ])

    prompt = f"""
    You are Trish, the 'Chaos-to-Clarity Architect'. Your most powerful asset is turning struggles into lived experience, teaching others how to build a business that works with their brain.

    **Task:** Write a high-level, synthetic blog post (~1000 words) by analyzing the underlying **trends and recurring issues** across the entire knowledge base from the past week. Do NOT summarize individual articles. Instead, synthesize the data into **one big theme** (e.g., "Why Automation is Your Only Scam Defense" or "The True Cost of Compliance Chaos").

    **Sources:**
    {all_sources}

    **Tone Guidelines (WCAG Compliant):**
    1.  **Authentic & Empowering:** Use plain language. Anchor the piece to the success of using 'Asynchronous Automation'.
    2.  **The Angle:** This is the 'Stability Blueprint' focus. How can automated systems save the owner's time and mental capacity?
    3.  **Call to Action:** Encourage them to book a "Digital Systems Audit."

    **Output Requirements:**
    -   Format: Clean HTML (h2, p, ul, li, strong).
    -   Compliance: Ensure content is easily scannable and digestible.
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 3000}
        )
        return response.text.strip()
    except Exception as e:
        logging.error(f"Gemini Synthetic Generation failed: {e}")
        return ""

def extract_title(html: str) -> str:
    """Safely extracts the <h2> title from the generated HTML."""
    if "<h2>" in html:
        try:
            start = html.find("<h2>") + 4
            end = html.find("</h2>", start)
            if end > start:
                return html[start:end].strip()
        except:
            pass
    return "Australian Business Insights" # Fallback title

def generate_thumbnail_with_imagen(title: str, output_path: Path) -> bool:
    """
    Generates an image using Vertex AI. 
    SAFE MODE: Uses a fixed abstract prompt to avoid Safety Filter triggers 
    caused by words like 'Scam', 'Trap', or 'Crisis' in the title.
    """
    try:
        # BRAND COLOR RULES: Navy (#1E3A8A), Purple (#7C3AED), Light Blue (#60A5FA).
        # We ignore the 'title' variable for the prompt to prevent safety blockers.
        
        safe_prompt = (
            "Abstract geometric technology illustration. "
            "Theme: Harmony, digital organization, clarity, efficiency. "
            "Color Palette: Primary Navy Blue and Purple with Light Blue accents. "
            "Style: Clean, flat vector art, minimal, corporate tech, white background. "
            "High quality, 4k, no text."
        )
        
        model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        image_response = model.generate_images(
            prompt=safe_prompt,
            number_of_images=1,
            aspect_ratio="16:9"
        )
        
        if image_response.images:
            with open(output_path, "wb") as f:
                f.write(image_response.images[0]._image_bytes)
            return True
        return False
    except Exception as e:
        # Log the specific error but don't crash the script
        logging.error(f"Imagen failed: {e}")
        return False

def upload_to_website(html_path: Path, image_path: Path, meta: Dict):
    """
    Uploads content to the PHP handler.
    Safe against missing images and file closure errors.
    """
    if not UPLOAD_URL or not UPLOAD_SECRET:
        logging.warning("Upload URL or Secret not set. Skipping upload.")
        return

    files = {} # Initialize dictionary to prevent 'UnboundLocalError'
    
    try:
        headers = {
            "Authorization": f"Bearer {UPLOAD_SECRET}"
        }

        # 1. Always prepare the HTML file
        files['insights_file'] = (html_path.name, open(html_path, 'rb'), 'text/html')
        
        # 2. Only try to upload the image if it actually exists
        if image_path.exists():
            files['insight_image'] = (image_path.name, open(image_path, 'rb'), 'image/png')
        else:
            logging.warning("Image file missing. Uploading article text only.")

        # 3. Send Request
        logging.info(f"Uploading to {UPLOAD_URL}...")
        response = requests.post(UPLOAD_URL, headers=headers, files=files)
        
        if response.status_code == 200:
            logging.info(f"Success! Server responded: {response.text}")
        else:
            logging.error(f"Upload failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        logging.error(f"Upload exception: {e}")
    finally:
        # Safely close any files that were actually opened
        for key, file_tuple in files.items():
            try:
                file_tuple[1].close()
            except Exception:
                pass
        
def send_email_alert(subject: str, body: str):
    # Re-included for completeness, uses original helper function logic
    if not EMAIL_ENABLED:
        return
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_USER
        msg["To"] = EMAIL_TO
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        logging.info("Email alert sent")
    except Exception as e:
        logging.error(f"Email failed: {e}")

def append_stability_blueprint(html_content: str) -> str:
    """
    Appends a structured, actionable 'Stability Blueprint' conclusion 
    to the generated article HTML, reinforcing the brand's core offering.
    """
    blueprint_html = """
<hr>
<h2>🧭 Your Stability Blueprint: Three Steps to Asynchronous Automation</h2>
<p>In a world where new laws, scams, and tech trends change weekly, you can't afford to be the last line of defense. Your most powerful move is building a 'set-it-forget-it' system that manages the chaos for you. This is **Asynchronous Automation** in action.</p>

<h3>Step 1: Audit Your Time-Sinks (Perceivable)</h3>
<ul>
    <li><strong>Identify the Rescuer Tasks:</strong> What tasks make you feel that 'flooding' or 'engulfment' dread? (e.g., chasing invoices, manually updating customer data after a purchase, scheduling social media posts).</li>
    <li><strong>Map the Chaos:</strong> Write down the process flow for the top three time-sinks. Use simple boxes and arrows. Where do you touch the process manually? That's your automation target.</li>
</ul>

<h3>Step 2: Define Your 'Set-It-Forget-It' System (Operable)</h3>
<ul>
    <li><strong>Compliance Gateways:</strong> Can a system automatically archive sensitive data after a set period (Privacy Act)? Can your forms ensure WCAG AA accessibility before publishing (DDA)?</li>
    <li><strong>System Triage:</strong> Implement tools (like Trello/Asana) to handle client onboarding without your real-time input. This preserves your child-free weekends and valuable deep work time.</li>
    <li><strong>AI Integration:</strong> Use your trusted tools (powered by Gemini/Google) to draft compliance summaries or first-pass email responses, but *never* for final decision-making.</li>
</ul>

<h3>Step 3: Schedule Your Clarity Review (Understandable & Robust)</h3>
<ul>
    <li><strong>Check, Don't Control:</strong> Dedicate 30 minutes, once a week, to check the health of your automated systems. Don't micro-manage them; simply ensure they ran successfully.</li>
    <li><strong>Iterate, Never React:</strong> If a system breaks, fix the *system*, not just the immediate emergency. This turns chaos into a stability lesson.</li>
</ul>
<p>Ready to move from chaos to clarity? Learn more about the Digital Systems Audit and how we build systems that work with your brain, not against it.</p>
"""
    # Append the blueprint HTML and return the combined content
    return html_content + blueprint_html

# ------------------ MAIN ------------------
def main():
    try:
        logging.info("Starting DigitalABCs Insight Pipeline...")
        
        # 1. Load & Parse Feeds
        kb = load_knowledge_base()
        seen = get_seen_keys(kb)
        new_articles = parse_feeds(seen) # Check for new articles
        full_kb = kb + new_articles
        save_knowledge_base(full_kb)
        
        # 2. Decide Content Strategy
        if not full_kb:
            msg = "Knowledge base is empty. Cannot generate content."
            logging.info(msg)
            return

        if new_articles:
            logging.info(f"Found {len(new_articles)} new articles. Generating timely report.")
            html_content = generate_detailed_article(new_articles)
        else:
            logging.warning(f"No new articles found. Synthesizing content from {len(full_kb)} stored items.")
            html_content = generate_synthetic_article(full_kb)
        
        if not html_content:
            raise Exception("Content generation returned empty content or failed.")
            
        # 3. Process Metadata & Save Locally
        title = extract_title(html_content)
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y%m%d_%H%M")

        html_path = OUTPUT_DIR / f"{timestamp}_article.html"
        image_path = OUTPUT_DIR / f"{timestamp}_thumbnail.png" # Changed to PNG for image quality
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # 4. Generate Brand-Compliant Image
        image_prompt = f"{title}. Australian small business owner focused on digital strategy and asynchronous automation."
        if not generate_thumbnail_with_imagen(image_prompt, image_path):
            logging.warning("Thumbnail generation failed. Ensure Vertex AI is configured correctly.")
            
        # 5. Upload to Website
        meta = {
            "title": title,
            "date": now.isoformat(),
            "tags": ["Automation", "Compliance", "Small Business", "Systems"],
            "word_count": len(html_content.split())
        }
        
        upload_to_website(html_path, image_path, meta)

        success_msg = f"Article '{title}' generated and uploaded."
        logging.info(success_msg)
        if EMAIL_ENABLED:
            send_email_alert("DigitalABCs Article Published", success_msg)

    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        logging.error(error_msg)
        if EMAIL_ENABLED:
            send_email_alert("DigitalABCs Insight Pipeline Error", error_msg)
        raise

if __name__ == "__main__":
    main()