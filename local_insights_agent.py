# insights.py - Generate Australian Business Insights as HTML
import os
from datetime import datetime

# --- CONFIG ---
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- MOCK DATA (replace this with your real feed-parsed data) ---
# For testing only — in your real script, this comes from feedparser
articles = [
    {
        "title": "Black Friday scam alert: AI-generated fake stores on the rise",
        "summary": "The Australian Banking Association warns shoppers to verify URLs and avoid fake parcel texts as scammers exploit holiday sales.",
        "url": "https://www.dynamicbusiness.com.au/black-friday-scams",
        "source": "Dynamic Business"
    },
    {
        "title": "80+ firms named 2025–2026 Inclusive Employers by DCA",
        "summary": "Diversity Council Australia recognises organisations excelling in workplace inclusion, linking it to performance and wellbeing.",
        "url": "https://www.smartcompany.com.au/inclusive-employers",
        "source": "SmartCompany"
    },
    {
        "title": "Chocolate On Purpose wins award for Indigenous-led social enterprise",
        "summary": "Australia’s first 100% Indigenous-owned chocolate company wins national award for economic sovereignty and impact.",
        "url": "https://www.itnews.com.au/chocolate-on-purpose",
        "source": "iTnews"
    },
    {
        "title": "OpenAI restricts legal advice: AI is a co-pilot, not a lawyer",
        "summary": "Businesses must now involve qualified professionals when using AI for legal decisions. DIY legal drafting is risky.",
        "url": "https://dynamicbusiness.com/zed-law-openai",
        "source": "Dynamic Business"
    }
]

# Group by topic (mock)
top_topics = [
    ("Consumer Protection", [articles[0]]),
    ("Inclusive Leadership", [articles[1]]),
    ("Indigenous Enterprise", [articles[2]])
]

# --- GENERATE HTML ---
timestamp = datetime.now().strftime("%Y-%m-%d")

# Build takeaways
takeaways = []
for topic, arts in top_topics:
    if arts:
        art = arts[0]
        takeaways.append({
            "title": art["title"],
            "summary": (art["summary"][:120] + ("…" if len(art["summary"]) > 120 else "")),
            "url": art["url"],
            "source": art["source"]
        })

# Start HTML
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-4JD4ZCN0G8"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());
      gtag('config', 'G-4JD4ZCN0G8');
    </script>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Australian Business Insights — DigitalABCs</title>
    <meta name="description" content="Actionable insights for Australian small businesses on AI, compliance, inclusion, and innovation.">
    <meta name="keywords" content="Australian business news, small business insights, AI compliance, inclusive leadership, Indigenous enterprise, scam alerts">
    <meta property="og:title" content="Australian Business Insights — DigitalABCs">
    <meta property="og:description" content="Curated, actionable intelligence for Australian SMEs.">
    <meta property="og:type" content="article">
    <meta property="og:locale" content="en_AU">
    <link rel="canonical" href="https://digitalabcs.com.au/insights.html">
    <link rel="icon" href="assets/favicon.ico" type="image/x-icon">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="style.css">
    <script async src="https://tally.so/widgets/embed.js"></script>
</head>
<body>
    <a href="#main-content" class="skip-link">Skip to main content</a>
    <header class="site-header">
        <div class="container">
            <a href="index.html" class="logo" aria-label="DigitalABCs Home">
                <img src="assets/logo.png" alt="DigitalABCs Logo" class="logo-img">
            </a>
            <nav class="main-nav">
                <ul>
                    <li><a href="index.html">Home</a></li>
                    <li><a href="about.html">About</a></li>
                    <li><a href="services.html">Services</a></li>
                    <li><a href="insights.html" class="active">Insights</a></li>
                    <li><a href="contact.html">Contact</a></li>
                    <li><a href="https://services.digitalabcs.com.au" target="_blank">Client Login</a></li>
                </ul>
            </nav>
        </div>
    </header>

    <main id="main-content">
        <section class="hero">
            <div class="container">
                <h1>Australian Business Insights</h1>
                <p class="subtitle">Curated for small business owners | Published: {timestamp}</p>
            </div>
        </section>

        <section class="insights-content">
            <div class="container">
                <h2>🔍 Top 3 Actionable Takeaways This Week</h2>
"""

# Add takeaways
for i, t in enumerate(takeaways[:3], 1):
    html += f"""
                <div class="takeaway">
                    <strong>{i}. {t['title']}</strong><br>
                    <span class="action">👉 Do this now:</span> {t['summary']}<br>
                    <a href="{t['url']}" target="_blank">Read more</a> • <em>via {t['source']}</em>
                </div>
"""

# Add deep dives
html += """
                <div class="deep-dive">
                    <h2>📚 Worth Your Time</h2>
"""
for art in articles[:8]:
    summary_snippet = art["summary"][:100] + ("…" if len(art["summary"]) > 100 else "")
    html += f"""
                    <p>
                        <strong><a href="{art['url']}" target="_blank">{art['title']}</a></strong><br>
                        {summary_snippet}<br>
                        <span class="cite">— {art['source']}</span>
                    </p>
"""

# Close HTML
html += """
                </div>
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
                        <li><a href="about.html">About Us</a></li>
                        <li><a href="services.html">Our Services</a></li>
                        <li><a href="insights.html">Insights</a></li>
                        <li><a href="contact.html">Contact</a></li>
                    </ul>
                </div>
                <div class="footer-col">
                    <h4>Legal</h4>
                    <ul>
                        <li><a href="privacy.html">Privacy Policy</a></li>
                        <li><a href="terms.html">Terms of Service</a></li>
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

    <style>
        .tally-float-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: #ba38eb;
            color: #fff;
            border: none;
            border-radius: 50px;
            padding: 14px 22px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            z-index: 9999;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease-in-out;
        }
        .tally-float-btn:hover {
            background-color: #10B981;
            transform: translateY(-2px);
        }
        @media (max-width: 600px) {
            .tally-float-btn {
                padding: 12px 18px;
                font-size: 14px;
                bottom: 15px;
                right: 15px;
            }
        }
        .takeaway {
            background: #f5f9ff;
            padding: 16px;
            margin: 20px 0;
            border-left: 4px solid #0066cc;
            border-radius: 0 4px 4px 0;
        }
        .action {
            font-weight: 600;
            color: #d32f2f;
        }
        .cite {
            font-size: 0.9em;
            color: #666;
        }
        .insights-content .container {
            max-width: 720px;
            margin: 0 auto;
            padding: 20px;
        }
    </style>

    <button class="tally-float-btn" data-tally-open="wkDaP1"
        data-tally-layout="modal" data-tally-width="700" data-tally-overlay="true"
        data-tally-hide-title="false" data-tally-emoji-text="💡"
        data-tally-emoji-animation="tada">
        Where are you in your AI journey?
    </button>
</body>
</html>
"""

# --- SAVE FILE ---
filename = f"{datetime.now().strftime('%Y%m%d')}_Insight.html"
filepath = os.path.join(OUTPUT_DIR, filename)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ Generated: {filepath}")