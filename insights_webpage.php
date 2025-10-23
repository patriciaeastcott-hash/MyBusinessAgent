<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Insights & Blog - DigitalABCs</title>
    <meta name="description" content="Explore the DigitalABCs blog for practical AI insights, educational articles, and vital cybersecurity updates for Australian small businesses.">
    <link rel="canonical" href="https://digitalabcs.com.au/insights.html">
    <link rel="icon" href="assets/favicon.ico" type="image/x-icon">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&family=Roboto+Mono&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="../style.css"> 
    
    <style>
        .dynamic-reports-grid {
            display: grid;
            grid-template-columns: 1fr; /* 1 column on mobile */
            gap: 30px;
            margin-bottom: 40px;
        }
        .blog-post-card {
            /* Using brand colours from your memory */
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            background-color: #f9f9f9;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            display: flex;
            flex-direction: column;
        }
        .blog-post-card h3 {
            color: var(--color-navy, #1E3A8A); 
            margin-top: 0;
        }
        .blog-post-card p {
            flex-grow: 1; /* Pushes the button to the bottom */
        }
        .btn-read-more {
            display: inline-block;
            background-color: var(--color-purple, #7C3AED); 
            color: #ffffff;
            padding: 10px 15px;
            border-radius: 5px;
            text-decoration: none;
            font-weight: 500;
            margin-top: 15px;
            transition: background-color 0.3s;
            text-align: center;
        }
        .btn-read-more:hover {
            background-color: var(--color-navy, #1E3A8A);
        }
        .section-divider {
            border: 0;
            border-top: 2px solid #eee;
            margin: 40px 0;
        }
        
        /* Make grid 2-col on larger screens */
        @media (min-width: 768px) {
            .dynamic-reports-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>
    <a href="#main-content" class="skip-link">Skip to main content</a>
    <header class="site-header">
        <div class="container">
            <div class="logo">DigitalABCs</div>
            <nav class="main-nav">
                <ul>
                    <li><a href="../index.html">Home</a></li>
                    <li><a href="../about.html">About</a></li>
                    <li><a href="../services.html">Services</a></li>
                    <li><a href="insights.php" class="active">Insights</a></li>
                    <li><a href="../contact.html">Contact</a></li>
                    <a href="https://services.digitalabcs.com.au" target="_blank">Client Login</a>
                </ul>
            </nav>
        </div>
    </header>
    <main id="main-content">
        <section class="hero-small">
            <div class="container">
                <h1>Insights from DigitalABCs</h1>
                <p class="subtitle">Stay informed with practical AI advice, industry trends, and essential cybersecurity tips for Australian small businesses.</p>
            </div>
        </section>
        
        <section class="insights-grid section-padding">
            <div class="container">

                <h2>Weekly AI & Automation Reports</h2>
                <p>Your latest AI-generated insights, delivered weekly. Newest reports are at the top.</p>
                <div class="dynamic-reports-grid">
                <?php
                // Set the directory to scan (current directory)
                $insightsDir = __DIR__;
                $webPath = '.'; // Relative web path
                
                // Find all generated insight files
                $files = glob($insightsDir . '/*_insights.html');
                
                // Sort files in reverse chronological order (newest first)
                if ($files) {
                    rsort($files);
                    
                    foreach ($files as $file) {
                        $doc = new DOMDocument();
                        // Suppress errors from parsing potentially imperfect HTML
                        @$doc->loadHTMLFile($file);
                        
                        $title = 'Weekly Insight'; // Default title
                        $summary = 'Read the latest report on AI and automation for small business.'; // Default summary
                        
                        // Get the real <title>
                        $titleNodes = $doc->getElementsByTagName('title');
                        if ($titleNodes->length > 0) {
                            $title = $titleNodes->item(0)->textContent;
                        }
                        
                        // Get the real summary
                        $xpath = new DOMXPath($doc);
                        $summaryNodes = $xpath->query("//p[contains(@class, 'summary-text')]");
                        if ($summaryNodes->length > 0) {
                            // Get first 200 chars for a snippet
                            $summary = substr(trim($summaryNodes->item(0)->textContent), 0, 200) . '...';
                        }
                        
                        // Get the web-accessible file name
                        $filename = basename($file);
                        
                        // Use file modification time for "Published" date
                        $published_date = date("F j, Y", filemtime($file));

                        // Create the HTML card
                        echo '<article class="blog-post-card">'; 
                        echo '<h3>' . htmlspecialchars($title) . '</h3>';
                        echo '<p class="post-meta">Published: ' . htmlspecialchars($published_date) . '</p>';
                        echo '<p>' . htmlspecialchars($summary) . '</p>';
                        echo '<a href="' . htmlspecialchars($webPath . '/' . $filename) . '" class="btn-read-more">Read Full Report</a>';
                        echo '</article>';
                    }
                } else {
                    echo '<p>No weekly reports have been generated yet. Please check back soon.</p>';
                }
                ?>
                </div>
                <hr class="section-divider"> <h2>From the Blog</h2>
                <p>Our foundational articles on AI, security, and local business growth.</p>

                <article class="blog-post">
                    <h2>AI in Western Sydney: How Local Businesses are Thriving</h2>
                    <p class="post-meta">By DigitalABCs Team | October 10, 2025 | Local Business, AI Adoption</p>
                    <img src="../assets/western-sydney-ai.jpg" alt="A vibrant cityscape of Western Sydney with subtle AI interface overlays, suggesting growth and innovation.">
                    <p>Western Sydney is a powerhouse of small businesses...</p>
                    <p>At DigitalABCs, we're passionate about helping our local community... <a href="../contact.html">Let's chat.</a></p>
                </article>
                <article class="blog-post">
                    <h2>Beyond ChatGPT: 5 AI Tools Small Businesses Aren't Using (But Should Be)</h2>
                    <p class="post-meta">By DigitalABCs Team | September 28, 2025 | AI Tools, Productivity, Business Growth</p>
                    <img src="../assets/ai-tools.jpg" alt="A diverse set of creative and technical tools, some with subtle glowing AI icons, suggesting varied applications.">
                    <p>Everyone knows ChatGPT, but the world of AI extends far beyond...</p>
                    <p>These tools are accessible, affordable... <a href="../services.html#ai-integrations">Check out our AI Integrations service.</a></p>
                </article>
                <article class="blog-post">
                    <h2>Australian Cyber Scams: Key Stats & How AI Helps You Stay Safe</h2>
                    <p class="post-meta">By DigitalABCs Team | September 20, 2025 | Cybersecurity, Scams, Data Protection</p>
                    <img src="../assets/cyber-security.jpg" alt="A digital padlock surrounded by abstract glowing lines, symbolising cybersecurity and data protection.">
                    <p>In our increasingly connected world, cyber scams are a constant threat...</p>
                    <p>Don't be a statistic... <a href="../contact.html">Protect your business today.</a></p>
                </article>
            </div>
        </section>
        <section class="cta-section section-padding">
            <div class="container">
                <h2>Want More Insights & Direct Support?</h2>
                <p>Subscribe to our newsletter for the latest AI trends and cybersecurity tips, or book a free call to discuss specific challenges your business faces.</p>
                <a href="../contact.html" class="btn btn-primary">Book a Free Discovery Call</a>
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
</body>
</html>