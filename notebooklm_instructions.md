# NotebookLM Instructions — Home & Lifestyle Ad Intelligence

Paste the text below as your prompt/source context in NotebookLM, then ask it to generate a presentation or audio overview.

---

## SOURCE DOCUMENT — paste this as your NotebookLM source

---

### Project: 1,000 Home & Lifestyle YouTube Ads Analyzed by AI

**What we did:**
We built an automated pipeline that found, downloaded, and analyzed 1,000 YouTube ads in the home remodeling and lifestyle space using Claude AI Vision (claude-sonnet-4-6). Each ad was labeled across 16 creative dimensions, producing a dataset of 16,000 labels. We then built a Streamlit dashboard and trained machine learning models to identify what makes ads successful.

**Total cost:** $26.50 (including failed/retried runs)
**Total pipeline runtime:** ~3 hours
**Dataset size:** 1,000 ads, 1.43 billion total views across the dataset

---

### Categories Covered (138 search queries)
Kitchen remodeling, bathroom renovation, basement finishing, flooring, windows, roofing, HVAC, general home remodeling, furniture (IKEA, Wayfair, Ashley Furniture, Rooms To Go, Pottery Barn, West Elm, Restoration Hardware, Crate & Barrel), mattresses (Casper, Purple, Sleep Number), home decor, kitchen appliances (fridge, dishwasher, oven), bathroom fixtures, paint (Sherwin-Williams, Benjamin Moore), patio furniture, smart home (Nest, Ring), plumbing/water, HomeGoods, TJMaxx, Pier 1.

---

### The Pipeline (How It Works)
1. **Search:** 138 YouTube queries run in parallel (8 threads) via yt-dlp
2. **Download:** Each ad downloaded at 360p (15–180 seconds, min 500 views)
3. **Frame extraction:** OpenCV extracts up to 20 evenly-spaced frames
4. **Transcript:** YouTube Transcript API (falls back to frames-only if blocked)
5. **Claude Vision analysis:** Frames + transcript sent to Claude Sonnet 4.6, which returns a JSON object with 16 labels
6. **Cache:** Results saved atomically to ads_cache.json after each video (crash-safe, incremental)

---

### The 16 Labels Claude Assigns Per Ad

**Categorical:**
- tone: upbeat / calm / inspirational / urgent / serious
- theme: product_demo / lifestyle / problem_solution / price_offer / before_after_transformation / testimonial
- hook_type: visual_transformation / pain_point / offer / celebrity / shocking_stat / question
- narrator_type: text_only / on_screen_talent / mixed / customer_testimonial / voiceover
- pacing: medium / slow_cinematic / fast_cuts
- color_palette: warm / cool / neutral / high_contrast
- music_mood: upbeat / calm / dramatic / tense / none
- setting: interior / exterior / studio / mixed
- cta_type: phone_number / website / visit_store / limited_time_offer / none
- target_audience: homeowners_general / families / budget_conscious / luxury / diy
- product_reveal_timing: early (<10s) / mid (10-30s) / late (>30s) / never

**Boolean:**
- has_cta: does the ad include a call to action?
- has_before_after: does it show before/after visuals?
- has_price_mention: does it mention a price?

**Numeric:** ad_length_seconds
**List:** emotional_appeal (aspiration, trust, pride, family, humor, fear)

---

### Key Statistics from 1,000 Ads

**Basic stats:**
- Median ad length: 30 seconds | Average: 45 seconds
- Median views: 11,580 | Average: 1,429,436 (skewed by viral outliers)
- Total views across all 1,000 ads: 1.43 billion

**Tone breakdown:**
- Upbeat: 462 ads (46.2%)
- Calm: 237 ads (23.7%)
- Inspirational: 171 ads (17.1%)
- Urgent: 73 ads (7.3%)
- Serious: 57 ads (5.7%)

**Theme breakdown:**
- Product Demo: 327 ads (32.7%) — most common
- Lifestyle: 297 ads (29.7%)
- Problem/Solution: 161 ads (16.1%)
- Price/Offer: 116 ads (11.6%)
- Before/After Transformation: 62 ads (6.2%)
- Testimonial: 37 ads (3.7%)

**Hook type breakdown:**
- Visual Transformation: 645 ads (64.5%) — dominant
- Pain Point: 176 ads (17.6%)
- Offer: 95 ads (9.5%)
- Celebrity: 39 ads (3.9%)
- Shocking Stat: 26 ads (2.6%)
- Question: 19 ads (1.9%)

**Boolean features:**
- 73.1% of ads have a CTA
- 16.0% use before/after visuals
- 22.1% mention a price

**Target audience:**
- Homeowners (general): 640 ads (64%)
- Families: 133 ads (13.3%)
- Budget conscious: 88 ads (8.8%)
- Luxury: 78 ads (7.8%)
- DIY: 61 ads (6.1%)

**Narrator type:**
- Text only: 532 ads (53.2%)
- On-screen talent: 361 ads (36.1%)
- Mixed: 78 ads (7.8%)
- Customer testimonial: 20 ads (2.0%)
- Voiceover: 9 ads (0.9%)

**Pacing:**
- Medium: 656 ads (65.6%)
- Slow cinematic: 219 ads (21.9%)
- Fast cuts: 125 ads (12.5%)

**Emotional appeals (multiple per ad):**
- Aspiration: 711 mentions
- Trust: 637 mentions
- Pride: 380 mentions
- Family: 300 mentions
- Humor: 231 mentions
- Fear: 74 mentions

---

### Performance Analysis — Median Views by Feature

**By hook type:**
- Shocking Stat: 46,501 median views ← BEST (5× above average)
- Pain Point: 30,952 median views
- Offer: 11,513 median views
- Celebrity: 10,935 median views
- Visual Transformation: 9,063 median views
- Question: 1,804 median views ← WORST

**By theme:**
- Problem/Solution: 29,867 median views ← BEST
- Lifestyle: 15,107 median views
- Price/Offer: 11,766 median views
- Product Demo: 8,683 median views
- Testimonial: 5,641 median views
- Before/After Transformation: 3,529 median views ← WORST

**By tone:**
- Upbeat: 15,149 median views ← BEST
- Inspirational: 9,613 median views
- Calm: 9,109 median views
- Serious: 6,786 median views
- Urgent: 5,859 median views ← WORST

**CTA paradox:**
- Ads WITHOUT a CTA: 23,019 median views
- Ads WITH a CTA: 8,326 median views
- Ads without CTAs get 2.8× more views than those with CTAs

**Before/After paradox:**
- Ads WITHOUT before/after: 12,168 median views
- Ads WITH before/after: 7,054 median views
- Before/after ads get 42% fewer views

---

### Performance Tiers

- Low (<25th percentile, <1,965 views): 250 ads (25%)
- Mid (25th–75th percentile, 1,965–176,189 views): 500 ads (50%)
- High (75th–90th percentile, 176K–2M views): 150 ads (15%)
- Viral (>90th percentile, >2M views): 100 ads (10%)

The top 10% of ads account for the vast majority of total views. The distribution is a classic power law / long tail.

---

### Key Insights

**Insight 1: Lead with a shocking stat, not a renovation reveal**
Shocking stat hooks earn 5× more median views than visual transformation hooks (46,501 vs 9,063). Opening with a surprising fact or number stops the scroll. The "before kitchen" shot is expected and ignored.

**Insight 2: Frame problems, not products**
Problem/Solution theme (29,867 views) outperforms Product Demo (8,683 views) by 3.4×. Viewers want to see themselves in the story, not a feature list. The narrative arc of: here's your problem → here's the solution → here's your life after, keeps viewers watching.

**Insight 3: Remove the CTA for brand-building campaigns**
On YouTube, soft-sell brand content outperforms direct-response by 2.8×. Ads without CTAs average 23,019 views vs 8,326 with CTAs. "Call now for a free quote" signals low production value and trains viewers to skip.

**Insight 4: Upbeat + Aspirational is the winning combination**
Upbeat tone earns the most views (15,149 median). Pair with aspiration emotional appeal — used by 71% of ads and still the top performer. Urgent tone (pressure tactics) underperforms by 61%.

**Insight 5: Before/After visuals are the category cliché**
Despite being visually compelling, ads with before/after visuals earn 42% fewer views. Every competitor uses them. The differentiated play is anything else: lifestyle moments, a bold stat, humor, or a problem-focused narrative.

---

### The Formula for a High-Performing Home & Lifestyle Ad

Based on ML feature importance and median view analysis:

= Shocking Stat OR Pain Point hook
+ Problem/Solution narrative structure
+ Upbeat tone with Aspirational emotional appeal
+ No direct CTA (brand-building, not direct-response)
+ No before/after reveal (too common, desensitized audience)
+ 30-second format with medium pacing
+ Text-heavy (designed for muted autoplay on mobile)

---

### Machine Learning Models

**Random Forest Regressor:** Predicts log(views) from 16 label features. 300 trees, max depth 6. R² = 0.65 in-sample. Top predictive features: hook_type, theme, has_cta, tone, ad_length_seconds.

**K-Means Clustering (k=4):** Discovered 4 distinct ad archetypes from the label data, visualized with PCA 2D scatter. Each archetype has a distinct tone × theme × hook profile and average performance level.

**Spearman Correlations:** Non-parametric rank correlations between each label value and view count, identifying features with significant monotonic relationships with performance.

**Interactive Success Predictor:** Users can select their ad's labels (tone, theme, hook, pacing, etc.) and instantly get: predicted views, percentile rank among the 1,000 ads, 80% confidence range, performance tier, and top 3 recommended changes to improve the prediction.

---

### Technical Stack
- Python 3.11 (Anaconda)
- yt-dlp: YouTube search and video download
- OpenCV: frame extraction from video
- youtube-transcript-api: transcript fetching
- Anthropic Claude API (claude-sonnet-4-6): vision + labeling
- Streamlit: interactive dashboard (5 tabs)
- Plotly: interactive charts
- scikit-learn: RandomForest, KMeans, PCA
- pandas + numpy: data processing
- GitHub + Streamlit Cloud: deployment

---

## SUGGESTED NOTEBOOKLM PROMPTS

After uploading the above as a source, try these prompts:

**For a presentation:**
"Create a 10-slide presentation from this research. Include: project overview, methodology, key statistics, the 5 most surprising findings, performance drivers, and actionable recommendations for a brand creating a home improvement ad. Make it persuasive and data-driven."

**For an audio overview / podcast:**
"Create a 5-minute podcast-style conversation between two hosts discussing the most surprising findings from this ad analysis research. Focus on the CTA paradox, the before/after myth, and what actually drives views."

**For an executive summary:**
"Write a 1-page executive summary of this research for a CMO at a home improvement company. Lead with the most actionable insights and include specific numbers."

**For a deep dive on one topic:**
"Based on this research, write a detailed analysis of why 'Problem/Solution' framing outperforms 'Product Demo' in home improvement advertising. Include the data, a hypothesis for why this is, and 3 specific creative recommendations."
