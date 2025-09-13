Of course. Here is a detailed, step-by-step execution plan for the **THEME: TrendSpotter** problem statement, formatted as a comprehensive markdown document.

---

# L'Oréal x Monash Datathon 2025: TrendSpotter Execution Plan
## Step-by-Step Guide to Building a Multi-Platform Trend Radar

### **Phase 0: Team Setup & Tooling (Week 1)**
*   **Objective:** Establish a collaborative and efficient working environment.
*   **Actions:**
    1.  **Set Up a GitHub Repository:**
        *   Initialize a `git` repo with a clear `README.md`.
        *   Create a standard project structure:
            ```
            project_root/
            ├── data/
            │   ├── raw/          # Original data from L'Oréal and external sources
            │   ├── processed/    # Cleaned and preprocessed data
            │   └── interim/      # Intermediate data files
            ├── notebooks/        # For exploratory data analysis (EDA)
            ├── src/              # Main source code (Python modules)
            │   ├── data_processing.py
            │   ├── modeling.py
            │   └── visualization.py
            ├── models/           # To save trained models
            ├── assets/           # For images, audio samples, etc.
            ├── requirements.txt  # Project dependencies
            └── README.md
            ```
    2.  **Define Roles:** Based on team strengths, assign roles (e.g., Data Engineer, ML Specialist, Front-end/Streamlit Developer, Presenter).
    3.  **Install Libraries:** Create and share a `requirements.txt` file.
        ```txt
        # Core Data Handling
        pandas==2.0.3
        numpy==1.24.3
        scikit-learn==1.3.0

        # Time-Series & Modeling
        statsmodels==0.14.0
        prophet==1.1.4
        scipy==1.11.2

        # NLP & Audio
        transformers==4.31.0
        datasets==2.14.4
        torch==2.0.1
        spacy==3.7.2
        librosa==0.10.1
        pydub==0.25.1

        # Visualization & Dashboard
        plotly==5.15.0
        matplotlib==3.7.2
        streamlit==1.26.0
        wordcloud==1.9.2

        # Utilities
        tqdm==4.66.1
        python-dotenv==1.0.0
        ```
    4.  **Familiarize with Data:** As a team, explore the provided `.zip` dataset. Understand its schema, columns, and potential links to the problem.

---

### **Phase 1: Data Acquisition & Augmentation (Days 1-3)**
*   **Objective:** Gather and consolidate all necessary data for analysis.
*   **Actions:**
    1.  **Ingest Provided Data:** Load the L'Oréal dataset into a pandas DataFrame. Document its structure.
    2.  **Source External Data (Ethically & Responsibly):**
        *   **For Audio Trends:** Use the `pytubex` or `tiktok-scraper` (if compliant with terms of service) to download metadata (not content) of trending videos in Beauty and Lifestyle categories. Focus on getting audio titles and IDs.
        *   **For Hashtags/Keywords:** Use Twitter (X) API v2 or Instagram Graph API (if available) or a web scraper like `snscrape` to get posts with popular beauty hashtags (`#BeautyTok`, #GlowSkin, #MakeupTutorial`). **CRITICAL: Check platforms' Terms of Service, use rate limits, and avoid scraping personal data.**
        *   **Public Datasets:** Consider augmenting with public datasets from Kaggle (e.g., "TikTok Trending Videos").
    3.  **Store Data:** Organize all collected data in the `data/raw/` directory. Use efficient formats like `.parquet` or `.feather` for large files.

---

### **Phase 2: Data Preprocessing & Feature Engineering (Days 4-7)**
*   **Objective:** Clean the data and transform it into features suitable for modeling.
*   **Actions:**
    1.  **Text Data Cleaning (Hashtags, Comments, Captions):**
        *   Lowercase all text.
        *   Remove URLs, user mentions, and special characters (except '#' for hashtags).
        *   Handle emojis: either remove them or convert them to text (using libraries like `emoji`).
        *   Tokenize text and remove stop words.
    2.  **Audio Data Processing:**
        *   For audio snippets, use `librosa` to extract features like Mel-Frequency Cepstral Coefficients (MFCCs), chroma, and spectral contrast. These can be used as input features for a model or for audio similarity comparison.
        *   Alternatively, use audio fingerprinting to match unknown audio clips to a database of known trends.
    3.  **Time-Series Engineering:**
        *   Aggregate data into time bins (e.g., 1-hour or 6-hour intervals).
        *   For each interval, calculate counts for:
            *   Unique audio IDs
            *   Unique hashtags
            *   Keywords (e.g., "hyaluronic acid", "skin barrier")
        *   This creates a time-series dataset for anomaly detection.
    4.  **Create a "Trend Candidate" Table:** A master table where each row is a unique `(timestamp, feature)` pair (e.g., `(2025-09-13 14:00, #SunProtection)` with its corresponding count and other engineered features (e.g., rolling mean, rate of change).

---

### **Phase 3: Model Development & Trend Detection (Days 8-12)**
*   **Objective:** Implement algorithms to identify emerging trends and analyze their lifecycle.
*   **Actions:**
    1.  **Anomaly Detection (The Core):**
        *   For each time-series of a feature (hashtag, audio ID), use the **STL (Seasonal-Trend decomposition using Loess)** model from `statsmodels` to decompose it into trend, seasonal, and residual components.
        *   Points where the residual is significantly large (e.g., beyond 3 standard deviations) are potential anomalies, i.e., emerging trends.
        *   **Output:** A list of features flagged as "anomalous" at a given timestamp.
    2.  **Correlation & Validation:**
        *   **Cross-Platform Validation:** An anomaly is more credible if a hashtag spikes on Instagram *at the same time* its associated audio spikes on TikTok. Calculate simple correlations between the time-series of different features from different sources.
        *   **Semantic Validation:** Use a pre-trained sentence transformer (e.g., `all-MiniLM-L6-v2` from Hugging Face) to generate embeddings for hashtags and audio titles. Cluster these embeddings to find groups of features that are semantically related and rising together. This strengthens a trend candidate.
    3.  **Segment & Sentiment Analysis:**
        *   **Demographics:** Use a rule-based approach with `spaCy` NER on user bios/profile descriptions to look for age indicators (e.g., "Gen Z", "25yo", "university student").
        *   **Category Classification:** Fine-tune a small pre-trained transformer (like `DistilBERT`) on a sample of data labeled with categories (`Skincare`, `Makeup`, `Hair`, `Lifestyle`) to classify posts.
        *   **Sentiment Analysis:** Use an off-the-shelf model like `cardiffnlp/twitter-roberta-base-sentiment` from Hugging Face to assign sentiment scores to the text surrounding a trend.
    4.  **Decay Detection:**
        *   For a confirmed trend, calculate the **first and second derivatives** of its engagement time-series.
        *   Define a rule: `if growth_rate > 0 and acceleration < 0 for period T: then trend_state = "Decaying"`.

---

### **Phase 4: Prototype & Dashboard Development (Days 10-13)**
*   **Objective:** Build a functional and presentable prototype that showcases your insights.
*   **Actions:**
    1.  **Choose Streamlit:** Build an interactive web app using Streamlit. It's Python-based and perfect for data apps.
    2.  **Design the Dashboard Layout:**
        *   **Main View:** A table or list of the top 10-20 emerging trends, sorted by a composite "Trend Score" (e.g., `(anomaly_score * correlation_score * volume)`).
        *   **Filters:** Dropdowns to filter by category, platform, and demographic.
        *   **Trend Drill-Down:** When a user clicks on a trend, show a new page or section with:
            *   A time-series graph (Plotly) showing its growth trajectory and your annotated peak/decay points.
            *   Key statistics: Current growth rate, estimated demographic split, sentiment distribution.
            *   A word cloud of top associated terms.
            *   Sample posts or audio player (if available).
        *   **Insights Panel:** A section with auto-generated bullet points summarizing the trend ("#SunProtection is growing 25% W/W, primarily with Millennials, with 80% positive sentiment").
    3.  **Integrate the Model:** Connect your Streamlit UI to the preprocessed data and model outputs. Use caching (`@st.cache_data`) to ensure the app remains responsive.

---

### **Phase 5: Submission & Preparation (Final Day)**
*   **Objective:** Finalize all deliverables and prepare for submission.
*   **Actions:**
    1.  **Finalize Code:**
        *   Comment your code thoroughly.
        *   Ensure the `README.md` has clear instructions on how to install dependencies and run the Streamlit app.
        *   Test the entire pipeline from data loading to dashboard display.
    2.  **Package for Submission:**
        *   Create a final Zip file containing your project folder (excluding the `data/raw` folder if data is too large, but include a script to download it).
        *   Ensure the Zip file includes:
            *   All source code (`src/`)
            *   `requirements.txt`
            *   A pre-trained model file (if any) or instructions to run the training script
            *   The Streamlit app script (`app.py`)
            *   **A Jupyter Notebook (`EDA_and_Model_Training.ipynb`) that clearly walks through your entire process, from EDA to model training. This is crucial for judges to understand your work.**
    3.  **Record the Presentation Video (<5 mins):**
        *   **Script:** Problem -> Your Solution -> Quick Demo -> Impact.
        *   **Demo:** Show the dashboard live. Filter for a trend, drill down, and explain the insights L'Oréal would get.
        *   **Impact:** Conclude by stating how this tool would help L'Oréal "stay relevant, boost engagement, and drive innovation."
    4.  **Create the Pitch Deck (PDF):** Structure it exactly as requested in the handbook, using your project information.
    5.  **Submit Early:** Upload the Pitch Deck link, Video link (e.g., YouTube/Google Drive), and GitHub/Zip file link before the deadline. **Designate one person to submit.**

---

### **Success Metrics & Judging Alignment**
*   **Innovation & Creativity (30%):** Using multi-modal (audio + text) correlation and deriving semantic meaning from trends.
*   **Technical Execution (20%):** Clean code, well-structured project, use of appropriate and advanced models (STL, Transformers).
*   **Functionality (30%):** The dashboard directly answers the sub-themes: identifies trends, shows segments, and indicates decay.
*   **Presentation & Documentation (20%):** A professional video, a complete pitch deck, and an exceptionally well-documented GitHub repository with a clear `README.md` and explanatory notebook.

Good luck