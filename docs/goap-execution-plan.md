# GOAP Execution Plan - Interview Transcript Analysis

## Goal-Oriented Action Planning for JTBD Analysis Pipeline

This document defines the complete action sequence using Goal-Oriented Action Planning (GOAP) principles to transform raw interview transcripts into actionable JTBD insights.

---

## World State Definitions

### Initial State
```python
{
    "data_loaded": False,
    "data_cleaned": False,
    "text_preprocessed": False,
    "jtbd_extracted": False,
    "embeddings_generated": False,
    "sentiment_analyzed": False,
    "pain_points_detected": False,
    "clusters_created": False,
    "visualizations_created": False,
    "reports_generated": False,
    "insights_validated": False,
    "pipeline_documented": False,
    "goal_achieved": False
}
```

### Goal State
```python
{
    "goal_achieved": True,
    "reports_generated": True,
    "insights_validated": True,
    "pipeline_documented": True
}
```

---

## Action Definitions

### Action 1: `setup_environment`
**Preconditions**:
- `environment_ready`: False

**Effects**:
- `environment_ready`: True
- `libraries_installed`: True

**Execution Type**: Code (deterministic)
**Cost**: 1
**Tools**: pip, conda, venv

**Implementation**:
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
```

**Validation**:
- All imports succeed
- spaCy models downloadable
- GPU available (optional, for speed)

---

### Action 2: `load_raw_data`
**Preconditions**:
- `environment_ready`: True
- `csv_files_exist`: True

**Effects**:
- `data_loaded`: True
- `raw_dataframes_available`: True

**Execution Type**: Code
**Cost**: 1
**Tools**: pandas

**Implementation**:
```python
import pandas as pd

def load_transcripts():
    """Load all CSV files into pandas DataFrames"""
    data_dir = "/Users/nateaune/Documents/code/anthropic-interview-analysis/data"

    creatives = pd.read_csv(f"{data_dir}/creatives_transcripts.csv")
    scientists = pd.read_csv(f"{data_dir}/scientists_transcripts.csv")
    workforce = pd.read_csv(f"{data_dir}/workforce_transcripts.csv")

    # Add cohort labels
    creatives['cohort'] = 'creative'
    scientists['cohort'] = 'scientist'
    workforce['cohort'] = 'workforce'

    # Combine all data
    all_data = pd.concat([creatives, scientists, workforce], ignore_index=True)

    return {
        'creatives': creatives,
        'scientists': scientists,
        'workforce': workforce,
        'combined': all_data
    }
```

**Validation**:
- Row counts match expected values
- No null values in critical columns
- Cohort labels correctly assigned

---

### Action 3: `parse_conversations`
**Preconditions**:
- `data_loaded`: True

**Effects**:
- `conversations_parsed`: True
- `user_responses_extracted`: True

**Execution Type**: Hybrid (regex + NLP)
**Cost**: 2
**Tools**: regex, spaCy

**Implementation**:
```python
import re

def parse_conversation(text):
    """Extract user responses from conversation transcript"""
    # Split by speaker labels
    pattern = r'(User|AI|Assistant):\s*([^]*?)(?=(?:User|AI|Assistant):|$)'
    matches = re.findall(pattern, text, re.DOTALL)

    # Extract only user responses
    user_responses = [
        match[1].strip()
        for match in matches
        if match[0] == 'User'
    ]

    return user_responses

def extract_all_user_responses(df):
    """Apply to all transcripts"""
    df['user_responses'] = df['text'].apply(parse_conversation)
    df['response_count'] = df['user_responses'].apply(len)

    # Explode to one response per row (optional)
    df_exploded = df.explode('user_responses').reset_index(drop=True)

    return df, df_exploded
```

**Validation**:
- User responses correctly separated
- No interviewer questions in output
- Response count matches expected patterns

---

### Action 4: `clean_text`
**Preconditions**:
- `conversations_parsed`: True

**Effects**:
- `data_cleaned`: True
- `text_normalized`: True

**Execution Type**: Code
**Cost**: 1
**Tools**: cleantext, ftfy, regex

**Implementation**:
```python
from cleantext import clean
import ftfy

def clean_transcript_text(text):
    """Clean and normalize text"""
    # Fix unicode/encoding issues
    text = ftfy.fix_text(text)

    # Clean with cleantext library
    text = clean(
        text,
        fix_unicode=True,
        to_ascii=False,
        lower=False,  # Preserve case for NER
        no_line_breaks=False,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=True,
        no_punct=False,
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
    )

    # Additional cleaning
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()

    return text

def apply_cleaning(df, text_column='user_responses'):
    """Apply cleaning to all texts"""
    df['cleaned_text'] = df[text_column].apply(clean_transcript_text)
    return df
```

**Validation**:
- No encoding errors
- Whitespace normalized
- Critical content preserved

---

### Action 5: `extract_jtbd_patterns`
**Preconditions**:
- `data_cleaned`: True
- `text_preprocessed`: True

**Effects**:
- `jtbd_extracted`: True
- `pattern_matches_available`: True

**Execution Type**: Hybrid (spaCy patterns + LLM)
**Cost**: 3
**Tools**: spaCy, sentence-transformers

**Implementation**:
```python
import spacy
from spacy.matcher import Matcher

def create_jtbd_matcher(nlp):
    """Create pattern matcher for JTBD statements"""
    matcher = Matcher(nlp.vocab)

    # Pattern: "I use [TOOL] for [TASK]"
    pattern1 = [
        {"LOWER": "i"},
        {"LEMMA": {"IN": ["use", "turn", "reach", "leverage"]}},
        {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"},  # Tool
        {"LOWER": {"IN": ["for", "to", "when"]}},
        {"POS": {"IN": ["VERB", "NOUN"]}, "OP": "+"}  # Task
    ]

    # Pattern: "I need to [ACTION]"
    pattern2 = [
        {"LOWER": "i"},
        {"LEMMA": {"IN": ["need", "want", "try", "attempt"]}},
        {"LOWER": "to"},
        {"POS": "VERB"},
        {"POS": {"IN": ["VERB", "NOUN", "ADJ", "ADV"]}, "OP": "*"}
    ]

    # Pattern: "When I [CONTEXT], I [ACTION]"
    pattern3 = [
        {"LOWER": "when"},
        {"LOWER": "i"},
        {"POS": "VERB"},
        {"OP": "*"},
        {"IS_PUNCT": True},
        {"LOWER": "i"},
        {"POS": "VERB"}
    ]

    matcher.add("JTBD_USE_FOR", [pattern1])
    matcher.add("JTBD_NEED_TO", [pattern2])
    matcher.add("JTBD_WHEN_I", [pattern3])

    return matcher

def extract_jtbd_from_text(text, nlp, matcher):
    """Extract JTBD statements from text"""
    doc = nlp(text)
    matches = matcher(doc)

    jtbd_statements = []
    for match_id, start, end in matches:
        span = doc[start:end]
        jtbd_statements.append({
            'text': span.text,
            'pattern': nlp.vocab.strings[match_id],
            'start': start,
            'end': end
        })

    return jtbd_statements

def extract_all_jtbd(df, text_column='cleaned_text'):
    """Apply JTBD extraction to entire dataset"""
    nlp = spacy.load("en_core_web_trf")
    matcher = create_jtbd_matcher(nlp)

    df['jtbd_matches'] = df[text_column].apply(
        lambda x: extract_jtbd_from_text(x, nlp, matcher)
    )

    # Explode to one JTBD per row
    df_jtbd = df.explode('jtbd_matches').reset_index(drop=True)
    df_jtbd = df_jtbd[df_jtbd['jtbd_matches'].notna()]

    return df_jtbd
```

**Validation**:
- JTBD statements correctly extracted
- Patterns capture expected linguistic structures
- Sample validation >80% accuracy

---

### Action 6: `extract_jtbd_semantic`
**Preconditions**:
- `jtbd_extracted`: True (pattern-based)
- `embeddings_generated`: False

**Effects**:
- `semantic_jtbd_extracted`: True
- `embeddings_generated`: True

**Execution Type**: LLM + ML
**Cost**: 5
**Tools**: transformers, sentence-transformers, Claude API

**Implementation**:
```python
from sentence_transformers import SentenceTransformer
from transformers import pipeline

def generate_embeddings(texts, model_name='all-mpnet-base-v2'):
    """Generate semantic embeddings for texts"""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings

def extract_jtbd_with_llm(text, api_key=None):
    """Use LLM for semantic JTBD extraction"""
    prompt = f"""
    Analyze this interview response and extract all Jobs-To-Be-Done (JTBD) statements.

    A JTBD describes what the person is trying to accomplish, the task they need to do,
    or the goal they want to achieve.

    Extract each JTBD as a clear, concise statement. Also identify:
    - The context or situation
    - The tools or methods used
    - Any pain points or challenges mentioned

    Interview response:
    {text}

    Return as JSON:
    {{
        "jtbd_statements": [
            {{
                "job": "clear JTBD description",
                "context": "when this happens",
                "tools": ["tool1", "tool2"],
                "pain_points": ["challenge1"]
            }}
        ]
    }}
    """

    # Use Claude API or local model
    # Implementation depends on available API access
    return extract_with_api(prompt, api_key)

def semantic_jtbd_extraction(df, text_column='cleaned_text'):
    """Combine embeddings + optional LLM extraction"""
    # Generate embeddings for all texts
    texts = df[text_column].tolist()
    embeddings = generate_embeddings(texts)
    df['embedding'] = list(embeddings)

    # Optional: LLM extraction for subset
    # (cost-effective: sample or low-confidence cases only)

    return df
```

**Validation**:
- Embeddings capture semantic similarity
- LLM extraction adds missing JTBD
- Combined recall >90%

---

### Action 7: `detect_sentiment`
**Preconditions**:
- `jtbd_extracted`: True

**Effects**:
- `sentiment_analyzed`: True
- `sentiment_scores_available`: True

**Execution Type**: Hybrid (lexicon + transformer)
**Cost**: 3
**Tools**: VADER, transformers

**Implementation**:
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

def analyze_sentiment_vader(text):
    """VADER sentiment analysis"""
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores

def analyze_sentiment_transformer(texts, batch_size=16):
    """Transformer-based sentiment"""
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=0  # GPU if available
    )

    results = sentiment_pipeline(texts, batch_size=batch_size)
    return results

def analyze_emotions(texts, batch_size=16):
    """Emotion detection"""
    emotion_pipeline = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=0
    )

    results = emotion_pipeline(texts, batch_size=batch_size)
    return results

def comprehensive_sentiment_analysis(df, text_column='cleaned_text'):
    """Apply multi-level sentiment analysis"""
    # VADER (lexicon-based)
    df['vader_sentiment'] = df[text_column].apply(analyze_sentiment_vader)
    df['vader_compound'] = df['vader_sentiment'].apply(lambda x: x['compound'])

    # Transformer-based
    texts = df[text_column].tolist()
    transformer_results = analyze_sentiment_transformer(texts)
    df['transformer_sentiment'] = transformer_results

    # Emotion detection
    emotion_results = analyze_emotions(texts)
    df['emotions'] = emotion_results

    # Extract dominant emotion
    df['dominant_emotion'] = df['emotions'].apply(
        lambda x: max(x, key=lambda e: e['score'])['label']
    )

    return df
```

**Validation**:
- Sentiment scores align with manual annotation
- Emotion detection captures frustration/satisfaction
- Accuracy >75%

---

### Action 8: `detect_pain_points`
**Preconditions**:
- `sentiment_analyzed`: True

**Effects**:
- `pain_points_detected`: True
- `pain_scores_calculated`: True

**Execution Type**: Hybrid (patterns + ML)
**Cost**: 2
**Tools**: regex, spaCy, custom classifiers

**Implementation**:
```python
import re

# Pain point indicators
PAIN_INDICATORS = {
    'time_consuming': [
        r'\btakes?\s+(?:too\s+)?long\b',
        r'\btime[-\s]consuming\b',
        r'\btakes?\s+forever\b',
        r'\bslow\b',
        r'\bhours?\s+(?:and\s+hours?|to)\b'
    ],
    'difficulty': [
        r'\bstruggl(?:e|ing)\b',
        r'\bdifficult\b',
        r'\bchallenging\b',
        r'\bhard\s+to\b',
        r'\bstuck\b',
        r'\bconfus(?:ed|ing)\b'
    ],
    'tedium': [
        r'\btedious\b',
        r'\brepetitive\b',
        r'\bboring\b',
        r'\bmundane\b',
        r'\bmonotonous\b',
        r'\bdull\b'
    ],
    'frustration': [
        r'\bfrustrat(?:ed|ing)\b',
        r'\bannoying\b',
        r'\bhate\b',
        r'\bpainful\b',
        r'\birritat(?:ed|ing)\b'
    ],
    'stress': [
        r'\bstressful\b',
        r'\boverwhelm(?:ed|ing)\b',
        r'\banxious\b',
        r'\bpressure\b',
        r'\bworr(?:y|ied)\b'
    ]
}

def detect_pain_indicators(text):
    """Detect pain point indicators in text"""
    pain_scores = {category: 0 for category in PAIN_INDICATORS}

    text_lower = text.lower()
    for category, patterns in PAIN_INDICATORS.items():
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            pain_scores[category] += len(matches)

    return pain_scores

def calculate_pain_score(row):
    """Calculate composite pain score"""
    vader = row['vader_compound']
    pain_indicators = row['pain_indicators']
    emotion = row['dominant_emotion']

    # Composite score
    score = 0

    # Negative sentiment
    if vader < -0.2:
        score += abs(vader) * 10

    # Pain indicators
    total_indicators = sum(pain_indicators.values())
    score += total_indicators * 5

    # Negative emotions
    if emotion in ['anger', 'disgust', 'fear', 'sadness']:
        score += 10

    return score

def detect_all_pain_points(df, text_column='cleaned_text'):
    """Apply pain point detection to dataset"""
    df['pain_indicators'] = df[text_column].apply(detect_pain_indicators)
    df['pain_score'] = df.apply(calculate_pain_score, axis=1)

    # Flag high-pain tasks
    df['is_pain_point'] = df['pain_score'] > 15

    return df
```

**Validation**:
- Pain indicators correctly identified
- Scores correlate with manual assessment
- High-pain tasks flagged appropriately

---

### Action 9: `apply_topic_modeling`
**Preconditions**:
- `embeddings_generated`: True

**Effects**:
- `topics_discovered`: True
- `topic_labels_assigned`: True

**Execution Type**: ML
**Cost**: 4
**Tools**: BERTopic, gensim

**Implementation**:
```python
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

def apply_bertopic(texts, embeddings, nr_topics='auto'):
    """Apply BERTopic for topic discovery"""
    # Custom vectorizer to preserve meaningful n-grams
    vectorizer = CountVectorizer(
        ngram_range=(1, 3),
        stop_words='english',
        min_df=2
    )

    # Initialize BERTopic
    topic_model = BERTopic(
        nr_topics=nr_topics,
        vectorizer_model=vectorizer,
        verbose=True,
        calculate_probabilities=True
    )

    # Fit model
    topics, probabilities = topic_model.fit_transform(texts, embeddings)

    # Get topic info
    topic_info = topic_model.get_topic_info()

    return topic_model, topics, probabilities, topic_info

def extract_topic_labels(df, text_column='cleaned_text'):
    """Apply topic modeling and assign labels"""
    texts = df[text_column].tolist()
    embeddings = df['embedding'].tolist()

    topic_model, topics, probabilities, topic_info = apply_bertopic(
        texts, embeddings
    )

    df['topic'] = topics
    df['topic_probability'] = [probs.max() for probs in probabilities]

    # Map topic IDs to labels
    topic_labels = dict(zip(
        topic_info['Topic'],
        topic_info['Name']
    ))
    df['topic_label'] = df['topic'].map(topic_labels)

    return df, topic_model, topic_info
```

**Validation**:
- Topics are semantically coherent
- Topic labels meaningful and distinct
- Outlier topics handled appropriately

---

### Action 10: `cluster_jtbd`
**Preconditions**:
- `embeddings_generated`: True
- `topics_discovered`: True

**Effects**:
- `clusters_created`: True
- `cluster_labels_assigned`: True

**Execution Type**: ML
**Cost**: 3
**Tools**: UMAP, HDBSCAN, scikit-learn

**Implementation**:
```python
import umap
import hdbscan
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def reduce_dimensions(embeddings, n_components=5):
    """Apply UMAP for dimensionality reduction"""
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=n_components,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )

    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings, reducer

def cluster_hdbscan(embeddings, min_cluster_size=5):
    """Apply HDBSCAN clustering"""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=3,
        metric='euclidean',
        cluster_selection_method='eom'
    )

    cluster_labels = clusterer.fit_predict(embeddings)

    # Calculate cluster statistics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    return clusterer, cluster_labels, n_clusters, n_noise

def hierarchical_subclustering(df, cluster_col='hdbscan_cluster'):
    """Apply hierarchical clustering within major clusters"""
    subclusters = []

    for cluster_id in df[cluster_col].unique():
        if cluster_id == -1:  # Skip noise
            subclusters.extend([-1] * sum(df[cluster_col] == -1))
            continue

        cluster_data = df[df[cluster_col] == cluster_id]
        embeddings = np.vstack(cluster_data['reduced_embedding'].values)

        # Determine optimal subclusters (max 5 per cluster)
        n_subclusters = min(5, len(cluster_data) // 10)

        if n_subclusters > 1:
            agg = AgglomerativeClustering(n_clusters=n_subclusters)
            sub_labels = agg.fit_predict(embeddings)
            subclusters.extend(
                [f"{cluster_id}.{label}" for label in sub_labels]
            )
        else:
            subclusters.extend([f"{cluster_id}.0"] * len(cluster_data))

    return subclusters

def comprehensive_clustering(df):
    """Apply multi-stage clustering"""
    import numpy as np

    # Extract embeddings
    embeddings = np.vstack(df['embedding'].values)

    # Step 1: Dimensionality reduction
    reduced_embeddings, reducer = reduce_dimensions(embeddings)
    df['reduced_embedding'] = list(reduced_embeddings)

    # Step 2: HDBSCAN clustering
    clusterer, cluster_labels, n_clusters, n_noise = cluster_hdbscan(
        reduced_embeddings
    )
    df['hdbscan_cluster'] = cluster_labels

    # Step 3: Hierarchical subclustering
    df['hierarchical_cluster'] = hierarchical_subclustering(df)

    # Calculate cluster quality
    if n_clusters > 1:
        silhouette = silhouette_score(
            reduced_embeddings[cluster_labels != -1],
            cluster_labels[cluster_labels != -1]
        )
    else:
        silhouette = -1

    print(f"Clusters: {n_clusters}, Noise: {n_noise}, Silhouette: {silhouette:.3f}")

    return df, clusterer, reducer, silhouette
```

**Validation**:
- Silhouette score >0.5
- Clusters semantically distinct
- Outliers appropriately handled

---

### Action 11: `create_visualizations`
**Preconditions**:
- `clusters_created`: True
- `sentiment_analyzed`: True
- `pain_points_detected`: True

**Effects**:
- `visualizations_created`: True
- `interactive_dashboards_available`: True

**Execution Type**: Code
**Cost**: 2
**Tools**: plotly, matplotlib, seaborn

**Implementation**:
```python
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def create_cluster_scatter(df):
    """Interactive 3D scatter plot of clusters"""
    # Use first 3 dimensions of reduced embeddings
    import numpy as np
    reduced = np.vstack(df['reduced_embedding'].values)

    fig = px.scatter_3d(
        df,
        x=reduced[:, 0],
        y=reduced[:, 1],
        z=reduced[:, 2],
        color='topic_label',
        hover_data=['jtbd_text', 'pain_score', 'cohort'],
        title='JTBD Clusters in 3D Space'
    )

    return fig

def create_pain_heatmap(df):
    """Heatmap of pain scores across clusters and cohorts"""
    pivot = df.pivot_table(
        values='pain_score',
        index='topic_label',
        columns='cohort',
        aggfunc='mean'
    )

    fig = px.imshow(
        pivot,
        labels=dict(x="User Cohort", y="JTBD Cluster", color="Pain Score"),
        title="Pain Point Intensity by Cluster and Cohort"
    )

    return fig

def create_sentiment_distribution(df):
    """Sentiment distribution across JTBD clusters"""
    fig = px.violin(
        df,
        x='topic_label',
        y='vader_compound',
        color='cohort',
        box=True,
        title='Sentiment Distribution by JTBD Cluster'
    )

    return fig

def create_word_clouds(df, sentiment_threshold=0.2):
    """Word clouds for positive vs negative sentiment"""
    from wordcloud import WordCloud

    positive_text = ' '.join(
        df[df['vader_compound'] > sentiment_threshold]['cleaned_text']
    )
    negative_text = ' '.join(
        df[df['vader_compound'] < -sentiment_threshold]['cleaned_text']
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Positive word cloud
    wc_pos = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='Greens'
    ).generate(positive_text)
    ax1.imshow(wc_pos, interpolation='bilinear')
    ax1.set_title('Positive Sentiment', fontsize=20)
    ax1.axis('off')

    # Negative word cloud
    wc_neg = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='Reds'
    ).generate(negative_text)
    ax2.imshow(wc_neg, interpolation='bilinear')
    ax2.set_title('Negative Sentiment (Pain Points)', fontsize=20)
    ax2.axis('off')

    return fig

def generate_all_visualizations(df, output_dir='visualizations'):
    """Generate complete visualization suite"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 3D cluster scatter
    fig_clusters = create_cluster_scatter(df)
    fig_clusters.write_html(f"{output_dir}/cluster_scatter.html")

    # Pain heatmap
    fig_heatmap = create_pain_heatmap(df)
    fig_heatmap.write_html(f"{output_dir}/pain_heatmap.html")

    # Sentiment distribution
    fig_sentiment = create_sentiment_distribution(df)
    fig_sentiment.write_html(f"{output_dir}/sentiment_distribution.html")

    # Word clouds
    fig_wordclouds = create_word_clouds(df)
    fig_wordclouds.savefig(f"{output_dir}/word_clouds.png", dpi=300, bbox_inches='tight')

    print(f"Visualizations saved to {output_dir}/")
```

**Validation**:
- All visualizations render correctly
- Interactive features functional
- Insights clearly visible

---

### Action 12: `generate_reports`
**Preconditions**:
- `clusters_created`: True
- `visualizations_created`: True

**Effects**:
- `reports_generated`: True
- `actionable_insights_documented`: True

**Execution Type**: Hybrid (code + LLM)
**Cost**: 3
**Tools**: jinja2, pandas, LLM for summaries

**Implementation**:
```python
from jinja2 import Template
import json

def generate_jtbd_taxonomy(df):
    """Create hierarchical JTBD taxonomy"""
    taxonomy = {}

    for topic in df['topic_label'].unique():
        topic_data = df[df['topic_label'] == topic]

        taxonomy[topic] = {
            'count': len(topic_data),
            'avg_pain_score': topic_data['pain_score'].mean(),
            'cohort_distribution': topic_data['cohort'].value_counts().to_dict(),
            'top_jtbd': topic_data.nlargest(5, 'pain_score')[
                ['jtbd_text', 'pain_score', 'cohort']
            ].to_dict('records'),
            'sentiment_stats': {
                'mean': topic_data['vader_compound'].mean(),
                'std': topic_data['vader_compound'].std()
            }
        }

    return taxonomy

def generate_pain_point_report(df, top_n=20):
    """Generate ranked list of pain points"""
    pain_points = df.nlargest(top_n, 'pain_score')[[
        'jtbd_text', 'pain_score', 'vader_compound',
        'dominant_emotion', 'cohort', 'topic_label',
        'pain_indicators'
    ]].to_dict('records')

    return pain_points

def generate_executive_summary(df, taxonomy, pain_points):
    """Generate executive summary with LLM"""
    summary_prompt = f"""
    Analyze this JTBD analysis data and create an executive summary.

    Data overview:
    - Total JTBD statements: {len(df)}
    - Number of clusters: {df['topic_label'].nunique()}
    - Average pain score: {df['pain_score'].mean():.2f}
    - Top pain point categories: {list(taxonomy.keys())[:5]}

    Top 5 pain points:
    {json.dumps(pain_points[:5], indent=2)}

    Create an executive summary that includes:
    1. Key findings (3-5 bullets)
    2. Most painful jobs/tasks
    3. Opportunity areas for improvement
    4. Recommended priority interventions

    Be specific and actionable.
    """

    # Use LLM to generate summary
    # Implementation depends on API access
    summary = "Executive summary generated by LLM"

    return summary

def create_html_report(df, taxonomy, pain_points, summary, output_file):
    """Create comprehensive HTML report"""
    template = Template("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>JTBD Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; }
            h2 { color: #666; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            .pain-high { background-color: #ffcccc; }
            .pain-medium { background-color: #ffffcc; }
            .pain-low { background-color: #ccffcc; }
        </style>
    </head>
    <body>
        <h1>Interview Transcript Analysis - JTBD & Pain Points</h1>

        <h2>Executive Summary</h2>
        <p>{{ summary }}</p>

        <h2>JTBD Taxonomy</h2>
        <table>
            <tr>
                <th>Cluster</th>
                <th>Count</th>
                <th>Avg Pain Score</th>
                <th>Sentiment</th>
            </tr>
            {% for cluster, data in taxonomy.items() %}
            <tr>
                <td>{{ cluster }}</td>
                <td>{{ data.count }}</td>
                <td>{{ "%.2f"|format(data.avg_pain_score) }}</td>
                <td>{{ "%.2f"|format(data.sentiment_stats.mean) }}</td>
            </tr>
            {% endfor %}
        </table>

        <h2>Top Pain Points</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>JTBD</th>
                <th>Pain Score</th>
                <th>Cohort</th>
                <th>Emotion</th>
            </tr>
            {% for idx, pain in enumerate(pain_points[:20]) %}
            <tr class="{% if pain.pain_score > 30 %}pain-high{% elif pain.pain_score > 15 %}pain-medium{% else %}pain-low{% endif %}">
                <td>{{ idx + 1 }}</td>
                <td>{{ pain.jtbd_text }}</td>
                <td>{{ "%.1f"|format(pain.pain_score) }}</td>
                <td>{{ pain.cohort }}</td>
                <td>{{ pain.dominant_emotion }}</td>
            </tr>
            {% endfor %}
        </table>
    </body>
    </html>
    """)

    html_content = template.render(
        summary=summary,
        taxonomy=taxonomy,
        pain_points=pain_points
    )

    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"Report saved to {output_file}")

def generate_all_reports(df, output_dir='reports'):
    """Generate complete report suite"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Generate components
    taxonomy = generate_jtbd_taxonomy(df)
    pain_points = generate_pain_point_report(df)
    summary = generate_executive_summary(df, taxonomy, pain_points)

    # Save as JSON
    with open(f"{output_dir}/taxonomy.json", 'w') as f:
        json.dump(taxonomy, f, indent=2)

    with open(f"{output_dir}/pain_points.json", 'w') as f:
        json.dump(pain_points, f, indent=2)

    # Create HTML report
    create_html_report(
        df, taxonomy, pain_points, summary,
        f"{output_dir}/analysis_report.html"
    )

    return taxonomy, pain_points, summary
```

**Validation**:
- Reports generated successfully
- All sections populated
- Actionable insights present

---

## Complete Action Sequence (Optimal Path)

Using A* pathfinding, the optimal action sequence is:

1. **setup_environment** → `environment_ready: True`
2. **load_raw_data** → `data_loaded: True`
3. **parse_conversations** → `conversations_parsed: True`
4. **clean_text** → `data_cleaned: True`, `text_preprocessed: True`
5. **extract_jtbd_patterns** → `jtbd_extracted: True`
6. **extract_jtbd_semantic** → `semantic_jtbd_extracted: True`, `embeddings_generated: True`
7. **detect_sentiment** → `sentiment_analyzed: True`
8. **detect_pain_points** → `pain_points_detected: True`
9. **apply_topic_modeling** → `topics_discovered: True`
10. **cluster_jtbd** → `clusters_created: True`
11. **create_visualizations** → `visualizations_created: True`
12. **generate_reports** → `reports_generated: True`, `insights_validated: True`

**Total Cost**: 30 units
**Estimated Time**: 3 weeks (with parallel execution where possible)

---

## Replanning Triggers

The system will replan if:
- Action fails (e.g., data loading errors)
- Validation thresholds not met (e.g., clustering silhouette <0.4)
- New requirements discovered
- Resource constraints encountered
- Quality metrics below acceptable levels

---

## Success Criteria

### Quantitative
- JTBD extraction recall >80%
- Sentiment accuracy >75%
- Cluster silhouette score >0.5
- Pain point detection precision >70%

### Qualitative
- Clusters semantically coherent
- Insights actionable and specific
- Reports clear and accessible
- Stakeholder validation positive

---

## Conclusion

This GOAP plan provides a deterministic, measurable path from raw transcripts to actionable insights. Each action has clear preconditions, effects, and validation criteria, enabling dynamic replanning if conditions change or actions fail.
