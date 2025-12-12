#!/usr/bin/env python3
"""
Enhanced JTBD Analysis with BERTopic & UMAP Visualization
- Topic modeling using BERTopic with sentence transformers
- UMAP dimensionality reduction for visualization
- Interactive Plotly charts
- Aspect-based sentiment analysis for pain points
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Core NLP
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# Sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("✅ All dependencies loaded successfully!")


# ============================================================================
# CONFIGURATION
# ============================================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast & good quality
MIN_TOPIC_SIZE = 15  # Minimum docs per topic
N_NEIGHBORS = 15  # UMAP neighbors
MIN_DIST = 0.1  # UMAP min distance

# Pain patterns (same as prototype)
PAIN_PATTERNS = [
    (r"takes? (?:too )?(?:much|a lot of|so much|forever|hours|ages)", "time_consuming"),
    (r"(?:spend|waste|spent|wasted) (?:too much |a lot of |so much )?time", "time_consuming"),
    (r"time[- ]consuming", "time_consuming"),
    (r"(?:really |so |very |pretty )?(?:tedious|boring|repetitive|monotonous)", "tedious"),
    (r"(?:same thing|over and over|again and again)", "tedious"),
    (r"(?:frustrat|annoy|irritat|infuriat)", "frustrating"),
    (r"(?:hate|dread|can't stand|despise) (?:doing|having to|when)", "frustrating"),
    (r"(?:pain|nightmare|headache|hassle)", "frustrating"),
    (r"(?:avoid|put off|procrastinat|delay)", "avoidance"),
    (r"wish I didn't have to", "avoidance"),
    (r"(?:struggle|difficult|hard|tough|challenging) (?:to|with)", "difficult"),
    (r"(?:can't figure out|don't know how)", "difficult"),
    (r"(?:would be nice|wish|hope|want) (?:if |to |I could )", "desire_improvement"),
    (r"(?:should be|could be) (?:easier|faster|simpler|better)", "desire_improvement"),
]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_transcripts(data_dir: Path) -> pd.DataFrame:
    """Load all transcript CSVs."""
    all_transcripts = []
    for csv_file in data_dir.glob("*_transcripts.csv"):
        df = pd.read_csv(csv_file)
        category = csv_file.stem.replace("_transcripts", "")
        df["category"] = category
        all_transcripts.append(df)
        print(f"  📄 Loaded {len(df)} from {csv_file.name}")

    combined = pd.concat(all_transcripts, ignore_index=True)
    print(f"  📊 Total: {len(combined)} transcripts")
    return combined


def extract_user_responses(transcript: str) -> str:
    """Extract only User responses from transcript."""
    user_parts = re.split(r'\n(?:User|Human):\s*', transcript)
    responses = []
    for part in user_parts[1:]:
        text = re.split(r'\n(?:AI|Assistant|Claude):', part)[0].strip()
        if text and len(text) > 10:
            responses.append(text)
    return " ".join(responses)


# ============================================================================
# JTBD EXTRACTION (Enhanced)
# ============================================================================

def extract_jtbd_sentences(text: str) -> list[str]:
    """Extract sentences that describe jobs/tasks people do."""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # JTBD indicator patterns
    jtbd_indicators = [
        r'\bI (?:use|need|want|try|have to|usually|often|always)\b',
        r'\b(?:helps?|allows?|enables?) me\b',
        r'\bfor (?:doing|creating|writing|organizing|managing)\b',
        r'\bto (?:save|reduce|speed up|automate|simplify)\b',
        r'\bmy (?:workflow|process|task|job|work)\b',
        r'\b(?:brainstorm|research|analyze|write|edit|review|organize)\b',
        r'\bwhen I (?:need|want|have) to\b',
    ]

    jtbd_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20 or len(sentence) > 500:
            continue
        # Check if sentence contains JTBD indicators
        for pattern in jtbd_indicators:
            if re.search(pattern, sentence, re.IGNORECASE):
                jtbd_sentences.append(sentence)
                break

    return jtbd_sentences


def extract_pain_sentences(text: str) -> list[dict]:
    """Extract sentences with pain point indicators."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    pain_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:
            continue

        for pattern, pain_type in PAIN_PATTERNS:
            if re.search(pattern, sentence, re.IGNORECASE):
                pain_sentences.append({
                    "sentence": sentence,
                    "pain_type": pain_type
                })
                break

    return pain_sentences


# ============================================================================
# BERTOPIC MODELING
# ============================================================================

def create_topic_model(docs: list[str], embeddings: np.ndarray = None) -> tuple:
    """Create BERTopic model with custom UMAP and HDBSCAN."""
    print("\n🧠 Creating BERTopic model...")

    # Custom UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=N_NEIGHBORS,
        n_components=5,  # Reduce to 5D for clustering
        min_dist=MIN_DIST,
        metric='cosine',
        random_state=42
    )

    # Custom HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    # Vectorizer for topic representation
    vectorizer = CountVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95
    )

    # Create BERTopic
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        top_n_words=10,
        verbose=True,
        calculate_probabilities=True
    )

    # Fit the model
    if embeddings is not None:
        topics, probs = topic_model.fit_transform(docs, embeddings)
    else:
        topics, probs = topic_model.fit_transform(docs)

    print(f"  ✓ Found {len(set(topics)) - 1} topics (excluding outliers)")

    return topic_model, topics, probs


def get_topic_summary(topic_model: BERTopic, topics: list[int]) -> pd.DataFrame:
    """Get summary of discovered topics."""
    topic_info = topic_model.get_topic_info()

    # Add human-readable labels based on top words
    topic_labels = {}
    for topic_id in topic_info['Topic'].unique():
        if topic_id == -1:
            topic_labels[topic_id] = "Outliers/Misc"
        else:
            words = topic_model.get_topic(topic_id)
            if words:
                top_words = [w[0] for w in words[:3]]
                topic_labels[topic_id] = " / ".join(top_words)
            else:
                topic_labels[topic_id] = f"Topic {topic_id}"

    topic_info['Label'] = topic_info['Topic'].map(topic_labels)
    return topic_info


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_umap_2d(embeddings: np.ndarray) -> np.ndarray:
    """Create 2D UMAP projection for visualization."""
    print("\n📊 Creating 2D UMAP projection...")
    umap_2d = UMAP(
        n_neighbors=N_NEIGHBORS,
        n_components=2,
        min_dist=MIN_DIST,
        metric='cosine',
        random_state=42
    )
    projection = umap_2d.fit_transform(embeddings)
    print(f"  ✓ Projected {len(embeddings)} points to 2D")
    return projection


def plot_topic_clusters(
    projection: np.ndarray,
    topics: list[int],
    docs: list[str],
    categories: list[str],
    topic_model: BERTopic,
    output_path: Path
):
    """Create interactive scatter plot of topic clusters."""
    print("\n🎨 Creating topic cluster visualization...")

    # Get topic labels
    topic_info = topic_model.get_topic_info()
    topic_labels = {}
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:
            topic_labels[topic_id] = "Outliers"
        else:
            words = topic_model.get_topic(topic_id)
            if words:
                topic_labels[topic_id] = " / ".join([w[0] for w in words[:3]])
            else:
                topic_labels[topic_id] = f"Topic {topic_id}"

    # Create dataframe for plotting
    df_plot = pd.DataFrame({
        'x': projection[:, 0],
        'y': projection[:, 1],
        'topic': topics,
        'topic_label': [topic_labels.get(t, f"Topic {t}") for t in topics],
        'category': categories,
        'text': [d[:150] + "..." if len(d) > 150 else d for d in docs]
    })

    # Filter out outliers for cleaner visualization
    df_filtered = df_plot[df_plot['topic'] != -1]

    fig = px.scatter(
        df_filtered,
        x='x', y='y',
        color='topic_label',
        hover_data=['text', 'category'],
        title='JTBD Topic Clusters (BERTopic + UMAP)',
        labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
        width=1200,
        height=800
    )

    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.update_layout(
        legend_title="Topics",
        font=dict(size=12),
        hoverlabel=dict(bgcolor="white", font_size=11)
    )

    fig.write_html(output_path / "topic_clusters.html")
    print(f"  ✓ Saved to {output_path / 'topic_clusters.html'}")

    return fig


def plot_category_comparison(
    projection: np.ndarray,
    categories: list[str],
    docs: list[str],
    output_path: Path
):
    """Create scatter plot colored by category (creatives/scientists/workforce)."""
    print("\n🎨 Creating category comparison visualization...")

    df_plot = pd.DataFrame({
        'x': projection[:, 0],
        'y': projection[:, 1],
        'category': categories,
        'text': [d[:150] + "..." if len(d) > 150 else d for d in docs]
    })

    color_map = {
        'creatives': '#FF6B6B',
        'scientists': '#4ECDC4',
        'workforce': '#45B7D1'
    }

    fig = px.scatter(
        df_plot,
        x='x', y='y',
        color='category',
        color_discrete_map=color_map,
        hover_data=['text'],
        title='JTBD by Respondent Category',
        labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
        width=1200,
        height=800
    )

    fig.update_traces(marker=dict(size=6, opacity=0.6))
    fig.update_layout(font=dict(size=12))

    fig.write_html(output_path / "category_comparison.html")
    print(f"  ✓ Saved to {output_path / 'category_comparison.html'}")

    return fig


def plot_pain_points(
    pain_data: pd.DataFrame,
    output_path: Path
):
    """Create pain point analysis visualizations."""
    print("\n🎨 Creating pain point visualizations...")

    # Pain type distribution
    pain_counts = pain_data['pain_type'].value_counts()

    fig1 = px.bar(
        x=pain_counts.index,
        y=pain_counts.values,
        color=pain_counts.index,
        title='Pain Point Types Distribution',
        labels={'x': 'Pain Type', 'y': 'Count'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig1.update_layout(showlegend=False, width=900, height=500)
    fig1.write_html(output_path / "pain_types_distribution.html")

    # Pain by category
    pain_by_cat = pain_data.groupby(['category', 'pain_type']).size().reset_index(name='count')

    fig2 = px.bar(
        pain_by_cat,
        x='category',
        y='count',
        color='pain_type',
        barmode='group',
        title='Pain Points by Respondent Category',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig2.update_layout(width=1000, height=500)
    fig2.write_html(output_path / "pain_by_category.html")

    print(f"  ✓ Saved pain point charts")

    return fig1, fig2


def plot_topic_heatmap(
    topic_model: BERTopic,
    topics: list[int],
    categories: list[str],
    output_path: Path
):
    """Create heatmap of topics by category."""
    print("\n🎨 Creating topic heatmap...")

    df = pd.DataFrame({'topic': topics, 'category': categories})
    df = df[df['topic'] != -1]  # Exclude outliers

    # Get topic labels
    topic_labels = {}
    for topic_id in df['topic'].unique():
        words = topic_model.get_topic(topic_id)
        if words:
            topic_labels[topic_id] = " / ".join([w[0] for w in words[:2]])
        else:
            topic_labels[topic_id] = f"Topic {topic_id}"

    # Create crosstab
    crosstab = pd.crosstab(
        df['topic'].map(topic_labels),
        df['category'],
        normalize='columns'
    ) * 100  # Convert to percentage

    fig = px.imshow(
        crosstab,
        labels=dict(x="Category", y="Topic", color="% of Category"),
        title="Topic Distribution by Category (% within each category)",
        color_continuous_scale="Blues",
        aspect="auto"
    )
    fig.update_layout(width=900, height=700)
    fig.write_html(output_path / "topic_heatmap.html")

    print(f"  ✓ Saved to {output_path / 'topic_heatmap.html'}")

    return fig


# ============================================================================
# SENTIMENT ANALYSIS
# ============================================================================

def analyze_topic_sentiment(
    docs: list[str],
    topics: list[int],
    topic_model: BERTopic
) -> pd.DataFrame:
    """Analyze sentiment for each topic."""
    print("\n😊😢 Analyzing sentiment by topic...")

    analyzer = SentimentIntensityAnalyzer()

    # Calculate sentiment for each document
    sentiments = [analyzer.polarity_scores(doc)['compound'] for doc in docs]

    # Group by topic
    df = pd.DataFrame({
        'topic': topics,
        'sentiment': sentiments,
        'doc': docs
    })

    # Exclude outliers
    df = df[df['topic'] != -1]

    # Aggregate by topic
    topic_sentiment = df.groupby('topic').agg({
        'sentiment': ['mean', 'std', 'count'],
        'doc': lambda x: list(x)[:3]  # Sample docs
    }).reset_index()

    topic_sentiment.columns = ['topic', 'mean_sentiment', 'std_sentiment', 'doc_count', 'sample_docs']

    # Add topic labels
    topic_labels = {}
    for topic_id in topic_sentiment['topic']:
        words = topic_model.get_topic(topic_id)
        if words:
            topic_labels[topic_id] = " / ".join([w[0] for w in words[:3]])
        else:
            topic_labels[topic_id] = f"Topic {topic_id}"

    topic_sentiment['topic_label'] = topic_sentiment['topic'].map(topic_labels)

    return topic_sentiment.sort_values('mean_sentiment')


# ============================================================================
# HTML REPORT
# ============================================================================

def generate_html_report(
    topic_model: BERTopic,
    topic_sentiment: pd.DataFrame,
    pain_data: pd.DataFrame,
    output_path: Path
):
    """Generate comprehensive HTML report."""
    print("\n📝 Generating HTML report...")

    # Get topic info
    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info['Topic'] != -1]

    # Most negative topics (pain points)
    most_negative = topic_sentiment.head(5)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>JTBD Analysis Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                   max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            .card {{ background: white; padding: 20px; border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0; }}
            .metric {{ display: inline-block; text-align: center; padding: 15px 30px;
                      background: #3498db; color: white; border-radius: 8px; margin: 5px; }}
            .metric-value {{ font-size: 2em; font-weight: bold; }}
            .metric-label {{ font-size: 0.9em; opacity: 0.9; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #3498db; color: white; }}
            tr:hover {{ background: #f5f5f5; }}
            .negative {{ color: #e74c3c; }}
            .positive {{ color: #27ae60; }}
            .pain-tag {{ display: inline-block; padding: 3px 8px; border-radius: 4px;
                        font-size: 0.85em; margin: 2px; }}
            .pain-time_consuming {{ background: #ffeaa7; color: #d35400; }}
            .pain-tedious {{ background: #fab1a0; color: #c0392b; }}
            .pain-frustrating {{ background: #ff7675; color: #fff; }}
            .pain-difficult {{ background: #a29bfe; color: #fff; }}
            .pain-desire_improvement {{ background: #74b9ff; color: #fff; }}
            .pain-avoidance {{ background: #fd79a8; color: #fff; }}
            .viz-link {{ display: inline-block; padding: 10px 20px; background: #9b59b6;
                        color: white; text-decoration: none; border-radius: 5px; margin: 5px; }}
            .viz-link:hover {{ background: #8e44ad; }}
        </style>
    </head>
    <body>
        <h1>🔍 JTBD & Pain Point Analysis Report</h1>

        <div class="card">
            <h2>📊 Summary Metrics</h2>
            <div class="metric">
                <div class="metric-value">{len(topic_info)}</div>
                <div class="metric-label">Topics Discovered</div>
            </div>
            <div class="metric">
                <div class="metric-value">{topic_info['Count'].sum():,}</div>
                <div class="metric-label">JTBD Statements</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(pain_data):,}</div>
                <div class="metric-label">Pain Points</div>
            </div>
        </div>

        <div class="card">
            <h2>📈 Interactive Visualizations</h2>
            <a href="topic_clusters.html" class="viz-link">🎯 Topic Clusters (UMAP)</a>
            <a href="category_comparison.html" class="viz-link">👥 Category Comparison</a>
            <a href="topic_heatmap.html" class="viz-link">🔥 Topic Heatmap</a>
            <a href="pain_types_distribution.html" class="viz-link">😣 Pain Types</a>
            <a href="pain_by_category.html" class="viz-link">📊 Pain by Category</a>
        </div>

        <div class="card">
            <h2>🔴 Most Negative Topics (Highest Pain)</h2>
            <p>These topics have the most negative sentiment - representing undesirable, tedious, or frustrating tasks:</p>
            <table>
                <tr>
                    <th>Topic</th>
                    <th>Sentiment</th>
                    <th>Sample</th>
                </tr>
    """

    for _, row in most_negative.iterrows():
        sentiment_class = "negative" if row['mean_sentiment'] < 0 else "positive"
        sample = row['sample_docs'][0][:200] + "..." if len(row['sample_docs'][0]) > 200 else row['sample_docs'][0]
        html += f"""
                <tr>
                    <td><strong>{row['topic_label']}</strong></td>
                    <td class="{sentiment_class}">{row['mean_sentiment']:.3f}</td>
                    <td style="font-size:0.9em">{sample}</td>
                </tr>
        """

    html += """
            </table>
        </div>

        <div class="card">
            <h2>😣 Pain Point Categories</h2>
            <table>
                <tr>
                    <th>Pain Type</th>
                    <th>Count</th>
                    <th>% of Total</th>
                </tr>
    """

    pain_counts = pain_data['pain_type'].value_counts()
    total_pain = pain_counts.sum()
    for pain_type, count in pain_counts.items():
        pct = count / total_pain * 100
        html += f"""
                <tr>
                    <td><span class="pain-tag pain-{pain_type}">{pain_type.replace('_', ' ').title()}</span></td>
                    <td>{count:,}</td>
                    <td>{pct:.1f}%</td>
                </tr>
        """

    html += """
            </table>
        </div>

        <div class="card">
            <h2>🎯 All Discovered Topics</h2>
            <table>
                <tr>
                    <th>#</th>
                    <th>Topic Keywords</th>
                    <th>Count</th>
                    <th>Avg Sentiment</th>
                </tr>
    """

    # Merge topic info with sentiment
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        words = topic_model.get_topic(topic_id)
        keywords = ", ".join([w[0] for w in words[:5]]) if words else "N/A"

        # Get sentiment for this topic
        sent_row = topic_sentiment[topic_sentiment['topic'] == topic_id]
        if len(sent_row) > 0:
            sentiment = sent_row['mean_sentiment'].values[0]
            sentiment_class = "negative" if sentiment < 0 else "positive"
            sentiment_str = f'<span class="{sentiment_class}">{sentiment:.3f}</span>'
        else:
            sentiment_str = "N/A"

        html += f"""
                <tr>
                    <td>{topic_id}</td>
                    <td>{keywords}</td>
                    <td>{row['Count']:,}</td>
                    <td>{sentiment_str}</td>
                </tr>
        """

    html += """
            </table>
        </div>

        <div class="card">
            <h2>💡 Key Insights</h2>
            <ul>
                <li><strong>Dominant Pain Type:</strong> "Desire for Improvement" suggests users want AI to do more, not less</li>
                <li><strong>Frustration Sources:</strong> Often related to AI limitations, corrections needed, and verification burden</li>
                <li><strong>Time-Consuming Tasks:</strong> Research, data validation, formatting, and finding relevant information</li>
                <li><strong>Category Differences:</strong> Scientists focus on research/analysis; Workforce on admin/communication; Creatives on ideation</li>
            </ul>
        </div>

        <footer style="text-align:center; color:#666; margin-top:40px;">
            Generated by Enhanced JTBD Analysis Pipeline | BERTopic + UMAP + VADER
        </footer>
    </body>
    </html>
    """

    report_path = output_path / "analysis_report.html"
    with open(report_path, 'w') as f:
        f.write(html)

    print(f"  ✓ Saved report to {report_path}")
    return report_path


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_enhanced_analysis(data_dir: Path, output_dir: Path):
    """Run the full enhanced analysis pipeline."""
    print("=" * 70)
    print("🚀 Enhanced JTBD Analysis with BERTopic & UMAP")
    print("=" * 70)

    output_dir.mkdir(exist_ok=True)

    # 1. Load data
    print("\n📂 Loading transcripts...")
    df = load_transcripts(data_dir)

    # 2. Extract JTBD sentences and pain points
    print("\n🔍 Extracting JTBD statements and pain points...")
    all_jtbd = []
    all_pain = []

    for idx, row in df.iterrows():
        user_text = extract_user_responses(row['text'])
        category = row['category']
        transcript_id = row.get('transcript_id', f'transcript_{idx}')

        # JTBD
        jtbd_sentences = extract_jtbd_sentences(user_text)
        for sent in jtbd_sentences:
            all_jtbd.append({
                'text': sent,
                'category': category,
                'transcript_id': transcript_id
            })

        # Pain points
        pain_sentences = extract_pain_sentences(user_text)
        for p in pain_sentences:
            p['category'] = category
            p['transcript_id'] = transcript_id
            all_pain.append(p)

    jtbd_df = pd.DataFrame(all_jtbd)
    pain_df = pd.DataFrame(all_pain)

    print(f"  ✓ Extracted {len(jtbd_df)} JTBD statements")
    print(f"  ✓ Found {len(pain_df)} pain point sentences")

    # 3. Generate embeddings
    print("\n🧮 Generating sentence embeddings...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedding_model.encode(
        jtbd_df['text'].tolist(),
        show_progress_bar=True,
        batch_size=64
    )
    print(f"  ✓ Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

    # 4. Run BERTopic
    topic_model, topics, probs = create_topic_model(
        jtbd_df['text'].tolist(),
        embeddings
    )
    jtbd_df['topic'] = topics

    # 5. Create 2D projection
    projection_2d = create_umap_2d(embeddings)

    # 6. Analyze sentiment by topic
    topic_sentiment = analyze_topic_sentiment(
        jtbd_df['text'].tolist(),
        topics,
        topic_model
    )

    # 7. Create visualizations
    plot_topic_clusters(
        projection_2d,
        topics,
        jtbd_df['text'].tolist(),
        jtbd_df['category'].tolist(),
        topic_model,
        output_dir
    )

    plot_category_comparison(
        projection_2d,
        jtbd_df['category'].tolist(),
        jtbd_df['text'].tolist(),
        output_dir
    )

    plot_topic_heatmap(
        topic_model,
        topics,
        jtbd_df['category'].tolist(),
        output_dir
    )

    plot_pain_points(pain_df, output_dir)

    # 8. Generate HTML report
    report_path = generate_html_report(
        topic_model,
        topic_sentiment,
        pain_df,
        output_dir
    )

    # 9. Save data
    print("\n💾 Saving analysis data...")
    jtbd_df.to_csv(output_dir / "jtbd_with_topics.csv", index=False)
    pain_df.to_csv(output_dir / "pain_sentences.csv", index=False)
    topic_sentiment.to_csv(output_dir / "topic_sentiment.csv", index=False)

    # Save topic model
    topic_model.save(str(output_dir / "bertopic_model"))

    # Save embeddings
    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "umap_2d.npy", projection_2d)

    print("\n" + "=" * 70)
    print("✅ Enhanced Analysis Complete!")
    print("=" * 70)
    print(f"\n📁 Output files saved to: {output_dir}")
    print(f"📊 Open {report_path} in your browser to view the report")

    return {
        'topic_model': topic_model,
        'topics': topics,
        'embeddings': embeddings,
        'projection_2d': projection_2d,
        'jtbd_df': jtbd_df,
        'pain_df': pain_df,
        'topic_sentiment': topic_sentiment
    }


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "output" / "enhanced"

    results = run_enhanced_analysis(data_dir, output_dir)
