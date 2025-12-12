# Anthropic Interview Transcript Analysis

NLP-powered analysis of interview transcripts to identify Jobs-To-Be-Done (JTBD), pain points, and tasks people want to delegate to AI.

## Overview

This project analyzes **1,445 interview transcripts** from the [Anthropic Interviewer dataset](https://huggingface.co/datasets/Anthropic/AnthropicInterviewer) to discover:

- **Jobs-To-Be-Done (JTBD)**: What tasks people are using AI for
- **Pain Points**: Which tasks cause frustration, tedium, or time waste
- **Handoff Candidates**: Tasks people most want to delegate to AI

The analysis uses BERTopic for topic modeling, UMAP for visualization, VADER for sentiment analysis, and custom pattern matching for task extraction.

## Key Findings

### Top 5 Tasks People Want to Hand Off to AI

| Rank | Task | Mentions | Pain Level |
|------|------|----------|------------|
| 1 | Email Management & Triage | 1,890 | HIGH |
| 2 | Administrative Paperwork | 3,831 | HIGH |
| 3 | Scheduling & Calendar | 601 | HIGH |
| 4 | Research & Comparison | 3,203 | MEDIUM-HIGH |
| 5 | Financial Tasks & Bills | 793 | MEDIUM |

### Pain Point Distribution

- **31.6%** - Desire for improvement ("wish AI could...")
- **29.8%** - Frustration with current process
- **12.6%** - Tedious/repetitive tasks
- **12.2%** - Difficult tasks
- **8.4%** - Time-consuming tasks
- **5.4%** - Avoidance behavior

## Data Source

The dataset contains interview transcripts from three cohorts:

| File | Respondents | Transcripts |
|------|-------------|-------------|
| `creatives_transcripts.csv` | Creative professionals | 125 |
| `scientists_transcripts.csv` | Scientists/researchers | 125 |
| `workforce_transcripts.csv` | General workforce | ~1,200 |

**Source**: [Anthropic Interviewer Dataset](https://huggingface.co/datasets/Anthropic/AnthropicInterviewer)
**Research Paper**: [anthropic.com/research/anthropic-interviewer](https://www.anthropic.com/research/anthropic-interviewer)

## Project Structure

```
anthropic-interview-analysis/
├── data/                           # Raw transcript CSVs
│   ├── creatives_transcripts.csv
│   ├── scientists_transcripts.csv
│   └── workforce_transcripts.csv
├── scripts/                        # Analysis scripts
│   ├── enhanced_analysis.py        # Main BERTopic + UMAP pipeline
│   ├── adulting_jtbd_analysis.py   # Life management task extraction
│   ├── personal_life_deep_search.py
│   ├── bertopic_crossref_and_handoff.py  # Final handoff report
│   └── prototype_analysis.py
├── output/                         # Generated results
│   ├── enhanced/                   # BERTopic visualizations & model
│   │   ├── analysis_report.html    # Main interactive report
│   │   ├── topic_clusters.html     # UMAP cluster visualization
│   │   ├── category_comparison.html
│   │   ├── topic_heatmap.html
│   │   ├── jtbd_with_topics.csv
│   │   └── bertopic_model/         # Saved BERTopic model
│   ├── adulting_jtbd/              # Life management analysis
│   └── personal_life_search/
├── docs/                           # Documentation & reports
│   ├── research-plan.md            # Full methodology
│   ├── adulting-jtbd-analysis-report.md
│   └── top-5-handoff-candidates-report.md
└── requirements.txt                # Python dependencies
```

## Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/anthropic-interview-analysis.git
cd anthropic-interview-analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (if needed)
python -m spacy download en_core_web_sm
```

### Dependencies

Key libraries used:

| Library | Purpose |
|---------|---------|
| `bertopic` | Topic modeling with transformers |
| `sentence-transformers` | Semantic embeddings |
| `umap-learn` | Dimensionality reduction |
| `hdbscan` | Density-based clustering |
| `vaderSentiment` | Lexicon-based sentiment |
| `plotly` | Interactive visualizations |
| `pandas` | Data manipulation |

## Usage

### Run the Full Analysis Pipeline

```bash
# Step 1: Enhanced JTBD Analysis with BERTopic
python scripts/enhanced_analysis.py

# Step 2: Adulting/Life Management Task Extraction
python scripts/adulting_jtbd_analysis.py

# Step 3: Personal Life Context Search
python scripts/personal_life_deep_search.py

# Step 4: Generate Final Handoff Report
python scripts/bertopic_crossref_and_handoff.py
```

### Quick Start: Single Script

```bash
# Run just the main analysis
python scripts/enhanced_analysis.py
```

This produces:
- `output/enhanced/analysis_report.html` - Interactive HTML report
- `output/enhanced/topic_clusters.html` - UMAP visualization
- `output/enhanced/jtbd_with_topics.csv` - Extracted JTBD with topic assignments

### View Results

Open the interactive reports in your browser:

```bash
# macOS
open output/enhanced/analysis_report.html

# Linux
xdg-open output/enhanced/analysis_report.html

# Windows
start output/enhanced/analysis_report.html
```

## Methodology

### 1. JTBD Extraction

Pattern-based extraction using linguistic markers:

```python
# Example patterns
"I use [TOOL] for [TASK]"
"I need to [ACTION]"
"When I [CONTEXT], I [ACTION]"
"I turn to [TOOL] when [SITUATION]"
```

### 2. Topic Modeling (BERTopic)

1. Generate sentence embeddings using `all-MiniLM-L6-v2`
2. Reduce dimensions with UMAP (15 neighbors, 5 components)
3. Cluster with HDBSCAN (min cluster size: 15)
4. Extract topic keywords with c-TF-IDF

### 3. Sentiment Analysis

Multi-level approach:
- **VADER**: Lexicon-based compound scores
- **Pain Patterns**: Custom regex for frustration, tedium, time-waste
- **Delegation Indicators**: "wish AI would", "hate doing", "tedious"

### 4. Visualization

- **UMAP 2D Projection**: Interactive cluster scatter plots
- **Topic Heatmaps**: Distribution across respondent categories
- **Pain Point Charts**: Bar charts by type and category

## Output Files

### Main Analysis (`output/enhanced/`)

| File | Description |
|------|-------------|
| `analysis_report.html` | Comprehensive interactive dashboard |
| `topic_clusters.html` | UMAP visualization of JTBD clusters |
| `category_comparison.html` | Creatives vs Scientists vs Workforce |
| `topic_heatmap.html` | Topic distribution heatmap |
| `pain_types_distribution.html` | Pain point breakdown |
| `jtbd_with_topics.csv` | All extracted JTBD with topic IDs |
| `topic_sentiment.csv` | Sentiment scores per topic |
| `embeddings.npy` | Cached sentence embeddings |
| `bertopic_model/` | Saved model for reuse |

### Adulting Analysis (`output/adulting_jtbd/`)

| File | Description |
|------|-------------|
| `adulting_jtbd_mentions.csv` | All life management task mentions |
| `adulting_jtbd_summary.csv` | Summary by task category |
| `top_handoff_quotes.csv` | Best delegation-desire quotes |

### Final Report (`output/`)

| File | Description |
|------|-------------|
| `top_5_handoff_candidates.csv` | Ranked handoff candidates with evidence |

## Configuration

Key parameters in `scripts/enhanced_analysis.py`:

```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast & good quality
MIN_TOPIC_SIZE = 15                   # Minimum docs per topic
N_NEIGHBORS = 15                      # UMAP neighbors
MIN_DIST = 0.1                        # UMAP min distance
```

## Reproducing Results

1. **Download the data** from [Hugging Face](https://huggingface.co/datasets/Anthropic/AnthropicInterviewer)
2. Place CSV files in `data/` directory
3. Run the scripts in order:
   ```bash
   python scripts/enhanced_analysis.py
   python scripts/adulting_jtbd_analysis.py
   python scripts/bertopic_crossref_and_handoff.py
   ```
4. View `output/enhanced/analysis_report.html` in browser

Expected runtime: ~5-10 minutes on a modern laptop (embedding generation is the bottleneck).

## Technical Notes

### Memory Requirements

- ~4GB RAM for embedding generation
- BERTopic model caches embeddings to avoid recomputation

### GPU Acceleration

Embeddings are generated on GPU if available via PyTorch. To check:

```python
import torch
print(torch.cuda.is_available())  # True if GPU available
```

### Model Persistence

The BERTopic model is saved to `output/enhanced/bertopic_model/` and can be reloaded:

```python
from bertopic import BERTopic
model = BERTopic.load("output/enhanced/bertopic_model")
```

## References

- [Anthropic Interviewer Dataset](https://huggingface.co/datasets/Anthropic/AnthropicInterviewer)
- [Anthropic Research Blog Post](https://www.anthropic.com/research/anthropic-interviewer)
- [BERTopic Documentation](https://maartengr.github.io/BERTopic/)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)

## License

This project is for research purposes. The underlying interview data is from Anthropic's public dataset.
