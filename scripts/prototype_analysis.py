#!/usr/bin/env python3
"""
Quick Prototype: JTBD Extraction & Pain Point Analysis
Analyzes interview transcripts to identify:
1. Jobs-To-Be-Done (JTBD) - tasks people are trying to accomplish
2. Pain points - tasks that are tedious, time-consuming, or frustrating
3. Clusters of related jobs/tasks
"""

import pandas as pd
import re
from collections import defaultdict, Counter
from pathlib import Path

# Minimal dependencies for quick prototype
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False
    print("⚠️  VADER not installed. Using basic sentiment patterns.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  sklearn not installed. Skipping clustering.")


# ============================================================================
# JTBD EXTRACTION PATTERNS
# ============================================================================

# Patterns that indicate someone describing a job/task they do
JTBD_PATTERNS = [
    # "I [verb] to/for..." patterns
    r"I (?:use|need|want|try|have to|usually|often|always|sometimes) (?:to )?(\w+(?:\s+\w+){1,8})",
    # "When I [verb]..." patterns
    r"[Ww]hen I (?:need to|want to|have to|am|'m) (\w+(?:\s+\w+){1,8})",
    # "I'm [verb]ing..." patterns
    r"I'?m (?:trying to|working on|doing|using) (\w+(?:\s+\w+){1,8})",
    # "to [verb] [object]" task patterns
    r"(?:helps? me|allows? me|lets? me|enables? me) (?:to )?(\w+(?:\s+\w+){1,6})",
    # "for [gerund]" patterns
    r"(?:use it |using it |it's good )for (\w+(?:\s+\w+){1,6})",
    # Direct task mentions
    r"(?:my (?:job|task|work|goal) is to) (\w+(?:\s+\w+){1,6})",
]

# Patterns indicating PAIN POINTS (negative sentiment around tasks)
PAIN_PATTERNS = [
    # Time-consuming indicators
    (r"takes? (?:too )?(?:much|a lot of|so much|forever|hours|ages)", "time_consuming"),
    (r"(?:spend|waste|spent|wasted) (?:too much |a lot of |so much )?time", "time_consuming"),
    (r"time[- ]consuming", "time_consuming"),

    # Tedious/boring indicators
    (r"(?:really |so |very |pretty )?(?:tedious|boring|repetitive|monotonous)", "tedious"),
    (r"(?:same thing|over and over|again and again)", "tedious"),

    # Frustration indicators
    (r"(?:frustrat|annoy|irritat|infuriat)", "frustrating"),
    (r"(?:hate|dread|can't stand|despise) (?:doing|having to|when)", "frustrating"),
    (r"(?:pain|nightmare|headache|hassle)", "frustrating"),

    # Avoidance indicators
    (r"(?:avoid|put off|procrastinat|delay)", "avoidance"),
    (r"wish I didn't have to", "avoidance"),

    # Difficulty indicators
    (r"(?:struggle|difficult|hard|tough|challenging) (?:to|with)", "difficult"),
    (r"(?:can't figure out|don't know how)", "difficult"),

    # Desire for improvement
    (r"(?:would be nice|wish|hope|want) (?:if |to |I could )", "desire_improvement"),
    (r"(?:should be|could be) (?:easier|faster|simpler|better)", "desire_improvement"),
]

# Keywords that often appear in JTBD contexts
JTBD_KEYWORDS = [
    "workflow", "process", "task", "project", "work", "job", "goal",
    "brainstorm", "create", "write", "edit", "review", "analyze",
    "research", "organize", "manage", "plan", "design", "build",
    "communicate", "collaborate", "share", "present", "report"
]


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def load_transcripts(data_dir: Path) -> pd.DataFrame:
    """Load all transcript CSVs into a single DataFrame."""
    all_transcripts = []

    for csv_file in data_dir.glob("*_transcripts.csv"):
        df = pd.read_csv(csv_file)
        # Extract category from filename
        category = csv_file.stem.replace("_transcripts", "")
        df["category"] = category
        all_transcripts.append(df)
        print(f"  Loaded {len(df)} transcripts from {csv_file.name} ({category})")

    combined = pd.concat(all_transcripts, ignore_index=True)
    print(f"\n📊 Total: {len(combined)} transcripts across {len(all_transcripts)} categories")
    return combined


def extract_user_responses(transcript: str) -> list[str]:
    """Extract only the User's responses from a transcript."""
    # Split by User: marker and extract their parts
    user_parts = re.split(r'\n(?:User|Human):\s*', transcript)
    # Skip the first part (before first User:) and clean up
    responses = []
    for part in user_parts[1:]:
        # Take text until next AI/Assistant marker
        text = re.split(r'\n(?:AI|Assistant|Claude):', part)[0].strip()
        if text and len(text) > 10:  # Skip very short responses
            responses.append(text)
    return responses


def extract_jtbd(text: str) -> list[dict]:
    """Extract Jobs-To-Be-Done from text using pattern matching."""
    jobs = []

    for pattern in JTBD_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Clean up the match
            job = match.strip().lower()
            # Filter out very short or very long matches
            if 3 <= len(job.split()) <= 10:
                jobs.append({
                    "job": job,
                    "pattern": pattern[:30] + "...",
                    "context": text[:200]
                })

    return jobs


def detect_pain_points(text: str) -> list[dict]:
    """Detect pain points using pattern matching and sentiment."""
    pain_points = []

    # Pattern-based detection
    for pattern, pain_type in PAIN_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Get surrounding context (50 chars before and after)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]

            pain_points.append({
                "type": pain_type,
                "trigger": match.group(),
                "context": context,
                "position": match.start()
            })

    return pain_points


def analyze_sentiment_vader(text: str) -> dict:
    """Analyze sentiment using VADER (if available)."""
    if not HAS_VADER:
        return {"compound": 0, "neg": 0, "neu": 1, "pos": 0}

    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)


def find_negative_sentences(text: str, threshold: float = -0.3) -> list[dict]:
    """Find sentences with negative sentiment."""
    if not HAS_VADER:
        return []

    analyzer = SentimentIntensityAnalyzer()
    negative_sentences = []

    # Split into sentences
    sentences = re.split(r'[.!?]+', text)

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:  # Skip very short sentences
            continue

        scores = analyzer.polarity_scores(sentence)
        if scores['compound'] < threshold:
            negative_sentences.append({
                "sentence": sentence,
                "sentiment_score": scores['compound'],
                "negative_score": scores['neg']
            })

    return negative_sentences


def cluster_jobs(jobs: list[str], n_clusters: int = 8) -> dict:
    """Cluster similar jobs using TF-IDF and K-means."""
    if not HAS_SKLEARN or len(jobs) < n_clusters:
        return {"error": "Not enough data or sklearn not available"}

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(jobs)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Group jobs by cluster
    clusters = defaultdict(list)
    for job, label in zip(jobs, labels):
        clusters[label].append(job)

    # Get top terms per cluster
    feature_names = vectorizer.get_feature_names_out()
    cluster_terms = {}
    for i, center in enumerate(kmeans.cluster_centers_):
        top_indices = center.argsort()[-5:][::-1]
        cluster_terms[i] = [feature_names[idx] for idx in top_indices]

    return {
        "clusters": dict(clusters),
        "cluster_terms": cluster_terms
    }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_transcripts(data_dir: Path) -> dict:
    """Run full analysis on all transcripts."""
    print("=" * 60)
    print("🔍 JTBD & Pain Point Analysis - Prototype")
    print("=" * 60)

    # Load data
    print("\n📂 Loading transcripts...")
    df = load_transcripts(data_dir)

    # Initialize results
    all_jobs = []
    all_pain_points = []
    all_negative_sentences = []
    category_stats = defaultdict(lambda: {"jobs": 0, "pain_points": 0, "negative_sentences": 0})

    print("\n🔬 Analyzing transcripts...")

    for idx, row in df.iterrows():
        transcript = row['text']
        category = row['category']
        transcript_id = row.get('transcript_id', f'unknown_{idx}')

        # Extract user responses only
        user_responses = extract_user_responses(transcript)
        user_text = " ".join(user_responses)

        # Extract JTBD
        jobs = extract_jtbd(user_text)
        for job in jobs:
            job['transcript_id'] = transcript_id
            job['category'] = category
        all_jobs.extend(jobs)
        category_stats[category]["jobs"] += len(jobs)

        # Detect pain points
        pain_points = detect_pain_points(user_text)
        for pp in pain_points:
            pp['transcript_id'] = transcript_id
            pp['category'] = category
        all_pain_points.extend(pain_points)
        category_stats[category]["pain_points"] += len(pain_points)

        # Find negative sentences
        neg_sentences = find_negative_sentences(user_text)
        for ns in neg_sentences:
            ns['transcript_id'] = transcript_id
            ns['category'] = category
        all_negative_sentences.extend(neg_sentences)
        category_stats[category]["negative_sentences"] += len(neg_sentences)

    print(f"  ✓ Extracted {len(all_jobs)} JTBD mentions")
    print(f"  ✓ Found {len(all_pain_points)} pain point indicators")
    print(f"  ✓ Identified {len(all_negative_sentences)} negative sentiment sentences")

    # Cluster jobs
    print("\n🎯 Clustering similar jobs...")
    job_texts = [j['job'] for j in all_jobs]
    cluster_results = cluster_jobs(job_texts, n_clusters=min(10, len(set(job_texts)) // 5 + 1))

    # Aggregate pain point types
    pain_type_counts = Counter(pp['type'] for pp in all_pain_points)

    return {
        "summary": {
            "total_transcripts": len(df),
            "total_jobs": len(all_jobs),
            "total_pain_points": len(all_pain_points),
            "total_negative_sentences": len(all_negative_sentences)
        },
        "category_stats": dict(category_stats),
        "jobs": all_jobs,
        "pain_points": all_pain_points,
        "negative_sentences": all_negative_sentences,
        "pain_type_counts": dict(pain_type_counts),
        "job_clusters": cluster_results
    }


def print_report(results: dict):
    """Print a formatted report of findings."""
    print("\n" + "=" * 60)
    print("📊 ANALYSIS REPORT")
    print("=" * 60)

    # Summary
    s = results['summary']
    print(f"\n📈 Summary:")
    print(f"   Transcripts analyzed: {s['total_transcripts']}")
    print(f"   JTBD mentions found: {s['total_jobs']}")
    print(f"   Pain points detected: {s['total_pain_points']}")
    print(f"   Negative sentences: {s['total_negative_sentences']}")

    # By category
    print(f"\n📁 By Category:")
    for cat, stats in results['category_stats'].items():
        print(f"   {cat.capitalize()}:")
        print(f"      Jobs: {stats['jobs']}, Pain points: {stats['pain_points']}, Negative: {stats['negative_sentences']}")

    # Pain point types
    print(f"\n😣 Pain Point Types:")
    for pain_type, count in sorted(results['pain_type_counts'].items(), key=lambda x: -x[1]):
        print(f"   {pain_type}: {count}")

    # Top negative sentences (potential undesirable tasks)
    print(f"\n🔴 Most Negative Sentences (Potential Pain Points):")
    sorted_neg = sorted(results['negative_sentences'], key=lambda x: x['sentiment_score'])[:10]
    for i, ns in enumerate(sorted_neg, 1):
        score = ns['sentiment_score']
        cat = ns['category']
        text = ns['sentence'][:100] + "..." if len(ns['sentence']) > 100 else ns['sentence']
        print(f"   {i}. [{cat}] (score: {score:.2f})")
        print(f"      \"{text}\"")

    # Job clusters
    if 'clusters' in results['job_clusters']:
        print(f"\n🎯 JTBD Clusters:")
        clusters = results['job_clusters']['clusters']
        terms = results['job_clusters']['cluster_terms']

        for cluster_id in sorted(clusters.keys()):
            jobs = clusters[cluster_id]
            top_terms = terms.get(cluster_id, [])
            print(f"\n   Cluster {cluster_id + 1} ({len(jobs)} jobs)")
            print(f"   Key themes: {', '.join(top_terms)}")
            # Show sample jobs
            for job in jobs[:3]:
                print(f"      • {job}")
            if len(jobs) > 3:
                print(f"      ... and {len(jobs) - 3} more")

    # Sample pain point contexts
    print(f"\n📍 Sample Pain Point Contexts:")
    for pain_type in ['time_consuming', 'tedious', 'frustrating']:
        matching = [pp for pp in results['pain_points'] if pp['type'] == pain_type][:2]
        if matching:
            print(f"\n   {pain_type.upper()}:")
            for pp in matching:
                print(f"      [{pp['category']}] \"...{pp['context']}...\"")


def save_results(results: dict, output_dir: Path):
    """Save results to files for further analysis."""
    output_dir.mkdir(exist_ok=True)

    # Save jobs
    jobs_df = pd.DataFrame(results['jobs'])
    jobs_df.to_csv(output_dir / "extracted_jobs.csv", index=False)

    # Save pain points
    pain_df = pd.DataFrame(results['pain_points'])
    pain_df.to_csv(output_dir / "pain_points.csv", index=False)

    # Save negative sentences
    neg_df = pd.DataFrame(results['negative_sentences'])
    neg_df.to_csv(output_dir / "negative_sentences.csv", index=False)

    print(f"\n💾 Results saved to {output_dir}/")


if __name__ == "__main__":
    # Set paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "output"

    # Run analysis
    results = analyze_transcripts(data_dir)

    # Print report
    print_report(results)

    # Save results
    save_results(results, output_dir)

    print("\n✅ Prototype analysis complete!")
