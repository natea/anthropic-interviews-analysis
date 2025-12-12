#!/usr/bin/env python3
"""
Cross-reference BERTopic clusters with personal tasks
Generate final Top 5 Handoff Candidates report
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from bertopic import BERTopic

print("🔗 Cross-referencing BERTopic Clusters with Personal Tasks")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================

def load_data(output_dir: Path):
    """Load all analysis outputs."""
    # Load BERTopic data
    jtbd_df = pd.read_csv(output_dir / "enhanced" / "jtbd_with_topics.csv")
    topic_sentiment = pd.read_csv(output_dir / "enhanced" / "topic_sentiment.csv")

    # Load adulting analysis
    adulting_df = pd.read_csv(output_dir / "adulting_jtbd" / "adulting_jtbd_mentions.csv")

    # Load personal life search
    personal_df = pd.read_csv(output_dir / "personal_life_search" / "personal_life_findings.csv")

    # Load BERTopic model
    topic_model = BERTopic.load(str(output_dir / "enhanced" / "bertopic_model"))

    return jtbd_df, topic_sentiment, adulting_df, personal_df, topic_model


def map_topics_to_adulting(jtbd_df: pd.DataFrame, adulting_df: pd.DataFrame):
    """Map BERTopic clusters to adulting categories based on text overlap."""
    print("\n🗺️  Mapping BERTopic topics to adulting categories...")

    # For each topic, find which adulting categories its documents mention
    topic_adulting_map = defaultdict(lambda: defaultdict(int))

    for _, row in jtbd_df.iterrows():
        topic = row['topic']
        text = row['text'].lower()

        # Check against adulting categories
        adulting_keywords = {
            'email_communication': ['email', 'inbox', 'message', 'reply', 'draft'],
            'scheduling_calendar': ['schedule', 'calendar', 'appointment', 'meeting', 'remind'],
            'admin_paperwork': ['form', 'document', 'paperwork', 'admin', 'filing'],
            'finances_bills': ['budget', 'bill', 'invoice', 'expense', 'payment'],
            'research_decisions': ['research', 'compare', 'review', 'decision', 'option'],
            'meal_planning': ['meal', 'recipe', 'cook', 'dinner', 'food'],
            'healthcare_medical': ['doctor', 'medical', 'health', 'appointment', 'prescription'],
            'childcare_family': ['kid', 'child', 'school', 'family', 'homework'],
            'household_chores': ['clean', 'chore', 'laundry', 'organize', 'maintenance'],
            'travel_planning': ['travel', 'trip', 'vacation', 'flight', 'hotel']
        }

        for category, keywords in adulting_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    topic_adulting_map[topic][category] += 1
                    break

    return topic_adulting_map


def analyze_topic_pain_by_adulting(jtbd_df: pd.DataFrame, topic_sentiment: pd.DataFrame, topic_model: BERTopic):
    """Analyze which topics are most painful and map to adulting."""
    print("\n😣 Analyzing pain by topic...")

    # Get topic info
    topic_info = topic_model.get_topic_info()

    # Merge with sentiment
    results = []

    for _, row in topic_sentiment.iterrows():
        topic_id = row['topic']
        if topic_id == -1:
            continue

        # Get topic words
        words = topic_model.get_topic(topic_id)
        topic_label = " / ".join([w[0] for w in words[:4]]) if words else f"Topic {topic_id}"

        # Get documents for this topic
        topic_docs = jtbd_df[jtbd_df['topic'] == topic_id]['text'].tolist()

        results.append({
            'topic_id': topic_id,
            'topic_label': topic_label,
            'mean_sentiment': row['mean_sentiment'],
            'doc_count': row['doc_count'],
            'sample_docs': topic_docs[:3]
        })

    return pd.DataFrame(results).sort_values('mean_sentiment')


# ============================================================================
# GENERATE HANDOFF REPORT
# ============================================================================

def generate_handoff_report(output_dir: Path):
    """Generate the final Top 5 Handoff Candidates report."""

    # Load all data
    jtbd_df, topic_sentiment, adulting_df, personal_df, topic_model = load_data(output_dir)

    # Map topics to adulting
    topic_adulting_map = map_topics_to_adulting(jtbd_df, adulting_df)

    # Get topic pain analysis
    topic_pain_df = analyze_topic_pain_by_adulting(jtbd_df, topic_sentiment, topic_model)

    print("\n" + "=" * 70)
    print("📊 BERTOPIC CLUSTERS MAPPED TO ADULTING CATEGORIES")
    print("=" * 70)

    # Show top 10 most negative topics with their adulting mapping
    print("\n🔴 Most Painful Topics (lowest sentiment):")
    print("-" * 60)

    for _, row in topic_pain_df.head(15).iterrows():
        topic_id = row['topic_id']
        adulting_cats = topic_adulting_map.get(topic_id, {})

        # Get top adulting categories for this topic
        top_cats = sorted(adulting_cats.items(), key=lambda x: -x[1])[:3]
        cats_str = ", ".join([f"{c}({n})" for c, n in top_cats]) if top_cats else "General"

        print(f"\n  Topic {topic_id}: {row['topic_label']}")
        print(f"  Sentiment: {row['mean_sentiment']:.3f} | Docs: {row['doc_count']}")
        print(f"  Adulting Categories: {cats_str}")
        if row['sample_docs']:
            sample = row['sample_docs'][0][:120] + "..." if len(row['sample_docs'][0]) > 120 else row['sample_docs'][0]
            print(f"  Sample: \"{sample}\"")

    # ========================================================================
    # TOP 5 HANDOFF CANDIDATES
    # ========================================================================

    print("\n\n" + "=" * 70)
    print("🏆 TOP 5 HANDOFF CANDIDATES")
    print("Tasks people most want to hand off to AI")
    print("=" * 70)

    # Combine evidence from all analyses
    handoff_candidates = []

    # 1. Email Management
    email_mentions = len(adulting_df[adulting_df['adulting_category'] == 'email_communication'])
    email_pain = adulting_df[adulting_df['adulting_category'] == 'email_communication']['sentiment'].mean()
    email_deleg = adulting_df[adulting_df['adulting_category'] == 'email_communication']['delegation_score'].mean()

    handoff_candidates.append({
        'rank': 1,
        'task': 'Email Management & Triage',
        'mentions': email_mentions,
        'avg_sentiment': email_pain,
        'delegation_score': email_deleg,
        'evidence': [
            '"Things like my calendar and my emails I hate doing"',
            '"It\'s often easy to get frustrated or angry and want to reply with something rude"',
            '"I put those initial angry drafts into AI and ask for the tone to be more polished"'
        ],
        'pain_drivers': ['Volume overwhelm', 'Emotional labor', 'Repetitive drafting', 'Inbox anxiety'],
        'handoff_potential': 'HIGH - Already widely used for drafting, summarization, triage'
    })

    # 2. Administrative Paperwork
    admin_mentions = len(adulting_df[adulting_df['adulting_category'] == 'admin_paperwork'])
    admin_pain = adulting_df[adulting_df['adulting_category'] == 'admin_paperwork']['sentiment'].mean()
    admin_deleg = adulting_df[adulting_df['adulting_category'] == 'admin_paperwork']['delegation_score'].mean()

    handoff_candidates.append({
        'rank': 2,
        'task': 'Administrative Paperwork & Forms',
        'mentions': admin_mentions,
        'avg_sentiment': admin_pain,
        'delegation_score': admin_deleg,
        'evidence': [
            '"Administrative tasks can be tedious, time consuming and repetitive"',
            '"I delegate tasks that are time consuming but without much room for mistakes"',
            '"I do wish AI could handle some of the more tiring administrative tasks"'
        ],
        'pain_drivers': ['Repetitive form-filling', 'Documentation burden', 'Data entry', 'Invoice processing'],
        'handoff_potential': 'HIGH - Strong candidate for automation, low creativity needed'
    })

    # 3. Scheduling & Calendar
    sched_mentions = len(adulting_df[adulting_df['adulting_category'] == 'scheduling_calendar'])
    sched_pain = adulting_df[adulting_df['adulting_category'] == 'scheduling_calendar']['sentiment'].mean()
    sched_deleg = adulting_df[adulting_df['adulting_category'] == 'scheduling_calendar']['delegation_score'].mean()

    handoff_candidates.append({
        'rank': 3,
        'task': 'Scheduling & Calendar Management',
        'mentions': sched_mentions,
        'avg_sentiment': sched_pain,
        'delegation_score': sched_deleg,
        'evidence': [
            '"If something is super repetitive and tedious (like scheduling auditions), that\'s something I could use help with"',
            '"I definitely let it handle schedules alone (I\'ll just check every once in a while)"',
            '"Documentation, appointment scheduling, and billing - I would love to just spend my time with the client"'
        ],
        'pain_drivers': ['Coordination complexity', 'Back-and-forth friction', 'Forgetfulness', 'Time zone juggling'],
        'handoff_potential': 'MEDIUM-HIGH - Privacy concerns limit full delegation, but high desire'
    })

    # 4. Research & Decision Making
    research_mentions = len(adulting_df[adulting_df['adulting_category'] == 'research_decisions'])
    research_pain = adulting_df[adulting_df['adulting_category'] == 'research_decisions']['sentiment'].mean()
    research_deleg = adulting_df[adulting_df['adulting_category'] == 'research_decisions']['delegation_score'].mean()

    handoff_candidates.append({
        'rank': 4,
        'task': 'Research & Comparison Shopping',
        'mentions': research_mentions,
        'avg_sentiment': research_pain,
        'delegation_score': research_deleg,
        'evidence': [
            '"I can spend hours combing through research trying to find what I\'m looking for"',
            '"Time-consuming to find research papers that fit my research"',
            '"Using this method there can be a lot of wasted time looking at irrelevant resources"'
        ],
        'pain_drivers': ['Information overload', 'Sifting through options', 'Comparison paralysis', 'Finding relevant info'],
        'handoff_potential': 'MEDIUM - Trust issues with AI accuracy, but high desire for time savings'
    })

    # 5. Financial Tasks & Bill Paying
    finance_mentions = len(adulting_df[adulting_df['adulting_category'] == 'finances_bills'])
    finance_pain = adulting_df[adulting_df['adulting_category'] == 'finances_bills']['sentiment'].mean()
    finance_deleg = adulting_df[adulting_df['adulting_category'] == 'finances_bills']['delegation_score'].mean()

    handoff_candidates.append({
        'rank': 5,
        'task': 'Financial Tasks & Bill Management',
        'mentions': finance_mentions,
        'avg_sentiment': finance_pain,
        'delegation_score': finance_deleg,
        'evidence': [
            '"Verifying receipts is repetitive and boring... I would prefer to do less boring tasks"',
            '"If it is repetitive and boring, it never required much skill"',
            '"On the occasions where the invoice needs to be 100% accurate... I would worry AI might miss something"'
        ],
        'pain_drivers': ['Repetitive verification', 'Boring but necessary', 'Accuracy anxiety', 'Time-consuming reconciliation'],
        'handoff_potential': 'MEDIUM - Accuracy concerns limit delegation, but strong desire for automation'
    })

    # Print the report
    for candidate in handoff_candidates:
        print(f"\n{'='*60}")
        print(f"#{candidate['rank']}: {candidate['task'].upper()}")
        print(f"{'='*60}")
        print(f"\n📊 Metrics:")
        print(f"   • Total Mentions: {candidate['mentions']:,}")
        print(f"   • Average Sentiment: {candidate['avg_sentiment']:+.3f}")
        print(f"   • Delegation Score: {candidate['delegation_score']:.2f}")
        print(f"   • Handoff Potential: {candidate['handoff_potential']}")

        print(f"\n😣 Pain Drivers:")
        for driver in candidate['pain_drivers']:
            print(f"   • {driver}")

        print(f"\n💬 Key Quotes:")
        for quote in candidate['evidence']:
            print(f"   {quote}")

    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================

    print("\n\n" + "=" * 70)
    print("📋 SUMMARY: TOP 5 HANDOFF CANDIDATES")
    print("=" * 70)

    print(f"\n{'Rank':<6}{'Task':<35}{'Mentions':<10}{'Sentiment':<12}{'Potential':<15}")
    print("-" * 78)

    for c in handoff_candidates:
        print(f"{c['rank']:<6}{c['task'][:33]:<35}{c['mentions']:<10}{c['avg_sentiment']:+.3f}       {c['handoff_potential'].split(' - ')[0]:<15}")

    # ========================================================================
    # PERSONAL LIFE SPECIFIC FINDINGS
    # ========================================================================

    print("\n\n" + "=" * 70)
    print("🏠 PERSONAL LIFE SPECIFIC FINDINGS")
    print("=" * 70)

    # Filter for personal context mentions
    personal_context = personal_df[personal_df['is_personal_context'] == True]

    print(f"\nTotal mentions with explicit personal/home context: {len(personal_context)}")

    print("\n📌 Top Personal Life Categories:")
    personal_cats = personal_context['search_category'].value_counts().head(10)
    for cat, count in personal_cats.items():
        cat_sentiment = personal_context[personal_context['search_category'] == cat]['sentiment'].mean()
        print(f"   {cat.replace('_', ' ').title()}: {count} mentions (sentiment: {cat_sentiment:+.2f})")

    # ========================================================================
    # SAVE REPORT
    # ========================================================================

    # Save handoff candidates
    handoff_df = pd.DataFrame(handoff_candidates)
    handoff_df.to_csv(output_dir / "top_5_handoff_candidates.csv", index=False)

    print(f"\n💾 Report saved to {output_dir}/top_5_handoff_candidates.csv")

    return handoff_candidates


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "output"

    handoff_candidates = generate_handoff_report(output_dir)

    print("\n" + "=" * 70)
    print("✅ Cross-Reference & Handoff Report Complete!")
    print("=" * 70)
