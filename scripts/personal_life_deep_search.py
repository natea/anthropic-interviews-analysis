#!/usr/bin/env python3
"""
Deep Search: Personal Life & Adulting Tasks
Broader search patterns to find personal (non-work) task mentions
"""

import pandas as pd
import re
from pathlib import Path
from collections import Counter, defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print("🔍 Deep Search: Personal Life & Adulting Tasks")
print("=" * 70)

# ============================================================================
# EXPANDED PERSONAL LIFE PATTERNS
# ============================================================================

PERSONAL_LIFE_SEARCHES = {
    # Family & Home Life
    "family_coordination": [
        r"family", r"spouse", r"husband", r"wife", r"partner", r"kids?(?:'s)?",
        r"children", r"son", r"daughter", r"parent", r"mom", r"dad",
        r"household", r"home life", r"personal life", r"outside (?:of )?work"
    ],
    "scheduling_personal": [
        r"personal (?:calendar|schedule|appointment)", r"family (?:schedule|calendar)",
        r"(?:my|our) schedule", r"busy schedule", r"juggling", r"coordinate",
        r"(?:soccer|baseball|piano|dance) practice", r"extracurricular",
        r"carpool", r"pick(?:ing)? up (?:the )?kids", r"school run"
    ],
    "trip_vacation": [
        r"vacation", r"trip", r"travel(?:ing)?", r"holiday", r"getaway",
        r"flight", r"hotel", r"airbnb", r"itinerary", r"road trip",
        r"pack(?:ing)?", r"destination", r"tourist", r"sightseeing"
    ],
    "gift_shopping": [
        r"gift", r"present", r"birthday", r"christmas", r"holiday (?:shopping|gift)",
        r"anniversary", r"what to (?:buy|get) (?:for|someone)", r"surprise",
        r"wish ?list", r"registry"
    ],
    "meal_food": [
        r"meal (?:plan|prep)", r"recipe", r"cook(?:ing)?", r"dinner", r"breakfast",
        r"lunch", r"what(?:'s| to) (?:for )?(?:eat|cook|dinner)", r"grocery",
        r"shopping list", r"ingredients", r"menu", r"diet", r"nutrition"
    ],
    "health_wellness": [
        r"doctor(?:'s)?", r"dentist", r"(?:check[- ]?up|checkup)", r"appointment",
        r"health", r"medical", r"prescription", r"medication", r"symptom",
        r"exercise", r"workout", r"gym", r"fitness", r"sleep", r"therapy"
    ],
    "finances_personal": [
        r"(?:personal )?(?:budget|finance)", r"(?:pay(?:ing)?|paid) (?:the )?bill",
        r"tax(?:es)?", r"saving", r"invest(?:ment|ing)?", r"retirement",
        r"mortgage", r"rent", r"insurance", r"credit card", r"debt"
    ],
    "household_tasks": [
        r"chore", r"clean(?:ing)?", r"laundry", r"dishes", r"vacuum",
        r"tidy(?:ing)?", r"organiz(?:e|ing)", r"declutter", r"maintenance",
        r"repair", r"fix(?:ing)?", r"yard", r"garden", r"lawn"
    ],
    "childcare_education": [
        r"school", r"homework", r"summer camp", r"daycare", r"babysit",
        r"tutor", r"lesson", r"class(?:es)?", r"grade", r"report card",
        r"parent[- ]teacher", r"PTA", r"field trip"
    ],
    "life_admin": [
        r"errand", r"to[- ]?do", r"adulting", r"life admin", r"paperwork",
        r"form", r"document", r"DMV", r"license", r"registration", r"renew"
    ],
    "social_personal": [
        r"party", r"event", r"gathering", r"celebration", r"invitation",
        r"RSVP", r"birthday party", r"wedding", r"baby shower", r"reunion"
    ],
    "shopping_general": [
        r"shop(?:ping)?", r"buy(?:ing)?", r"purchase", r"order(?:ing)?",
        r"amazon", r"online (?:shop|order)", r"return", r"delivery"
    ],

    # Emotional/Desire Patterns
    "outside_work_mentions": [
        r"outside (?:of )?work", r"personal(?:ly)?", r"at home", r"in my (?:personal |free )?time",
        r"not (?:for|at) work", r"non[- ]?work", r"off the clock", r"after hours",
        r"weekend", r"free time", r"spare time", r"leisure"
    ],
    "delegation_desire": [
        r"(?:wish|want|love|hope) (?:AI|it|someone) (?:could|would|to)",
        r"(?:hand off|delegate|outsource|offload)", r"(?:don't|do not) want to",
        r"(?:hate|dread|avoid) (?:doing|having to)", r"tedious", r"boring",
        r"time[- ]consuming", r"(?:waste|spend) (?:so much |too much )?time"
    ]
}

# Context patterns that indicate personal vs work
PERSONAL_CONTEXT_INDICATORS = [
    r"(?:my|our) (?:family|home|house|kid|child|spouse|partner)",
    r"(?:at|from) home", r"personal(?:ly)?", r"outside (?:of )?work",
    r"in my (?:personal |free |spare )?time", r"on (?:the )?weekend",
    r"after (?:work|hours)", r"not for work", r"non[- ]?work"
]


def load_all_text(data_dir: Path) -> list:
    """Load all transcripts with user text extracted."""
    all_data = []
    for csv_file in data_dir.glob("*_transcripts.csv"):
        df = pd.read_csv(csv_file)
        category = csv_file.stem.replace("_transcripts", "")
        for _, row in df.iterrows():
            # Extract user responses
            transcript = row['text']
            user_parts = re.split(r'\n(?:User|Human):\s*', transcript)
            user_text = ""
            for part in user_parts[1:]:
                text = re.split(r'\n(?:AI|Assistant|Claude):', part)[0].strip()
                if text:
                    user_text += " " + text

            all_data.append({
                'id': row.get('transcript_id', f'unknown'),
                'category': category,
                'text': user_text.strip()
            })
        print(f"  📄 Loaded {len(df)} from {csv_file.name}")
    return all_data


def search_with_context(text: str, patterns: list, context_window: int = 100) -> list:
    """Search for patterns and return matches with surrounding context."""
    matches = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start = max(0, match.start() - context_window)
            end = min(len(text), match.end() + context_window)
            context = text[start:end]
            matches.append({
                'match': match.group(),
                'context': context,
                'position': match.start()
            })
    return matches


def is_personal_context(text: str) -> bool:
    """Check if the text appears to be about personal life (not work)."""
    for pattern in PERSONAL_CONTEXT_INDICATORS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def run_deep_search(data_dir: Path, output_dir: Path):
    """Run deep search for personal life tasks."""
    output_dir.mkdir(exist_ok=True)

    print("\n📂 Loading transcripts...")
    all_data = load_all_text(data_dir)
    print(f"  📊 Total: {len(all_data)} transcripts")

    analyzer = SentimentIntensityAnalyzer()
    all_findings = []
    category_stats = defaultdict(lambda: {'total': 0, 'personal_context': 0})

    print("\n🔍 Searching for personal life task mentions...")

    for item in all_data:
        text = item['text']

        for search_category, patterns in PERSONAL_LIFE_SEARCHES.items():
            matches = search_with_context(text, patterns)

            for match in matches:
                context = match['context']
                is_personal = is_personal_context(context)
                sentiment = analyzer.polarity_scores(context)['compound']

                all_findings.append({
                    'transcript_id': item['id'],
                    'respondent_category': item['category'],
                    'search_category': search_category,
                    'matched_term': match['match'],
                    'context': context,
                    'is_personal_context': is_personal,
                    'sentiment': sentiment
                })

                category_stats[search_category]['total'] += 1
                if is_personal:
                    category_stats[search_category]['personal_context'] += 1

    findings_df = pd.DataFrame(all_findings)
    print(f"  ✓ Found {len(findings_df)} total matches")

    # ========================================================================
    # REPORT
    # ========================================================================

    print("\n" + "=" * 70)
    print("📊 PERSONAL LIFE TASK MENTIONS")
    print("=" * 70)

    # Summary by category
    print("\n🏠 TASK CATEGORY FREQUENCY:")
    print("-" * 60)

    summary_data = []
    for cat, stats in sorted(category_stats.items(), key=lambda x: -x[1]['total']):
        total = stats['total']
        personal = stats['personal_context']
        pct_personal = (personal / total * 100) if total > 0 else 0

        cat_df = findings_df[findings_df['search_category'] == cat]
        avg_sentiment = cat_df['sentiment'].mean() if len(cat_df) > 0 else 0

        summary_data.append({
            'category': cat,
            'total_mentions': total,
            'personal_context': personal,
            'pct_personal': pct_personal,
            'avg_sentiment': avg_sentiment
        })

        sentiment_emoji = "😣" if avg_sentiment < 0 else "😐" if avg_sentiment < 0.2 else "😊"
        personal_flag = "🏠" if pct_personal > 10 else ""

        print(f"  {cat:25} | {total:5} mentions | {personal:4} personal ({pct_personal:4.1f}%) {personal_flag} | sentiment: {avg_sentiment:+.2f} {sentiment_emoji}")

    # Personal context findings (most interesting!)
    print("\n\n🏠 EXPLICITLY PERSONAL CONTEXT MENTIONS:")
    print("-" * 70)

    personal_findings = findings_df[findings_df['is_personal_context'] == True]
    print(f"Found {len(personal_findings)} mentions with clear personal/home context\n")

    # Group by category and show examples
    for cat in personal_findings['search_category'].unique():
        cat_personal = personal_findings[personal_findings['search_category'] == cat]
        if len(cat_personal) > 0:
            cat_display = cat.replace("_", " ").title()
            print(f"\n📌 {cat_display} ({len(cat_personal)} personal mentions):")

            # Show top examples (most negative sentiment = pain points)
            examples = cat_personal.nsmallest(3, 'sentiment')
            for _, ex in examples.iterrows():
                context = ex['context'][:200] + "..." if len(ex['context']) > 200 else ex['context']
                print(f"   [{ex['respondent_category']}] (sent: {ex['sentiment']:.2f})")
                print(f"   \"...{context}...\"")
                print()

    # Delegation desire mentions
    print("\n\n🤖 DELEGATION DESIRE MENTIONS (want AI to do this):")
    print("-" * 70)

    deleg_findings = findings_df[findings_df['search_category'] == 'delegation_desire']
    print(f"Found {len(deleg_findings)} expressions of wanting to delegate\n")

    # Most negative (most pain)
    for _, row in deleg_findings.nsmallest(15, 'sentiment').iterrows():
        context = row['context'][:180] + "..." if len(row['context']) > 180 else row['context']
        print(f"  [{row['respondent_category']}] (sent: {row['sentiment']:.2f})")
        print(f"  \"...{context}...\"")
        print()

    # Outside work mentions
    print("\n\n🏠 'OUTSIDE OF WORK' / PERSONAL TIME MENTIONS:")
    print("-" * 70)

    outside_work = findings_df[findings_df['search_category'] == 'outside_work_mentions']
    print(f"Found {len(outside_work)} mentions of personal/outside-work AI use\n")

    for _, row in outside_work.head(20).iterrows():
        context = row['context'][:200] + "..." if len(row['context']) > 200 else row['context']
        print(f"  [{row['respondent_category']}]")
        print(f"  \"...{context}...\"")
        print()

    # Save outputs
    findings_df.to_csv(output_dir / "personal_life_findings.csv", index=False)
    pd.DataFrame(summary_data).to_csv(output_dir / "personal_life_summary.csv", index=False)
    personal_findings.to_csv(output_dir / "personal_context_only.csv", index=False)

    print(f"\n💾 Results saved to {output_dir}/")

    return findings_df, summary_data


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "output" / "personal_life_search"

    findings_df, summary_data = run_deep_search(data_dir, output_dir)

    print("\n" + "=" * 70)
    print("✅ Personal Life Deep Search Complete!")
    print("=" * 70)
