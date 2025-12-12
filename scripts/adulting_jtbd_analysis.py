#!/usr/bin/env python3
"""
Targeted JTBD Analysis: "Adulting" and Life Management Tasks
Extracts explicit mentions of personal life tasks people want AI to handle.
"""

import pandas as pd
import re
from pathlib import Path
from collections import Counter, defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print("🎯 Targeted JTBD Analysis: Adulting & Life Management Tasks")
print("=" * 70)

# ============================================================================
# ADULTING TASK CATEGORIES
# ============================================================================

ADULTING_CATEGORIES = {
    "scheduling_calendar": {
        "keywords": [
            r"schedul", r"calendar", r"appointment", r"meeting", r"booking",
            r"remind", r"plan(?:ning)? (?:my |the )?(?:day|week|month)",
            r"time management", r"organize my time", r"busy schedule"
        ],
        "examples": ["family schedules", "appointments", "calendar management"]
    },
    "travel_planning": {
        "keywords": [
            r"trip planning", r"travel", r"vacation", r"flight", r"hotel",
            r"itinerar", r"book(?:ing)? (?:a )?(?:trip|flight|hotel)",
            r"road trip", r"destination", r"pack(?:ing)? (?:for|list)"
        ],
        "examples": ["trip planning", "vacation research", "travel booking"]
    },
    "gift_buying": {
        "keywords": [
            r"gift", r"present", r"birthday", r"christmas", r"holiday shopping",
            r"what to (?:buy|get)", r"shopping for (?:someone|people)",
            r"gift idea", r"surprise"
        ],
        "examples": ["gift buying", "present ideas", "holiday shopping"]
    },
    "household_chores": {
        "keywords": [
            r"chore", r"clean(?:ing)?", r"laundry", r"dishes", r"vacuum",
            r"house(?:hold)? (?:work|task|chore)", r"tidying", r"organizing home",
            r"declutter", r"home maintenance", r"repair", r"fix(?:ing)? (?:things|stuff)"
        ],
        "examples": ["house chores", "cleaning", "home maintenance"]
    },
    "finances_bills": {
        "keywords": [
            r"financ", r"budget", r"bill", r"pay(?:ing|ment)", r"tax",
            r"expense", r"money", r"invest", r"saving", r"bank",
            r"credit card", r"mortgage", r"loan", r"debt"
        ],
        "examples": ["paying bills", "budgeting", "financial planning"]
    },
    "meal_planning": {
        "keywords": [
            r"meal (?:plan|prep)", r"recipe", r"cook(?:ing)?", r"dinner",
            r"breakfast", r"lunch", r"what to (?:eat|cook|make)",
            r"menu", r"food prep", r"nutrition", r"diet"
        ],
        "examples": ["meal planning", "recipe finding", "cooking"]
    },
    "grocery_shopping": {
        "keywords": [
            r"grocer", r"shopping list", r"supermarket", r"food shopping",
            r"buy(?:ing)? (?:food|groceries)", r"pantry", r"ingredients"
        ],
        "examples": ["grocery shopping", "shopping lists"]
    },
    "email_communication": {
        "keywords": [
            r"email", r"inbox", r"message", r"respond(?:ing)?", r"reply",
            r"draft(?:ing)?", r"correspondence", r"triage", r"sort(?:ing)? (?:email|mail)"
        ],
        "examples": ["email triage", "inbox management", "drafting responses"]
    },
    "healthcare_medical": {
        "keywords": [
            r"doctor", r"dentist", r"appointment", r"check[- ]?up", r"medical",
            r"health(?:care)?", r"prescri", r"medic(?:ation|ine)", r"symptom",
            r"diagnos", r"therapy", r"insurance claim"
        ],
        "examples": ["doctor appointments", "medical research", "health tracking"]
    },
    "childcare_family": {
        "keywords": [
            r"kid", r"child", r"parent", r"school", r"homework", r"summer camp",
            r"daycare", r"babysit", r"family", r"carpool", r"extracurricular",
            r"soccer practice", r"piano lesson", r"playdate"
        ],
        "examples": ["summer camps", "kids activities", "school research"]
    },
    "admin_paperwork": {
        "keywords": [
            r"paperwork", r"form", r"document", r"filing", r"admin(?:istrat)",
            r"bureaucra", r"application", r"registration", r"renew(?:al)?",
            r"license", r"permit", r"DMV", r"government"
        ],
        "examples": ["paperwork", "forms", "administrative tasks"]
    },
    "research_decisions": {
        "keywords": [
            r"research(?:ing)?", r"compar(?:e|ing)", r"review", r"find(?:ing)? (?:the best|a good)",
            r"looking for", r"decision", r"which (?:one|to choose)",
            r"recommend", r"option", r"alternative"
        ],
        "examples": ["product research", "comparing options", "decision making"]
    },
    "home_services": {
        "keywords": [
            r"plumber", r"electrician", r"contractor", r"handyman", r"repair",
            r"service provider", r"quote", r"estimate", r"home improvement",
            r"renovation", r"landscap"
        ],
        "examples": ["finding contractors", "home repairs", "service booking"]
    },
    "personal_errands": {
        "keywords": [
            r"errand", r"to[- ]?do list", r"task list", r"chore list",
            r"things to do", r"daily task", r"routine"
        ],
        "examples": ["errands", "to-do management", "daily tasks"]
    },
    "social_events": {
        "keywords": [
            r"party", r"event planning", r"invitation", r"RSVP", r"gathering",
            r"celebration", r"birthday party", r"wedding", r"reunion"
        ],
        "examples": ["party planning", "event organization", "invitations"]
    }
}

# Pain/delegation indicators
DELEGATION_INDICATORS = [
    # Explicit delegation desire
    (r"(?:wish|want|love|hope) (?:AI|it) (?:could|would|to) (?:do|handle|take care of)", "explicit_wish", 3),
    (r"(?:hand off|delegate|offload|outsource) (?:to AI|this)", "explicit_delegation", 3),
    (r"AI (?:could|should|can) (?:just )?(?:do|handle|take over)", "ai_capability", 2),

    # Time/effort complaints
    (r"(?:takes?|spend|waste) (?:so much|too much|a lot of) time", "time_waste", 2),
    (r"time[- ]consuming", "time_consuming", 2),
    (r"(?:hours|forever) (?:doing|on|spent)", "hours_spent", 2),

    # Negative emotions
    (r"(?:hate|dread|can't stand|despise) (?:doing|having to)", "hate_task", 3),
    (r"(?:tedious|boring|repetitive|monotonous|mundane)", "tedious", 2),
    (r"(?:frustrat|annoy|irritat)", "frustrating", 2),
    (r"(?:pain|nightmare|headache|hassle)", "painful", 2),

    # Avoidance behavior
    (r"(?:avoid|put off|procrastinat|delay)", "avoidance", 2),
    (r"(?:don't want to|rather not|wish I didn't)", "dont_want", 2),

    # Low value perception
    (r"(?:mindless|brainless|no[- ]brainer|simple|trivial)", "low_value", 1),
    (r"(?:anyone could|doesn't require|no skill)", "low_skill", 1),

    # Already using AI for it
    (r"(?:use|using) (?:AI|ChatGPT|Claude) (?:for|to help with)", "already_using", 1),
    (r"(?:AI|it) (?:helps?|assists?) (?:me )?(?:with|to)", "ai_helps", 1),
]


# ============================================================================
# DATA LOADING & EXTRACTION
# ============================================================================

def load_transcripts(data_dir: Path) -> pd.DataFrame:
    """Load all transcripts."""
    all_data = []
    for csv_file in data_dir.glob("*_transcripts.csv"):
        df = pd.read_csv(csv_file)
        df["category"] = csv_file.stem.replace("_transcripts", "")
        all_data.append(df)
        print(f"  📄 Loaded {len(df)} from {csv_file.name}")
    return pd.concat(all_data, ignore_index=True)


def extract_user_text(transcript: str) -> str:
    """Extract only user responses."""
    user_parts = re.split(r'\n(?:User|Human):\s*', transcript)
    responses = []
    for part in user_parts[1:]:
        text = re.split(r'\n(?:AI|Assistant|Claude):', part)[0].strip()
        if text and len(text) > 10:
            responses.append(text)
    return " ".join(responses)


def find_adulting_mentions(text: str, category: str, keywords: list) -> list:
    """Find mentions of adulting tasks in text."""
    mentions = []
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:
            continue

        for keyword in keywords:
            if re.search(keyword, sentence, re.IGNORECASE):
                mentions.append({
                    "sentence": sentence,
                    "category": category,
                    "matched_keyword": keyword
                })
                break  # Only match once per sentence

    return mentions


def calculate_delegation_score(text: str) -> tuple:
    """Calculate how much someone wants to delegate this task."""
    score = 0
    triggers = []

    for pattern, indicator_type, weight in DELEGATION_INDICATORS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            score += weight * len(matches)
            triggers.append(indicator_type)

    return score, triggers


def analyze_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> float:
    """Get sentiment score for text."""
    return analyzer.polarity_scores(text)['compound']


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_analysis(data_dir: Path, output_dir: Path):
    """Run the adulting JTBD analysis."""
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("\n📂 Loading transcripts...")
    df = load_transcripts(data_dir)

    # Initialize
    analyzer = SentimentIntensityAnalyzer()
    all_mentions = []
    category_counts = defaultdict(int)

    print("\n🔍 Scanning for adulting/life management tasks...")

    # Process each transcript
    for idx, row in df.iterrows():
        user_text = extract_user_text(row['text'])
        respondent_category = row['category']
        transcript_id = row.get('transcript_id', f'transcript_{idx}')

        # Search for each adulting category
        for adulting_cat, config in ADULTING_CATEGORIES.items():
            mentions = find_adulting_mentions(user_text, adulting_cat, config['keywords'])

            for mention in mentions:
                sentence = mention['sentence']
                delegation_score, triggers = calculate_delegation_score(sentence)
                sentiment = analyze_sentiment(sentence, analyzer)

                all_mentions.append({
                    'transcript_id': transcript_id,
                    'respondent_category': respondent_category,
                    'adulting_category': adulting_cat,
                    'sentence': sentence,
                    'matched_keyword': mention['matched_keyword'],
                    'delegation_score': delegation_score,
                    'delegation_triggers': ", ".join(triggers) if triggers else "",
                    'sentiment': sentiment,
                    'examples': ", ".join(config['examples'])
                })

                category_counts[adulting_cat] += 1

    mentions_df = pd.DataFrame(all_mentions)
    print(f"  ✓ Found {len(mentions_df)} mentions across {len(category_counts)} categories")

    # ========================================================================
    # ANALYSIS & REPORTING
    # ========================================================================

    print("\n" + "=" * 70)
    print("📊 ADULTING JTBD FREQUENCY ANALYSIS")
    print("=" * 70)

    # 1. Overall frequency by category
    print("\n🏆 MOST FREQUENT ADULTING TASKS (by mention count):")
    print("-" * 50)

    freq_df = mentions_df.groupby('adulting_category').agg({
        'sentence': 'count',
        'delegation_score': 'mean',
        'sentiment': 'mean'
    }).rename(columns={'sentence': 'count'})
    freq_df = freq_df.sort_values('count', ascending=False)

    for i, (cat, row) in enumerate(freq_df.iterrows(), 1):
        sentiment_emoji = "😣" if row['sentiment'] < 0 else "😐" if row['sentiment'] < 0.2 else "😊"
        deleg_emoji = "🤖" if row['delegation_score'] > 1 else ""

        cat_display = cat.replace("_", " ").title()
        print(f"  {i:2}. {cat_display:25} | {int(row['count']):4} mentions | "
              f"sentiment: {row['sentiment']:+.2f} {sentiment_emoji} | "
              f"delegation: {row['delegation_score']:.1f} {deleg_emoji}")

    # 2. Tasks people MOST want to hand off (high delegation score + negative sentiment)
    print("\n\n🤖 TASKS PEOPLE MOST WANT TO DELEGATE TO AI:")
    print("   (Ranked by delegation desire + negative sentiment)")
    print("-" * 70)

    # Calculate "handoff score" = delegation_score - sentiment (lower sentiment = more pain)
    mentions_df['handoff_score'] = mentions_df['delegation_score'] - mentions_df['sentiment']

    handoff_df = mentions_df.groupby('adulting_category').agg({
        'handoff_score': 'mean',
        'delegation_score': 'mean',
        'sentiment': 'mean',
        'sentence': 'count'
    }).rename(columns={'sentence': 'count'})
    handoff_df = handoff_df.sort_values('handoff_score', ascending=False)

    for i, (cat, row) in enumerate(handoff_df.head(10).iterrows(), 1):
        cat_display = cat.replace("_", " ").title()
        print(f"  {i:2}. {cat_display:25}")
        print(f"      Handoff Score: {row['handoff_score']:.2f} | "
              f"Mentions: {int(row['count'])} | Sentiment: {row['sentiment']:+.2f}")

        # Show example quotes
        examples = mentions_df[
            (mentions_df['adulting_category'] == cat) &
            (mentions_df['delegation_score'] > 0)
        ].nlargest(2, 'delegation_score')

        for _, ex in examples.iterrows():
            quote = ex['sentence'][:120] + "..." if len(ex['sentence']) > 120 else ex['sentence']
            print(f"      💬 \"{quote}\"")
        print()

    # 3. Most painful specific mentions
    print("\n\n😫 MOST PAINFUL TASK MENTIONS (lowest sentiment + delegation desire):")
    print("-" * 70)

    painful = mentions_df[mentions_df['delegation_score'] > 0].nsmallest(15, 'sentiment')

    for i, (_, row) in enumerate(painful.iterrows(), 1):
        cat_display = row['adulting_category'].replace("_", " ").title()
        quote = row['sentence'][:150] + "..." if len(row['sentence']) > 150 else row['sentence']
        print(f"  {i:2}. [{cat_display}] (sentiment: {row['sentiment']:.2f})")
        print(f"      \"{quote}\"")
        if row['delegation_triggers']:
            print(f"      Triggers: {row['delegation_triggers']}")
        print()

    # 4. Explicit "wish AI would do this" quotes
    print("\n\n✨ EXPLICIT 'WISH AI WOULD DO THIS' QUOTES:")
    print("-" * 70)

    explicit_wishes = mentions_df[
        mentions_df['delegation_triggers'].str.contains('explicit_wish|explicit_delegation|hate_task', na=False)
    ]

    for i, (_, row) in enumerate(explicit_wishes.head(20).iterrows(), 1):
        cat_display = row['adulting_category'].replace("_", " ").title()
        quote = row['sentence'][:180] + "..." if len(row['sentence']) > 180 else row['sentence']
        print(f"  {i:2}. [{cat_display}]")
        print(f"      \"{quote}\"")
        print()

    # 5. Category breakdown by respondent type
    print("\n\n👥 ADULTING TASKS BY RESPONDENT TYPE:")
    print("-" * 70)

    crosstab = pd.crosstab(
        mentions_df['adulting_category'],
        mentions_df['respondent_category'],
        margins=True
    )
    crosstab = crosstab.sort_values('All', ascending=False)
    print(crosstab.to_string())

    # ========================================================================
    # SAVE OUTPUTS
    # ========================================================================

    # Save full data
    mentions_df.to_csv(output_dir / "adulting_jtbd_mentions.csv", index=False)

    # Save summary
    summary_df = freq_df.reset_index()
    summary_df.columns = ['category', 'mention_count', 'avg_delegation_score', 'avg_sentiment']
    summary_df['handoff_score'] = handoff_df['handoff_score'].values
    summary_df = summary_df.sort_values('handoff_score', ascending=False)
    summary_df.to_csv(output_dir / "adulting_jtbd_summary.csv", index=False)

    # Save top quotes for each category
    top_quotes = []
    for cat in ADULTING_CATEGORIES.keys():
        cat_mentions = mentions_df[mentions_df['adulting_category'] == cat]
        if len(cat_mentions) > 0:
            top = cat_mentions.nlargest(5, 'handoff_score')
            for _, row in top.iterrows():
                top_quotes.append({
                    'category': cat,
                    'quote': row['sentence'],
                    'sentiment': row['sentiment'],
                    'delegation_score': row['delegation_score'],
                    'triggers': row['delegation_triggers']
                })

    pd.DataFrame(top_quotes).to_csv(output_dir / "top_handoff_quotes.csv", index=False)

    print(f"\n💾 Results saved to {output_dir}/")
    print(f"   - adulting_jtbd_mentions.csv ({len(mentions_df)} rows)")
    print(f"   - adulting_jtbd_summary.csv")
    print(f"   - top_handoff_quotes.csv")

    return mentions_df, summary_df


# ============================================================================
# SPECIFIC TASK DEEP DIVE
# ============================================================================

def search_specific_tasks(df: pd.DataFrame, data_dir: Path):
    """Search for very specific tasks mentioned in the prompt."""

    print("\n\n" + "=" * 70)
    print("🔎 SEARCHING FOR SPECIFIC TASKS YOU MENTIONED")
    print("=" * 70)

    specific_searches = {
        "family_schedules": [r"family schedul", r"kid.*schedul", r"family calendar", r"coordinate.*family"],
        "trip_planning": [r"trip plan", r"vacation plan", r"travel plan", r"plan.*trip", r"itinerar"],
        "gift_buying": [r"gift", r"present", r"birthday.*buy", r"christmas.*shop", r"what to get"],
        "house_chores": [r"house ?chore", r"clean(?:ing)? (?:the )?house", r"laundry", r"dishes"],
        "finances_bills": [r"pay.*bill", r"budget", r"financ", r"tax", r"expense"],
        "meal_planning": [r"meal plan", r"what.*(?:cook|eat|dinner)", r"recipe", r"meal prep"],
        "grocery_shopping": [r"grocer", r"shopping list", r"food shop"],
        "email_triage": [r"email.*triage", r"inbox", r"sort.*email", r"email.*manage"],
        "doctor_dentist": [r"doctor.*appoint", r"dentist", r"check[- ]?up", r"medical.*appoint"],
        "paying_bills": [r"pay(?:ing)? (?:the )?bill", r"bill pay"],
        "summer_camps": [r"summer camp", r"camp.*kid", r"kid.*camp"],
        "adulting_general": [r"adult(?:ing)?", r"grown[- ]?up.*task", r"life admin"]
    }

    # Load raw transcripts for full-text search
    all_text = []
    for csv_file in data_dir.glob("*_transcripts.csv"):
        df_raw = pd.read_csv(csv_file)
        for _, row in df_raw.iterrows():
            user_text = extract_user_text(row['text'])
            all_text.append({
                'text': user_text,
                'category': csv_file.stem.replace("_transcripts", ""),
                'id': row.get('transcript_id', 'unknown')
            })

    for task_name, patterns in specific_searches.items():
        matches = []
        for item in all_text:
            for pattern in patterns:
                found = re.findall(r'.{0,60}' + pattern + r'.{0,60}', item['text'], re.IGNORECASE)
                for f in found:
                    matches.append({
                        'context': f.strip(),
                        'category': item['category']
                    })

        task_display = task_name.replace("_", " ").title()
        print(f"\n📌 {task_display}: {len(matches)} mentions")

        if matches:
            # Show up to 5 examples
            for m in matches[:5]:
                print(f"   [{m['category']}] \"...{m['context']}...\"")
        else:
            print("   (No explicit mentions found)")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "output" / "adulting_jtbd"

    mentions_df, summary_df = run_analysis(data_dir, output_dir)

    # Also search for the very specific tasks mentioned
    search_specific_tasks(mentions_df, data_dir)

    print("\n" + "=" * 70)
    print("✅ Adulting JTBD Analysis Complete!")
    print("=" * 70)
