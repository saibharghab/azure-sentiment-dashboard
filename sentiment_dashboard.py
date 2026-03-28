"""
Azure AI Real-Time Sentiment Analysis Dashboard
================================================
Uses Azure Text Analytics API to process text records,
extract sentiment, key phrases, and named entities,
then visualizes results in an interactive dashboard.

Requirements:
    pip install azure-ai-textanalytics pandas matplotlib seaborn flask python-dotenv

Setup:
    Create a .env file with:
        AZURE_LANGUAGE_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
        AZURE_LANGUAGE_KEY=<your-key>
"""

import os
import json
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dotenv import load_dotenv
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from flask import Flask, request, jsonify, render_template_string

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
KEY = os.getenv("AZURE_LANGUAGE_KEY")
BATCH_SIZE = 10          # Azure Text Analytics max batch size
RATE_LIMIT_DELAY = 0.5   # seconds between batches


# ── Azure Client ──────────────────────────────────────────────────────────────

def get_client() -> TextAnalyticsClient:
    """Authenticate and return a TextAnalyticsClient."""
    if not ENDPOINT or not KEY:
        raise EnvironmentError(
            "Missing AZURE_LANGUAGE_ENDPOINT or AZURE_LANGUAGE_KEY in environment."
        )
    credential = AzureKeyCredential(KEY)
    return TextAnalyticsClient(endpoint=ENDPOINT, credential=credential)


# ── Analysis Pipeline ─────────────────────────────────────────────────────────

def analyze_sentiment_batch(client: TextAnalyticsClient, texts: list[str]) -> list[dict]:
    """
    Analyze a batch of texts for sentiment, key phrases, and entities.
    Returns a list of result dicts.
    """
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        logger.info(f"Processing batch {i // BATCH_SIZE + 1} ({len(batch)} records)...")

        try:
            # Sentiment + opinion mining
            sentiment_response = client.analyze_sentiment(
                batch, show_opinion_mining=True
            )
            # Key phrases
            key_phrase_response = client.extract_key_phrases(batch)
            # Named entities
            entity_response = client.recognize_entities(batch)

            for idx, (sent, kp, ent) in enumerate(
                zip(sentiment_response, key_phrase_response, entity_response)
            ):
                record = {
                    "text": batch[idx],
                    "sentiment": sent.sentiment if not sent.is_error else "error",
                    "confidence_positive": sent.confidence_scores.positive if not sent.is_error else 0,
                    "confidence_neutral": sent.confidence_scores.neutral if not sent.is_error else 0,
                    "confidence_negative": sent.confidence_scores.negative if not sent.is_error else 0,
                    "key_phrases": kp.key_phrases if not kp.is_error else [],
                    "entities": [
                        {"text": e.text, "category": e.category, "confidence": e.confidence_score}
                        for e in ent.entities
                    ] if not ent.is_error else [],
                    "processed_at": datetime.utcnow().isoformat(),
                }
                results.append(record)

        except Exception as e:
            logger.error(f"Batch {i // BATCH_SIZE + 1} failed: {e}")
            for text in batch:
                results.append({"text": text, "sentiment": "error", "error": str(e)})

        time.sleep(RATE_LIMIT_DELAY)

    return results


def process_file(filepath: str) -> pd.DataFrame:
    """
    Read a text file (one record per line) or CSV (column named 'text'),
    run the full analysis pipeline, and return results as a DataFrame.
    """
    if filepath.endswith(".csv"):
        df_input = pd.read_csv(filepath)
        if "text" not in df_input.columns:
            raise ValueError("CSV must contain a 'text' column.")
        texts = df_input["text"].dropna().tolist()
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(texts)} records from {filepath}")
    client = get_client()
    results = analyze_sentiment_batch(client, texts)
    return pd.DataFrame(results)


# ── Reporting & Visualization ─────────────────────────────────────────────────

def generate_report(df: pd.DataFrame, output_dir: str = "output") -> None:
    """
    Save results to CSV and generate summary charts.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save raw results
    csv_path = os.path.join(output_dir, "sentiment_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # ── Chart 1: Sentiment distribution ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Sentiment Analysis Dashboard", fontsize=16, fontweight="bold")

    sentiment_counts = df["sentiment"].value_counts()
    colors = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c", "error": "#bdc3c7"}
    bar_colors = [colors.get(s, "#bdc3c7") for s in sentiment_counts.index]

    axes[0].bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors, edgecolor="white")
    axes[0].set_title("Sentiment Distribution")
    axes[0].set_ylabel("Record Count")
    for bar, count in zip(axes[0].patches, sentiment_counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count), ha="center", va="bottom", fontweight="bold"
        )

    # ── Chart 2: Confidence score distributions ───────────────────────────────
    score_cols = ["confidence_positive", "confidence_neutral", "confidence_negative"]
    df_scores = df[score_cols].dropna()
    axes[1].boxplot(
        [df_scores[c] for c in score_cols],
        labels=["Positive", "Neutral", "Negative"],
        patch_artist=True,
        boxprops=dict(facecolor="#3498db", alpha=0.6),
    )
    axes[1].set_title("Confidence Score Distributions")
    axes[1].set_ylabel("Confidence Score")
    axes[1].set_ylim(0, 1)

    # ── Chart 3: Top key phrases ──────────────────────────────────────────────
    all_phrases = []
    for phrases in df["key_phrases"].dropna():
        if isinstance(phrases, list):
            all_phrases.extend(phrases)
        elif isinstance(phrases, str):
            try:
                all_phrases.extend(json.loads(phrases))
            except Exception:
                pass

    if all_phrases:
        phrase_series = pd.Series(all_phrases).value_counts().head(15)
        axes[2].barh(phrase_series.index[::-1], phrase_series.values[::-1], color="#9b59b6")
        axes[2].set_title("Top 15 Key Phrases")
        axes[2].set_xlabel("Frequency")
    else:
        axes[2].text(0.5, 0.5, "No key phrases found", ha="center", va="center")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "sentiment_dashboard.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Dashboard chart saved to {chart_path}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    total = len(df)
    success = df[df["sentiment"] != "error"]
    print("\n" + "=" * 50)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total records processed : {total}")
    print(f"Successfully analyzed   : {len(success)}")
    print(f"Errors                  : {total - len(success)}")
    print(f"\nSentiment breakdown:")
    for sentiment, count in sentiment_counts.items():
        pct = count / total * 100
        print(f"  {sentiment:<12} {count:>5} ({pct:.1f}%)")
    if not success.empty:
        avg_pos = success["confidence_positive"].mean()
        print(f"\nAverage positive confidence: {avg_pos:.3f}")
    print("=" * 50)


# ── Flask API (optional) ──────────────────────────────────────────────────────

app = Flask(__name__)
_client = None

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head><title>Sentiment Dashboard</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }
  textarea { width: 100%; height: 120px; font-size: 14px; padding: 8px; }
  button { background: #3498db; color: white; border: none; padding: 10px 24px;
           font-size: 15px; border-radius: 4px; cursor: pointer; margin-top: 8px; }
  button:hover { background: #2980b9; }
  #result { margin-top: 24px; background: #f8f9fa; padding: 16px; border-radius: 6px; }
  .positive { color: #27ae60; font-weight: bold; }
  .negative { color: #e74c3c; font-weight: bold; }
  .neutral  { color: #7f8c8d; font-weight: bold; }
</style></head>
<body>
<h2>Azure AI Sentiment Analysis</h2>
<p>Enter text records (one per line) to analyze sentiment.</p>
<textarea id="inputText" placeholder="Enter text here, one record per line..."></textarea><br>
<button onclick="analyze()">Analyze</button>
<div id="result"></div>
<script>
async function analyze() {
  const texts = document.getElementById('inputText').value
    .split('\\n').map(t => t.trim()).filter(t => t.length > 0);
  if (!texts.length) { alert('Please enter some text.'); return; }
  document.getElementById('result').innerHTML = '<p>Analyzing...</p>';
  const resp = await fetch('/analyze', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ texts })
  });
  const data = await resp.json();
  if (data.error) { document.getElementById('result').innerHTML = '<p>Error: ' + data.error + '</p>'; return; }
  let html = '<h3>Results</h3>';
  data.results.forEach((r, i) => {
    const cls = r.sentiment;
    html += `<div style="margin-bottom:12px; padding:10px; border-left: 4px solid #3498db;">
      <strong>Record ${i+1}:</strong> ${r.text}<br>
      Sentiment: <span class="${cls}">${r.sentiment.toUpperCase()}</span>
      (+ ${(r.confidence_positive*100).toFixed(1)}% / 
       ~ ${(r.confidence_neutral*100).toFixed(1)}% / 
       - ${(r.confidence_negative*100).toFixed(1)}%)<br>
      Key Phrases: ${r.key_phrases.join(', ') || 'None'}
    </div>`;
  });
  document.getElementById('result').innerHTML = html;
}
</script>
</body></html>
"""

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route("/analyze", methods=["POST"])
def analyze():
    global _client
    try:
        data = request.get_json()
        texts = data.get("texts", [])
        if not texts:
            return jsonify({"error": "No texts provided"}), 400
        if _client is None:
            _client = get_client()
        results = analyze_sentiment_batch(_client, texts)
        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # CLI mode: python sentiment_dashboard.py data.csv
        input_path = sys.argv[1]
        df_results = process_file(input_path)
        generate_report(df_results)
    else:
        # Web dashboard mode
        print("Starting Flask dashboard at http://localhost:5000")
        print("Pass a file path as argument to run in CLI mode.")
        app.run(debug=True, port=5000)
