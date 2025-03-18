# agents/macro_economic_agent/processing/sentiment_analyzer.py
import json, re, time
from typing import Dict, List
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

class SentimentAnalyzer:
    PROMPT = (
        "You are a financial news sentiment classifier.\n"
        "News Title: \"{title}\"\n"
        "Respond JSON: {{\"sentiment\":\"<p/n/n>\",\"score\":<float>}}"
    )

    def __init__(self, key: str, model="models/gemini-2.5-flash-preview-04-17", pause=1.0):
        genai.configure(api_key=key)
        self.m = genai.GenerativeModel(model)
        self.pause = pause

    def _call(self, prompt: str) -> Dict[str, str | float]:
        try:
            r = self.m.generate_content(prompt)
        except ResourceExhausted:
            time.sleep(5);  # 간단 재시도 대기
            r = self.m.generate_content(prompt)

        raw = r.text or r.candidates[0].content.parts[0].text
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I | re.S)
        return json.loads(raw)

    def analyze_sentiment(self, news: List[Dict]) -> List[Dict]:
        for art in news:
            try:
                res = self._call(self.PROMPT.format(**art))
                art |= {"sentiment": res["sentiment"], "score": float(res["score"])}
            except Exception:
                art |= {"sentiment": "neutral", "score": 0.0}
            time.sleep(self.pause)
        return news
