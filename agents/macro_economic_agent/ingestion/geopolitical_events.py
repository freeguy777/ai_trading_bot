import requests
from datetime import datetime, timedelta
import spacy
import en_core_web_sm

class GeoPoliticalEventsFetcher:
    def __init__(self, config):
        self.api_key = config['NEWS_API_KEY']
        self.endpoint = "https://newsapi.org/v2/everything"
        self.default_keywords = [
            'trade war', 'economic sanction', 'military conflict',
            'presidential election', 'geopolitical tension', 'summit meeting',
            'international dispute', 'diplomatic crisis'
        ]
        # Load spaCy model once
        self.nlp = en_core_web_sm.load()
        # Known countries fallback list
        self.known_countries = [
            'United States', 'USA', 'China', 'Russia', 'Israel', 'Iran', 'Ukraine',
            'South Korea', 'North Korea', 'Canada', 'Australia', 'Ethiopia',
            'Eritrea', 'Sudan', 'Chad', 'Congo', 'Mongolia', 'Cambodia', 'Laos',
            'Myanmar'
        ]

    def fetch_events(self, query=None, days_back=30, max_results=100):
        if query is None:
            query = ' OR '.join(self.default_keywords)

        params = {
            'q': query,
            'from': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
            'sortBy': 'relevance',
            'language': 'en',
            'apiKey': self.api_key,
            'pageSize': max_results
        }

        response = requests.get(self.endpoint, params=params)
        response.raise_for_status()
        articles = response.json().get('articles', [])

        events = []
        for article in articles:
            title = article.get('title') or ''
            # Filter out video/watch titles
            if title.startswith("WATCH:"):
                continue
            event = self._extract_event(article)
            if event:
                events.append(event)

        # Sort events by date (newest first)
        events.sort(key=lambda e: e['date'] or datetime.min, reverse=True)
        return events

    def _extract_country(self, article):
        content = ' '.join(filter(None, [article.get('title'), article.get('description')]))
        doc = self.nlp(content)
        # Use spaCy NER for GPE entities
        countries = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
        if countries:
            return countries[0]
        # Fallback to known_countries list
        content_lower = content.lower()
        for country in self.known_countries:
            if country.lower() in content_lower:
                return country
        return 'Unknown'

    def _classify_event(self, article):
        text = (article.get('title') or '').lower()
        for kw in self.default_keywords:
            if kw in text:
                return kw.title()
        return 'Geopolitical Event'

    def _extract_event(self, article):
        country = self._extract_country(article)
        event_type = self._classify_event(article)
        title = article.get('title', '')
        url = article.get('url', '')

        published = article.get('publishedAt', '')
        # Parse ISO date string to datetime
        try:
            date = datetime.fromisoformat(published.replace('Z', '+00:00'))
        except Exception:
            date = None

        return {
            'country': country,
            'event': event_type,
            'title': title,
            'date': date,
            'url': url
        }


# 테스트 코드
if __name__ == "__main__":
    import os
    import sys
    #####절대 경로 임포트 사용
    current_dir = os.path.dirname(os.path.abspath(__file__))
    agents_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(agents_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from macro_economic_agent.config_loader import ConfigLoader

    config = ConfigLoader.load()
    fetcher = GeoPoliticalEventsFetcher(config)
    events = fetcher.fetch_events()
    for event in events:
        print({
            'country': event['country'],
            'event': event['event'],
            'title': event['title'],
            'date': event['date'].isoformat() if event['date'] else '',
            'url': event['url']
        })