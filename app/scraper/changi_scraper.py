import requests
from bs4 import BeautifulSoup
import json, os

WIKI_PAGES = [
    ("jewel_changi", "https://en.wikipedia.org/wiki/Jewel_Changi_Airport"),
    ("skytrain","https://en.wikipedia.org/wiki/Changi_Airport_Skytrain")
]

def scrape_wiki():
    combined = []
    for slug, url in WIKI_PAGES:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "lxml")
        # Grab all paragraphs under the content
        content_div = soup.find("div", {"class": "mw-parser-output"})
        if content_div:
            for tag in content_div.find_all("p"):
                text = tag.get_text(separator=" ", strip=True)

                # Filter out junk: citations, external links, short refs
                if text and not text.startswith("^") and len(text.split()) > 5:
                    combined.append({"source": slug, "text": text})
    os.makedirs("data", exist_ok=True)
    with open("data/wiki_content.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"Scraped {len(combined)} text entries.")

if __name__ == "__main__":
    scrape_wiki()
