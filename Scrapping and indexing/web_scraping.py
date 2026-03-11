import requests
from bs4 import BeautifulSoup
import json

# Disguise the scraper as a standard web browser to avoid getting blocked
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# --- 1. Scrape recipe URLs from sitemaps ---
urls = []
for i in range(1, 17):
    page = requests.get(f'https://www.atyabtabkha.com/wp-sitemap-posts-recipe-{i}.xml', headers=headers)
    soup = BeautifulSoup(page.content, features="xml")
    
    # Extract all <loc> tags which contain the page URLs
    loc_tags = soup.find_all('loc')
    for loc in loc_tags:
        urls.append(loc.text)

    print(f"Sitemap {i} done. Total URLs gathered: {len(urls)}")

# --- 2. Scrape content from each recipe page ---
all_recipes = []

for url in urls:
    try:
        page = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(page.content, 'html.parser')

        # Extract Title
        section1 = soup.find('section', {'class': 'section-secondary'})
        title = section1.find('h1').get_text(strip=True)

        # Extract Ingredients
        section2 = soup.find('section', {'class': 'section section--alt'})
        ul2 = section2.find('ul', {'class': 'recipe-list-products'})
        ingredients = '\n'.join([li.get_text(strip=True) for li in ul2.find_all('li')])

        # Extract Steps
        section3 = soup.find('section', {'class': 'section section--alt'})
        ul3 = section3.find('ul', {'class': 'list-recipe'})
        steps_raw = '\n'.join([li.get_text(strip=True) for li in ul3.find_all('li')])
        
        # Clean up empty lines in steps
        steps = "\n".join([line for line in steps_raw.splitlines() if line.strip()])

        # Combine into a single text block for the RAG embedding model
        rag_text = f"عنوان الوصفة: {title}\nالمكونات:\n{ingredients}\nخطوات التحضير:\n{steps}"

        # Structure as a dictionary
        recipe_data = {
            "url": url,
            "title": title,
            "ingredients": ingredients,
            "steps": steps,
            "text_for_embedding": rag_text 
        }
        
        all_recipes.append(recipe_data)
        print(f"Scraped successfully: {title}")

    except Exception as e:
        print(f"Failed to scrape {url}: {e}")

# --- 3. Save as JSONL ---
with open('rag_recipes_dataset.jsonl', 'w', encoding='utf-8') as f:
    for recipe in all_recipes:
        # ensure_ascii=False keeps the Arabic characters intact
        json_record = json.dumps(recipe, ensure_ascii=False)
        f.write(json_record + '\n')

print(f"Saved {len(all_recipes)} recipes for RAG ingestion!")