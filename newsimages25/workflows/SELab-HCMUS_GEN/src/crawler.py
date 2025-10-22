import os
import sys
# export the python path 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import csv
import json
import argparse
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from tqdm import tqdm

# Import configuration
from config import INPUT_CSV, OUTPUT_DIR, CRAWLER_TIMEOUT, MAX_RETRIES


def parse_csv_to_dict_list(file_path):
    """Parse CSV file to list of dictionaries."""
    data_dicts = []
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_dicts.append(row)
    return data_dicts


def get_raw_html(url):
    """Fetch raw HTML from URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    response = requests.get(url, headers=headers, timeout=CRAWLER_TIMEOUT)
    response.raise_for_status()
    return response.text


def extract_open_graph(html):
    """Extract Open Graph metadata from HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    og_data = {}
    for tag in soup.find_all("meta"):
        prop = tag.get("property")
        if prop and prop.startswith("og:"):
            content = tag.get("content", "")
            og_data[prop] = content
    return og_data


def sanitize_html(html):
    """Remove NULL bytes and control characters from HTML."""
    if html is None:
        return None
    html = html.replace('\x00', '')
    html = re.sub(r'[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]', '', html)
    return html


def extract_main_text_from_html(html):
    """Extract main text content from HTML."""
    if html is None:
        return ""
    html = sanitize_html(html)
    if not html:
        return ""
    article = Article('')
    article.set_html(html)
    try:
        article.parse()
        article.nlp()
    except Exception:
        pass
    return article.text or ""


def crawl_single_article(url):
    """Crawl a single article with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            html = get_raw_html(url)
            return html, None
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                continue
            else:
                return None, str(e)


def enrich_data_with_crawled_content(data_dict, crawled_html_list):
    """Enrich data with crawled HTML content."""
    enriched = []
    for item, html in zip(data_dict, crawled_html_list):
        if html is None:
            enriched.append({
                'article_id': item.get('article_id', ''),
                'title': item.get('article_title', ''),
                'url': item.get('article_url', ''),
                'article_img': item['image_id'] + '.jpg' if 'image_id' in item and item['image_id'] else '',
                'main_text': '',
                'og_data': {}
            })
            continue
        main_text = extract_main_text_from_html(html)
        og_data = extract_open_graph(html)
        enriched_item = {
            'article_id': item.get('article_id', ''),
            'title': item.get('article_title', ''),
            'url': item.get('article_url', ''),
            'article_img': item['image_id'] + '.jpg' if 'image_id' in item and item['image_id'] else '',
            'main_text': main_text,
            'og_data': og_data
        }
        enriched.append(enriched_item)
    return enriched


def main():
    """Main function for the crawler."""
    parser = argparse.ArgumentParser(description="Crawl news articles")
    parser.add_argument('--id', type=str, help='Specific article ID to process')
    args = parser.parse_args()
    
    # Load CSV data
    print(f"Loading data from {INPUT_CSV}...")
    data_dict = parse_csv_to_dict_list(INPUT_CSV)
    
    # Add article IDs (1-based indexing)
    for idx, item in enumerate(data_dict):
        item['article_id'] = str(idx + 1)
    
    # Filter for specific ID if provided
    if args.id:
        try:
            article_id = int(args.id)
            if article_id < 1 or article_id > len(data_dict):
                print(f"Error: Article ID {args.id} not found. Available IDs: 1-{len(data_dict)}")
                return
            # Get specific article (convert to 0-based index)
            data_dict = [data_dict[article_id - 1]]
            print(f"Processing article ID: {args.id}")
        except ValueError:
            print(f"Error: Invalid article ID '{args.id}'. Must be a number.")
            return
    else:
        print(f"Processing all {len(data_dict)} articles")
    
    # Crawl articles
    html_list = []
    for item in tqdm(data_dict, desc="Crawling articles"):
        url = item.get('article_url', '')
        html, error = crawl_single_article(url)
        html_list.append(html)
        if error:
            print(f"Failed to crawl {item.get('article_id', 'unknown')}: {error}")
    
    # Enrich data with crawled content
    enriched_data = enrich_data_with_crawled_content(data_dict, html_list)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save results
    if args.id:
        output_file = os.path.join(OUTPUT_DIR, f'crawled_article_{args.id}.json')
    else:
        output_file = os.path.join(OUTPUT_DIR, 'crawled_articles_all.json')
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enriched_data, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
