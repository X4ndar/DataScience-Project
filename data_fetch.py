import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

def scrape_categories(base_url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    response = requests.get(base_url, headers=headers)
    response.raise_for_status()
    response.encoding = 'utf-8'

    soup = BeautifulSoup(response.text, 'html.parser')
    categories = soup.find_all('h2', class_='lsa-widget-title')

    category_info = []
    for category in categories:
        link = category.find('a')
        if link:
            category_info.append((link.text.strip(), link['href']))

    with open('output.txt', 'w', encoding='utf-8') as f:
        for title, url in category_info:
            f.write(f"Category Title: {title}, URL: {url}\n")
    
    return category_info

def scrape_articles(category_info):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    articles_info = []
    for title, url in tqdm(category_info, desc="Scraping categories"):
        cat_response = requests.get(url, headers=headers)
        cat_response.raise_for_status()
        cat_response.encoding = 'utf-8'

        cat_soup = BeautifulSoup(cat_response.text, 'html.parser')
        articles = cat_soup.find_all('div', class_='card-img-top')

        for article in articles:
            article_link = article.find('a')
            if article_link and 'title' in article_link.attrs:
                articles_info.append((title, article_link['title'], article_link['href']))

    with open('articles_for_category.txt', 'w', encoding='utf-8') as f:
        for category_title, article_title, article_url in tqdm(articles_info, desc="Writing articles"):
            f.write(f"Category: {category_title}, Article Title: {article_title}, URL: {article_url}\n")
    
    return articles_info

def scrape_comments(articles_info):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    comments_info = []
    for category_title, article_title, article_url in tqdm(articles_info, desc="Scraping articles for comments"):
        article_response = requests.get(article_url, headers=headers)
        article_response.raise_for_status()
        article_response.encoding = 'utf-8'

        article_soup = BeautifulSoup(article_response.text, 'html.parser')
        comment_section = article_soup.find('div', class_='comments')

        if comment_section:
            comments_list = comment_section.find('ul', class_='comment-list hide-comments')
            if comments_list:
                for comment in comments_list.find_all('li'):
                    author_div = comment.find('div', class_='comment-author vcard')
                    text_div = comment.find('div', class_='comment-text')
                    if author_div and text_div:
                        author_name = author_div.find('span').text.strip()
                        comment_text = text_div.find('p').text.strip()
                        comments_info.append((category_title, article_title, author_name, comment_text))
    
    with open('comments.txt', 'w', encoding='utf-8') as f:
        for category_title, article_title, author_name, comment_text in tqdm(comments_info, desc="Writing comments"):
            f.write(f"Category: {category_title}, Article: {article_title}, Author: {author_name}, Comment: {comment_text}\n")

def main():
    base_url = 'https://www.hespress.com'
    category_info = scrape_categories(base_url)
    articles_info = scrape_articles(category_info)
    scrape_comments(articles_info)
    print("Scraping complete. Data written to output.txt, articles_for_category.txt, and comments.txt.")

if __name__ == "__main__":
    main()
