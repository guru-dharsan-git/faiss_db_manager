from newspaper import Article
import requests
url = 'https://www.indiatoday.in/india/story/ranchi-airport-bird-presence-unchecked-construction-raise-safety-concerns-2741775-2025-06-17'
rep=requests.get(url)
article = Article(rep)
article.download()
article.parse()

print(article.text)