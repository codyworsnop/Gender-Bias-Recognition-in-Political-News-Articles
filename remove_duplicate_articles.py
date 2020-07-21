#This removes duplicate articles if the scraper grabs extra, duplicated articles
import time
import json

with open('articles_random_v4_cleaned.json') as f:
  article_data = json.load(f)

new_article_data = {}
duplicate_article_data = {}

news_sources = article_data.keys()

duplicate_urls = []
for source in news_sources:
  new_article_data[source] = {}
  duplicate_article_data[source] = {}

  new_article_data[source]['articles'] = []
  duplicate_article_data[source]['articles'] = []

  url_set = []
  index = 0
  for article in article_data[source]['articles']:
    if article not in new_article_data[source]['articles']:
      #if source.startswith('b') and article['labels']['target_name'].startswith("A"):
      #  print(article['url'])
      if article['url'] not in url_set:
        new_article_data[source]['articles'].append(article)
        url_set.append(article['url'])
      else:
        if article['url'] not in duplicate_urls:
          duplicate_article_data[source]['articles'].append(article)
        duplicate_urls.append(article['url'])
         

  print(source,len(article_data[source]['articles']), len(new_article_data[source]['articles']))


#new_article_data2 = {}
#for source in news_sources:
#  new_article_data2[source] = {}
#  new_article_data2[source]['articles'] = []
#  index = 0
#  for article in article_data[source]['articles']:
#    if article['url'] not in duplicate_urls:
#      new_article_data2[source]['articles'].append(article)
#
#  print(source,len(article_data[source]['articles']), len(new_article_data2[source]['articles']))
#
with open('articles_random_v4_cleaned_nodup.json', 'w') as json_file:
    json.dump(new_article_data, json_file, indent=2)

#with open('v3_duplicate_articles.json', 'w') as json_file:
#    json.dump(duplicate_article_data, json_file)

