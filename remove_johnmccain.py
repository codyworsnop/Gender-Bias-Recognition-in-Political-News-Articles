#This removes john mccain, who was originally a target candidate that we removed from our set
import time
import json

with open('articles_random_v3_noduplicates.json') as f:
  article_data = json.load(f)

new_article_data = {}
duplicate_article_data = {}

news_sources = article_data.keys()

duplicate_urls = []
target_names = []
new_target_names = []
for source in news_sources:
  new_article_data[source] = {}

  new_article_data[source]['articles'] = []

  for article in article_data[source]['articles']:
    target_name = article['labels']['target_name']
    if target_name not in target_names:
      target_names.append(target_name)
    if 'John' not in target_name:
      if "Barrack" in target_name:
        article['labels']['target_name'] = "Barack_Obama"
      if "biden" in target_name:
        article['labels']['target_name'] = "Joe_Biden"
      new_article_data[source]['articles'].append(article)
      if article['labels']['target_name'] not in new_target_names:
        new_target_names.append(article['labels']['target_name'])
      
         
  print(source,len(article_data[source]['articles']), len(new_article_data[source]['articles']))
print(target_names)
print(new_target_names)

with open('articles_random_v3_noduplicates_nomccain.json', 'w') as json_file:
    json.dump(new_article_data, json_file, indent=2)

