#This file gets article counts for various stats
import time
import json


with open('articles_random_v3_noduplicates.json') as f:
  article_data = json.load(f)

news_sources = article_data.keys()

for source in news_sources:
  print(source,len(article_data[source]['articles']))

with open('articles_random_v3_noduplicates_nomccain.json') as f:
  article_data = json.load(f)
for source in news_sources:
  print(source,len(article_data[source]['articles']))

with open('articles_random_v3_cleaned.json') as f:
  article_data = json.load(f)
for source in news_sources:
  print(source,len(article_data[source]['articles']))

