'''This file calculates sentiment of each article, and graphs/leaning
#NOTE: articles need to be gathered, and run_preprocessor.py needs to have been run prior to using this, and the Hu and
Liu, 2004 sentiment dataset needs to be downloaded. It can be found here: http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar
The words "trump, vice, and right" must be removed to remain consistent with our cleaning, and then saved using
the naming convention found on line 39.
'''
import numpy as np
import json
from matplotlib import pyplot as plt
import time
import seaborn as sns
import ApplicationConstants
sns.set()
sns.color_palette("colorblind", 2)

PER_THRESH = 0.05
ratio = 0.2

STOP_WORDS = []
#f = open('stop_words.txt', 'r')
#for line in f:
#  line = line.strip()
#  STOP_WORDS.append(line)
#f.close()
STOP_WORDS = set(STOP_WORDS)

NEG_WORDS = []
POS_WORDS = []

f = open('opinion-lexicon-English/negative-words-novice.txt', 'rb')
for line in f:
  line = line.decode('ascii', 'ignore')
  if not line.startswith(';'):
    line = line.strip()
    NEG_WORDS.append(line)
f.close()
NEG_WORDS = set(NEG_WORDS)

f = open('opinion-lexicon-English/positive-words-notrump.txt', 'rb') #you will need to remove trump, vice, and right from the downloaded lexicon
for line in f:
  line = line.decode('ascii', 'ignore')
  if not line.startswith(';'):
    line = line.strip()
    POS_WORDS.append(line)
f.close()
POS_WORDS = set(POS_WORDS)

with open(ApplicationConstants.all_articles_random_v4_cleaned) as f:
  article_data = json.load(f)

curated_data = {}

news_sources = article_data.keys()

pos_female = {}
pos_female_score = {}
pos_male = {}
pos_male_score = {}
neg_male = {}
neg_male_score = {}
neg_female = {}
neg_female_score = {}

no_content_cnt = {}
for source in news_sources:
  pos_female[source] = {}
  pos_male[source] = {}
  neg_female[source] = {}
  neg_male[source] = {}
  pos_female_score[source] = 0
  pos_male_score[source] = 0
  neg_female_score[source] = 0
  neg_male_score[source] = 0
  curated_data[source] = {}
  curated_data[source]['articles'] = []

  index = 0
  for article in article_data[source]['articles']:
    stop_cnt = 0
    content = article['content']
    temp_content = content
    labels = article['labels']
    title = article['title']
    content = content.lower().split()
    neg_cnt = 0
    pos_cnt = 0
    article['pos_words'] = []
    article['neg_words'] = []
    for item in content:
      item = item.replace('.', '')
      item = item.replace('"', '')
      item = item.replace(',', '')
      item = item.replace("'", '')
      item = item.replace(':', '')
      item = item.replace(';', '')
      item = item.replace('?', '')
      if item in NEG_WORDS:
        neg_cnt += 1
        article['neg_words'].append(item)
      elif item in POS_WORDS:
        pos_cnt += 1
        article['pos_words'].append(item)


    if len(content) > 0: # and labels['target_name'].startswith("Alex"): 
      pos_per = pos_cnt/float(len(content)-stop_cnt)
      neg_per = neg_cnt/float(len(content)-stop_cnt)
      article['pos_cnt'] = pos_per
      article['neg_cnt'] = neg_per
      if pos_per > PER_THRESH and neg_per <= pos_per*ratio:
        curated_data[source]['articles'].append(article)
        if labels['target_gender'] == 'Female' or labels['target_gender'] == 0:
          if labels['target_name'] in pos_female[source]:
            pos_female[source][labels['target_name']] += 1
          else:
            pos_female[source][labels['target_name']] = 1

          #pos_female[source] += 1
          pos_female_score[source] += pos_per
        else:
          if labels['target_name'] in pos_male[source]:
            pos_male[source][labels['target_name']] += 1
          else:
            pos_male[source][labels['target_name']] = 1
          #pos_male[source] += 1
          pos_male_score[source] += pos_per
      if neg_per > PER_THRESH and pos_per <=  neg_per*ratio:
        curated_data[source]['articles'].append(article)
        if labels['target_gender'] == 'Female' or labels['target_gender'] == 0:
          if labels['target_name'] in neg_female[source]:
            neg_female[source][labels['target_name']] += 1
          else:
            neg_female[source][labels['target_name']] = 1
          #neg_female[source] += 1
          neg_female_score[source] += neg_per
        else:
          if labels['target_name'] in neg_male[source]:
            neg_male[source][labels['target_name']] += 1
          else:
            neg_male[source][labels['target_name']] = 1
          #neg_male[source] += 1
          neg_male_score[source] += neg_per
    index += 1
    

with open(ApplicationConstants.all_articles_sentiment_counts, 'w') as json_file:
  json.dump(curated_data, json_file)

data = {}
percentages = {}
data['f'] = {}
data['m'] = {}
percentages['f'] = {}
percentages['m'] = {}
for g in ['f','m']:
  for a in ['pos','neg']:
    data[g][a] = {}
    percentages[g][a] = {}

fout = open('sentiment_output.txt', 'w')
for source in news_sources:
  fout.write("Source: " + source + '\n')
  tot = 0
  fout.write("Pos Female: \n")
  for item in pos_female[source]:
    fout.write("     " + item + ": " + str(pos_female[source][item]) + "\n")
    tot += pos_female[source][item]
  data['f']['pos'][source] = tot
  percentages['f']['pos'][source] = tot
  tot = 0
  fout.write("Neg Female: \n")
  for item in neg_female[source]:
    fout.write("     " + item + ": " + str(neg_female[source][item]) + "\n")
    tot += neg_female[source][item]
  data['f']['neg'][source] = tot
  percentages['f']['neg'][source] = tot

  tot = 0
  fout.write("Pos Male: \n")
  for item in pos_male[source]:
    fout.write("     " + item + ": " + str(pos_male[source][item]) + "\n")
    tot += pos_male[source][item]
  data['m']['pos'][source] = tot
  percentages['m']['pos'][source] = tot
  tot = 0
  fout.write("Neg Male: \n")
  for item in neg_male[source]:
    fout.write("     " + item + ": " + str(neg_male[source][item]) + "\n")
    tot += neg_male[source][item]
  data['m']['neg'][source] = tot
  percentages['m']['neg'][source] = tot

fout.write("\n")
for g in ['f','m']:
  for a in ['pos','neg']:
    fout.write(g + " " + a + " " + str(data[g][a]) + "\n")
    print(g, a, data[g][a])
  for source in news_sources:
    percentages[g]['neg'][source] = data[g]['neg'][source]/(data[g]['neg'][source] + data[g]['pos'][source])
    percentages[g]['pos'][source] = data[g]['pos'][source]/(data[g]['neg'][source] + data[g]['pos'][source])
fout.close()


news_source_list = list(news_sources)
#PLOTTING
ind = np.arange(len(news_sources))    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence
offset = 0.01
print(ind)
pos_female = []
neg_female = []
pos_male = []
neg_male = []
for i in ind:
  source = news_source_list[i]
  pos_female.append(percentages['f']['pos'][source])
  neg_female.append(percentages['f']['neg'][source])
  pos_male.append(percentages['m']['pos'][source])
  neg_male.append(percentages['m']['neg'][source])
p1 = plt.bar((ind - width / 2) - offset, neg_female, width, color='crimson')
p2 = plt.bar((ind - width / 2) - offset, pos_female, width, bottom=neg_female, color='slateblue')
p3 = plt.bar((ind + width / 2) + offset, neg_male, width, color='crimson', hatch='////'),
p4 = plt.bar((ind + width / 2) + offset, pos_male, width, bottom=neg_male, color='slateblue', hatch='////')

plt.xticks(ind, (news_source_list))
plt.yticks(np.arange(0, 1, 0.1))
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Female Negative', 'Female Positve', 'Male Negative', 'Male Positive'))

plt.ylabel('Mean Leaning Sentiment Positive:Negative Ratio')
plt.title('Positive and Negative Sentiment by Leaning and Gender')
#plt.show()
plt.savefig('./visualizations/sentiment_output_percentages_' + str(PER_THRESH) + '.png')
