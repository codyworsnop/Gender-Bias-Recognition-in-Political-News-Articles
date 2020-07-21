#######This file searches for sentences containing target name#######

import json
import time
import re
import ApplicationConstants



def split_into_sentences(text):
    alphabets = "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt|Sen|Gov)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov|me|edu)"
    digits = "([0-9])"

    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text) 
    text = text.replace("e.g.","e<prd>g<prd>")
    text = text.replace("i.e.","i<prd>e<prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    #if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def run():
    with open(ApplicationConstants.all_articles_random_v4) as f:
      article_data = json.load(f)

    news_sources = article_data.keys()

    sentence_data = article_data.copy()

    for source in news_sources:
      index = 0
      removed = 0
      for article in article_data[source]['articles']:
        content = article['content']
        labels = article['labels']
        target = labels['target_name']
        if target.startswith('Alexandria'):
          target = "Alexandria_Ocasio-Cortez"
        if target.startswith('Barrack'):
          target = "Barack_Obama"
        target_split = target.split('_')
        title = article['title']
        sentences = split_into_sentences(content)
        #sentences = tokenize.sent_tokenize(content)
        paragraph = ''
        sent_cnt = 0
        for sentence in sentences:
          if target in sentence:
            paragraph = paragraph + ' ' + sentence
            sent_cnt += 1
          elif target_split[0] in sentence:
            paragraph = paragraph + ' ' + sentence
            sent_cnt += 1
          elif target_split[1] in sentence:
            paragraph = paragraph + ' ' + sentence
            sent_cnt += 1
          elif target.startswith('Alex'):
            if 'aoc' in sentence or 'AOC' in sentence:
              paragraph = paragraph + ' ' + sentence
              sent_cnt += 1
        if sent_cnt < 2:
          paragraph = ''
          removed += 1
        sentence_data[source]['articles'][index]['content'] = paragraph
        index += 1
      print(source, removed)


    with open(ApplicationConstants.all_articles_random_v4_candidate_names, 'w') as outfile:
      article_data = json.dump(sentence_data, outfile, indent=2)
