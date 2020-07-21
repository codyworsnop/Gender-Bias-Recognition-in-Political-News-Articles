#######This file cleans data as it's read, if clean = True #######
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import unicodedata
import string
import re
import StopWords
import json
from collections import Counter
import spacy
from spacy import displacy
import en_core_web_lg
import RegexSearchPatterns

class Preprocessor():

    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")

    def cleanup(self, token, lower = True):
        if lower:
            token = token = token.lower()
        return token.strip()

    def Clean_POS(self, data: str):
        # normalize the data, removing punctuation
        data = unicodedata.normalize('NFKC', data)

        # remove numbers
        data = re.sub('\d+', '  ', data)

        # remove any word containing woman replace with person
        doc = self.nlp(data)

        data = ''
        # Used tagged entities to replace names of places and people
        for token in doc:
            if token.pos_ == "ADJ":
                data = data + ' ' + str(token)

        # replace gendered words with person and non gendered pronouns
        for pattern, replacement in RegexSearchPatterns.Patterns:
            data = re.sub(pattern, replacement, data, flags=re.I)

        emoticons = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0000200B"
                               u"\U0000202C"
                               u"\U00002069"
                               u"\U00002066"
                               u"\U0000202A" 
                               "]+", flags=re.UNICODE)
        data = emoticons.sub(' ', data)

        # remove whole words from stop list
        for word in StopWords.StopWords:
            reg_string = '\\b' + word + '((-[a-zA-Z]*)|([,.!?;"\' ]))'
            data = re.sub(reg_string, ' ', data, flags=re.I)

        # Cleanup the punctuation / symbols
        data = re.sub(' ,', ',', data, flags=re.I)
        data = re.sub('  ', ' ', data, flags=re.I)
        data = re.sub('\+', ' ', data, flags=re.I)
        data = re.sub('&', ' ', data, flags=re.I)
        data = re.sub('  ', ' ', data, flags=re.I)
        # replace huperson with human (an unintended consequence of our man/person regex swap)
        data = re.sub('huperson', 'human', data, flags=re.I)
        data = re.sub('ombudsperson', 'person', data, flags=re.I)
        data = re.sub('ottoperson', 'norp', data, flags=re.I)

        # return processed_data
        return data


    def Clean(self, data : str):
        ''' Removes POS that are NNP, PRP, or PRP$, and removes all stop words  '''

        #normalize the data, removing punctuation 
        data = unicodedata.normalize('NFKC', data)
        
        #remove numbers
        data = re.sub('\d+', '  ', data)
    
        #remove any word containing woman replace with person
        doc = self.nlp(data)
      
        #Used tagged entities to replace names of places and people
        for ent in reversed(doc.ents):
            # print(ent.text, ent.start_char, ent.end_char, ent.label_)
            if ent.label_ == 'PERSON' or ent.label_ == 'NORP' or ent.label_ == 'GPE' or ent.label == 'LOC':
                data = data[:ent.start_char] + ent.label_.lower() + " " + data[ent.end_char:]


        #replace gendered words with person and non gendered pronouns
        for pattern, replacement in RegexSearchPatterns.Patterns:
            data = re.sub(pattern, replacement, data, flags = re.I)


        emoticons =  re.compile("["
                   u"\U0001F600-\U0001F64F"  # emoticons
                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                   u"\U00002702-\U000027B0"
                   u"\U000024C2-\U0001F251"
                   u"\U0000200B"
                   u"\U0000202C"
                   u"\U00002069"
                   u"\U00002066"
                   u"\U0000202A"             
                   "]+", flags=re.UNICODE)
        data = emoticons.sub(' ', data)



        #remove whole words from stop list 
        for word in StopWords.StopWords: 
            reg_string = '\\b' + word + '((-[a-zA-Z]*)|([,.!?;"\' ]))'
            data = re.sub(reg_string, ' ', data, flags = re.I)


        #Cleanup the punctuation / symbols
        data = re.sub(' ,', ',', data, flags = re.I)
        data = re.sub('  ', ' ', data, flags = re.I)
        data = re.sub('\+', ' ', data, flags = re.I)
        data = re.sub('&', ' ', data, flags = re.I)
        data = re.sub('  ', ' ', data, flags=re.I)
        data = re.sub('â€', ' ', data, flags=re.I)
        data = re.sub('ão', ' ', data, flags=re.I)
        #replace huperson with human (an unintended consequence of our man/person regex swap)
        data = re.sub('huperson', 'human', data,  flags = re.I)
        data = re.sub('ombudsperson', 'person', data, flags=re.I)
        data = re.sub('ottoperson', 'norp', data, flags=re.I)


        #return processed_data
        return data

if __name__ == "__main__":

    process = Preprocessor()
    #process.Clean("president Trump, Donald Trump, AOC, secretary Clinton, Trump. president")
    print(process.Clean("Ocasio-Cortez (D-Ny.), gop GOP G.O.P g.o.p. + 51st & CBS's Impeachment he he-hiadw helicopter who is a self-described 'democratic socialist,' made the comments in an interview published Tuesday. Yahoo! News reported: 'Are we headed to fascism? Yes. I don't think there's a question,' the congresswoman told Yahoo News hours after she 'toured' the detention facilities run by Customs and Border Protection. 'If you actually take the time to study, and to look at the steps, and to see how government transforms under authoritarian regimes, and look at the political decisions and patterns of this president, the answer is yes.' Last month Ocasio-Cortez sparked controversy when she described the migrant detention facilities as 'concentration camps on our southern border.' Her comment drew an immediate backlash from critics who accused her of trivializing Nazi concentration camps, while others, including some Holocaust survivors and scholars, said the comparison was a valid one. The freshman Democratic lawmaker from New York refused to back down from that comparison, and in her conversation with Yahoo News, Ocasio-Cortez argued that many things about President Trump echo that dark period in history. Read the full Yahoo! News article here. Also on Monday evening, National Border Patrol Council president Brandon Judd told Breitbart News Tonight that Ocasio-Cortez had lied about conditions at facilities where illegal aliens are being held near the border, as well as about the behavior of officers at the facilities. 'How can you have the moral high ground if you are going to throw facts out the window and spew falsehoods?', he commented. Ocasio-Cortez was elected in November as part of a Democratic Party victory in the U.S. House of Representatives, matching a pattern throughout recent U.S. history in which the opposition fares well in the midterm elections. Since taking office, Ocasio-Cortez has proposed ambitious new policies that grant sweeping powers to the government over the U.S. economy, envisioning a highly controlled system in which the state controls private industry and steers it toward overarching utopian goals, such as a shift to 100% renewable energy and the elimination of fossil fuels. Hers, his, she, he, Trump, shis, president Trump, president obama, secretary clinton, secretary of education, secretary of state, department of education, department of state person person "))