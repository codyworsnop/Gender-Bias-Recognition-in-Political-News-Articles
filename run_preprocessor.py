#######This file will generate various versions (see below) of jsons after gathering them using links #######

from Orchestrator import *
import ApplicationConstants
from parse_sentences import run

orchestrator = Orchestrator()
#Clean the articles
orchestrator.read_data(path = "Data/articles_random_v3.json", save = True,
                       savePath = "Data/articles_random_v4_cleaned.json", random = True, number_of_articles=1000)

#Create the candidate name sentence json using parse_sentences
run()

#Clean the sentences containing the candidate name, then do POS tagging, save ADJ
orchestrator.read_data(path = ApplicationConstants.all_articles_random_v4_candidate_names, save = True,
                       savePath = ApplicationConstants.all_articles_random_v4_cleaned_pos_candidate_names, clean = True,
                        random = True, number_of_articles = 1000, pos_tagged = True)

#Clean all of the sentences (regardless of if candidate name is there), then do POS tagging, save ADJ
orchestrator.read_data(path = ApplicationConstants.all_articles_random_v4, save = True,
                       savePath = ApplicationConstants.all_articles_random_v4_cleaned_pos, clean = True,
                        random = True, number_of_articles = 1000, pos_tagged = True)