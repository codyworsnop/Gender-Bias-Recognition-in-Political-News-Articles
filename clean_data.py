from Orchestrator import *

orchestrator = Orchestrator()
#orchestrator.read_data(path = "Data/articles_random_v4.json", random= True, number_of_articles = 1000, savePath = "Data/articles_random_v4_cleaned.json", save = True, clean = True)
orchestrator.read_data_csv(path = "store/all-the-news-2-1.csv", random= True, number_of_articles = 1000, savePath = "store/all-the-news_cleaned.csv", save = True, clean = True)
