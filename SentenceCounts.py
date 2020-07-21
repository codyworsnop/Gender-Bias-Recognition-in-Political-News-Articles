#######This file calculates gers sentence counts of news sources and candidate targets #######
from nltk.tokenize import sent_tokenize
import ApplicationConstants
from DataReader import DataReader

def get_count(articles) -> int:
	article_count = 0 

	for article in articles:
		tokens = sent_tokenize(article)
		article_count += len(tokens)

	return article_count

def get_article_count(articles) -> int: 

	return len(articles) 

if __name__ == "__main__":

	reader = DataReader()
	splits = reader.Load_Splits(ApplicationConstants.all_articles_random_v2, None, number_of_articles=100, clean=False, save=False, shouldRandomize=False)
	
	for leaning in splits[0]:

		print("Source:", leaning)

		all_leaning_data = splits[0][leaning][ApplicationConstants.Train] + splits[0][leaning][ApplicationConstants.Validation] + splits[0][leaning][ApplicationConstants.Test]
		all_content = list(map(lambda article: article, all_leaning_data))
		leaning_count = get_article_count(all_content)

		print("Total Count:", leaning_count)

		#get gendered counts
		male = list(filter(lambda article: article.Label.TargetGender == 1, all_leaning_data))
		female = list(filter(lambda article: article.Label.TargetGender == 0, all_leaning_data))

		print("Total male count:", get_article_count(list(map(lambda article: article, male))))
		print("Total female count:", get_article_count(list(map(lambda article: article, female))))

		#get candidate count
		dt = list(filter(lambda article: article.Label.TargetName == ApplicationConstants.DonaldTrump, all_leaning_data))
		jb = list(filter(lambda article: article.Label.TargetName == ApplicationConstants.JoeBiden, all_leaning_data))
		bs = list(filter(lambda article: article.Label.TargetName == ApplicationConstants.BernieSanders, all_leaning_data))
		mm = list(filter(lambda article: article.Label.TargetName == ApplicationConstants.MitchMcconnell, all_leaning_data))
		bo = list(filter(lambda article: article.Label.TargetName == ApplicationConstants.BarrackObama, all_leaning_data))
		hc = list(filter(lambda article: article.Label.TargetName == ApplicationConstants.HillaryClinton, all_leaning_data))
		sp = list(filter(lambda article: article.Label.TargetName == ApplicationConstants.SarahPalin, all_leaning_data))
		aoc = list(filter(lambda article: article.Label.TargetName == ApplicationConstants.AlexandriaOcasioCortez, all_leaning_data))
		bd = list(filter(lambda article: article.Label.TargetName == ApplicationConstants.BetsyDevos, all_leaning_data))
		ew = list(filter(lambda article: article.Label.TargetName == ApplicationConstants.ElizabethWarren, all_leaning_data))

		print("trump:", get_article_count(list(map(lambda article: article, dt))))
		print("joe biden:", get_article_count(list(map(lambda article: article, jb))))
		print("bernie:", get_article_count(list(map(lambda article: article, bs))))
		print("mitch:", get_article_count(list(map(lambda article: article, mm))))
		print("obama:", get_article_count(list(map(lambda article: article, bo))))
		print("hillary:", get_article_count(list(map(lambda article: article, hc))))
		print("sarah:", get_article_count(list(map(lambda article: article, sp))))
		print("aoc:", get_article_count(list(map(lambda article: article, aoc))))
		print("betsy:", get_article_count(list(map(lambda article: article, bd))))
		print("warren:", get_article_count(list(map(lambda article: article, ew))))