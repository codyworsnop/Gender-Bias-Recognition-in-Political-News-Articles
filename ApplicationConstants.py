#Application Constants

#Filenames
all_articles = "./Data/articles.updatedv2.json"
all_articles_random_v2 = "./Data/articles_random_v2.json"
all_articles_random_v2_cleaned = "./Data/articles_random_v2_cleaned.json"
all_articles_random_v3 = "./Data/articles_random_v3.json"
all_articles_random_nodups_v3 = "./Data/articles_random_v3_noduplicates.json"
all_articles_random_v4= "./Data/articles_random_v4.json"
all_articles_random_v3_cleaned = "./Data/articles_random_v3_cleaned.json"
all_articles_random_v4_cleaned = "./Data/articles_random_v4_cleaned.json"
fold_1 = "./folds/fold1.txt"
fold_2 = "./folds/fold2.txt"
fold_3 = "./folds/fold3.txt"
fold_4 = "./folds/fold4.txt"
fold_5 = "./folds/fold5.txt"
all_the_news_path = './store/all-the-news.db'
all_the_news_newer_path = './store/all-the-news-2-1.csv'

#leaning constants
FarLeft = "Far_Left"
Left = "Left"
Neutral = "Neutral"
Right = "Right"
FarRight = "Far_Right"

#label constants
Male = "Male"
Female = "Female"

#candidate constants 
JoeBiden = "Joe_Biden"
BarrackObama = "Barack_Obama"
BernieSanders = "Bernie_Sanders"
DonaldTrump = "Donald_Trump"
JohnMccain = "John_Mccain"
HillaryClinton = "Hillary_Clinton"
BetsyDevos = "Betsy_Devos"
ElizabethWarren = "Elizabeth_Warren"
AlexandriaOcasioCortez = "Alexandria_ocasio-cortez"
SarahPalin = "Sarah_Palin"
MitchMcconnell = "Mitch_Mcconnell"

#sources
Breitbart = "breitbart"
New_york_times = "new_york_times"
usa_today = "usa_today"
Fox = "fox"
HuffPost = "huffpost"

#split constants
Train = "train"
Validation = "val"
Test = "test"

#female and male meanings
female_value = 0 
male_value = 1 

#sentiment values
postive_sentiment = 1 
negative_sentiment = 0 

#paths
cleaned_news_root_path = './cleaned_article_data/all_articles_random_cleaned.json'

all_articles_doc2vec_vector_cleaned_path = './store/all_article_embeddings_vectors_cleaned.npy'
all_articles_doc2vec_labels_cleaned_path = './store/all_article_embeddings_labels_cleaned.npy'
all_articles_doc2vec_model_cleaned_path = './store/all_articles_cleaned_doc2vec.model'
all_articles_doc2vec_vector_uncleaned_path = './store/all_article_embeddings_vectors_uncleaned.npy'
all_articles_doc2vec_labels_uncleaned_path = './store/all_article_embeddings_labels_uncleaned.npy'
all_articles_doc2vec_model_uncleaned_path = './store/all_articles_uncleaned_doc2vec.model'
imdb_sentiment_vector_uncleaned_path = "./store/imdb_sentiment_uncleaned.npy"
imdb_sentiment_label_uncleaned_path = "./store/imdb_sentiment_labels_uncleaned.npy"
imdb_sentiment_vector_cleaned_path = "./store/imdb_sentiment_cleaned.npy"
imdb_sentiment_label_cleaned_path = "./store/imdb_sentiment_labels_cleaned.npy"

qualitive_cleaned_path = './Data/qualitive_cleaned.json'
qualitive_uncleaned_path = './Data/qualitive_uncleaned.json'