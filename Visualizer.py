from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np 

import ApplicationConstants

cmap = ['red','blue']

class Visualizer():

    def hover(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            cont, ind = self.sc.contains(event)
            if cont:
                self.update_annot(ind)
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.fig.canvas.draw_idle()

    def update_annot(self, ind):

        pos = self.sc.get_offsets()[ind["ind"][0]]
        self.annot.xy = pos
        text = "{}".format(" ".join([self.articles[n].Title for n in ind["ind"]]))
        self.annot.set_text(text)
       # self.annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        self.annot.get_bbox_patch().set_alpha(0.4) 

    def plot_TSNE(self, leaning, weights, true_labels, articles):
        
        self.articles = articles

        self.genders = list(map(lambda label: 'Male' if label == 1 else 'Female', true_labels))
        tsne = TSNE(verbose=1)
        results = tsne.fit_transform(weights)

        self.fig, self.ax = plt.subplots()

        self.annot = self.ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)

        #self.sc = plt.scatter(x=results[:,0], y=results[0:,1], c=true_labels, cmap=matplotlib.colors.ListedColormap(cmap))
        self.sc = sns.scatterplot(x=results[:,0], y=results[0:,1], palette=sns.color_palette("hls", 2), hue=self.genders)

        #plt.setp(ax.get_legend().get_texts(), fontsize='40')
        plt.legend( loc='best', prop={'size': 15})
        #plt.legend(*self.sc.legend_elements(), loc='best', prop={'size': 20})
        plt.title('t-SNE Article Distribution for ' + leaning, fontsize=20)
        #self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        plt.show()

    def graph_sentiment(self, Fsentiment, Msentiment):

        pos_counts_per_leaning_female = [] 
        neg_counts_per_leaning_male = []
        pos_counts_per_leaning_male = [] 
        neg_counts_per_leaning_female = []
        leanings = ["Huffpost", "New York Times",  "USA Today", "Fox", "Breitbart"]

        breitbart_female_sentiments = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.Breitbart, Fsentiment))))
        breitbart_male_sentiments = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.Breitbart, Msentiment))))
        fox_female_sentiments = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.Fox, Fsentiment))))
        fox_male_sentiments = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.Fox, Msentiment))))
        usa_female_sentiments = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.usa_today, Fsentiment))))
        usa_male_sentiments = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.usa_today, Msentiment))))
        nyt_female_sentiments = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.New_york_times, Fsentiment))))
        nyt_male_sentiments = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.New_york_times, Msentiment))))
        hp_female_sentiments = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.HuffPost, Fsentiment))))
        hp_male_sentiments = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.HuffPost, Msentiment))))

        male_leanings = [hp_male_sentiments, nyt_male_sentiments, usa_male_sentiments, fox_male_sentiments, breitbart_male_sentiments]
        female_leanings = [hp_female_sentiments, nyt_female_sentiments, usa_female_sentiments, fox_female_sentiments, breitbart_female_sentiments]
   
        for leaning in range(5): 

            femaleVals = []
            maleVals = []

            male_articles_length = len(male_leanings[leaning])
            female_articles_length = len(female_leanings[leaning])
            
            for sentiment in female_leanings[leaning]:
                femaleVals.append(self.calc_sent(sentiment))

            for sentiment in male_leanings[leaning]:
                maleVals.append(self.calc_sent(sentiment))

            female_pos = len(list(filter(lambda sent: sent == 'pos', femaleVals)))
            female_neg = len(list(filter(lambda sent: sent == 'neg', femaleVals)))
            male_pos = len(list(filter(lambda sent: sent == 'pos', maleVals)))
            male_neg = len(list(filter(lambda sent: sent == 'neg', maleVals)))

            print(leaning)
            print("Num Female Pos: " + str(female_pos))
            print("Num Female Neg: " + str(female_neg))
            print("Num Male Pos: " + str(male_pos))
            print("Num Male Neg: " + str(male_neg))

            pos_counts_per_leaning_female.append(female_pos / female_articles_length)
            neg_counts_per_leaning_female.append(female_neg / female_articles_length)
            neg_counts_per_leaning_male.append(male_neg / male_articles_length)
            pos_counts_per_leaning_male.append(male_pos / male_articles_length)

        plt.plot(leanings, pos_counts_per_leaning_female, marker='D', label='Positive Female Articles', color='seagreen')
        plt.plot(leanings, neg_counts_per_leaning_female, marker='D', label='Negative Female Articles', color='slateblue')
        plt.plot(leanings, pos_counts_per_leaning_male, marker='D', label='Positive Male Articles', color='orange')
        plt.plot(leanings, neg_counts_per_leaning_male, marker='D', label='Negative Male Articles', color='crimson')

        plt.ylabel('Mean Leaning Sentiment Positive:Negative Ratio')
        plt.title('Positive and Negative Sentiment by Leaning and Gender')
        plt.xticks(leanings)
        plt.ylim((0, 1))
        plt.legend(loc='best')
        plt.show()

    def graph_neutral(self, neutral_sents):

        pos_sent_count = [] 
        neg_sent_count = []
        leaning_names = ["Huffpost", "New York Times",  "USA Today", "Fox", "Breitbart"]

        breitbart = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.Breitbart, neutral_sents))))
        fox = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.Fox, neutral_sents))))
        usa = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.usa_today, neutral_sents))))
        nyt = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.New_york_times, neutral_sents))))
        hp = list(map(lambda sentiment: sentiment[1], list(filter(lambda leaning: leaning[0] == ApplicationConstants.HuffPost, neutral_sents))))

        leanings = [hp, nyt, usa, fox, breitbart]
   
        for leaning in leanings: 

            article_length = len(leaning)
            calculated_sent = []

            for sentiment in leaning:
                calculated_sent.append(self.calc_sent(sentiment))

            pos = len(list(filter(lambda sent: sent == 'pos', calculated_sent)))
            neg = len(list(filter(lambda sent: sent == 'neg', calculated_sent)))

            print(leaning)
            print("pos: " + str(pos))
            print("neg: " + str(neg))

            pos_sent_count.append(pos / article_length)
            neg_sent_count.append(neg / article_length)

        plt.plot(leaning_names, pos_sent_count, marker='D', label='Positive Articles', color='seagreen')
        plt.plot(leaning_names, neg_sent_count, marker='D', label='Negative Articles', color='slateblue')

        plt.ylabel('Mean Leaning Sentiment Positive:Negative Ratio')
        plt.title('Positive and Negative Sentiment by Leaning and Gender')
        plt.xticks(leaning_names)
        plt.ylim((0, 1))
        plt.legend(loc='best')
        plt.show()
        
    def calc_sent(self, sentiment):

        if sentiment == 0:
            return 'neg'
        else:
            return 'pos'
        #score = sentiment[0]
        #magnitude = sentiment[1]

        #if score > 0.25:
        #    return 'pos'
        #elif score < -0.25:
        #    return 'neg'

    def plot_avg_distance(self):

        #CLEANED
        # In order female pos/female neg/male pos/male neg
        # Breitbart: 0.27750405826258234 -0.653181590060444 0.2903213814210077 -0.5961900713637649
        # Fox: 0.29441780333956813 -0.7107146607394222 0.30068177440338595 -0.5740697997089191
        # USA: 0.29950806804842184 -0.6853463022397768 0.33511702559452905 -0.6820991237348888
        # Huffpost: 0.32162210509099975 -0.8307027327502395 0.3163770866849656 -0.6258323260268398
        # NYT: 0.32922644074905877 -0.8747165386461293 0.7581123567127453 -0.7461601907196513

        #UNCLEANED
        # In order female pos/female neg/male pos/male neg
        # Breitbart: 0.3171811433577991 -0.6986200705361315 0.3683481953105339 -0.7809928044192702
        # Fox: 0.3397301614597068 -0.7156135265721492 0.34968552630370314 -0.5640244497359626
        # USA: 0.3300914998649323 -0.7565994750684083 0.39410964144499805 -0.8878224386171693
        # Huffpost: 0.3496310036232018 -0.8844523817045514 0.36350725125617994 -0.6993278922888551
        # NYT: 0.4088732952324052 -0.8844714898803295 0.7911512034957858 -0.6280179226086297

        leanings = ["Huffpost", "New York Times",  "USA Today", "Fox", "Breitbart"]

        #hp, nyt, usa, fox, breitbart
        cleaned_female_neg = [0.8307027327502395, 0.8747165386461293, 0.6853463022397768, 0.7107146607394222, 0.653181590060444] 
        cleaned_male_neg = [0.6258323260268398, 0.7461601907196513, 0.6820991237348888, 0.5740697997089191, 0.5961900713637649] 

        uncleaned_female_neg = [0.8844523817045514, 0.8844714898803295, 0.7565994750684083, 0.7156135265721492, 0.6986200705361315]
        uncleaned_male_neg = [0.6993278922888551, 0.6280179226086297, 0.8878224386171693, 0.5640244497359626, 0.7809928044192702] 

        plt.plot(leanings, cleaned_female_neg, marker='D', label='Cleaned Female Distance', color='seagreen')
        plt.plot(leanings, cleaned_male_neg, marker='D', label='Cleaned Male Distance', color='slateblue')
        plt.plot(leanings, uncleaned_female_neg, marker='D', label='Uncleaned Female Distance', color='orange')
        plt.plot(leanings, uncleaned_male_neg, marker='D', label='Uncleaned Male Distance', color='crimson')

        #plt.ylabel('')
        plt.title('Average Distance to The hyperplane')
        plt.xticks(leanings)
        plt.ylim((0.4, .95))
        plt.legend(loc='lower left')
        plt.show()

# v = Visualizer() 
# v.plot_avg_distance()