#This is old code for plotting sentiment
ind = np.arange(len(leanings))    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence
offset = 0.01
p1 = plt.bar((ind - width / 2) - offset, neg_counts_per_leaning_female, width, color='crimson')
p2 = plt.bar((ind - width / 2) - offset, pos_counts_per_leaning_female, width, bottom=neg_counts_per_leaning_female, color='slateblue')
p3 = plt.bar((ind + width / 2) + offset, neg_counts_per_leaning_male, width, color='crimson', hatch='////'),
p4 = plt.bar((ind + width / 2) + offset, pos_counts_per_leaning_male, width, bottom=neg_counts_per_leaning_male, color='slateblue', hatch='////')

plt.xticks(ind, (leanings))
plt.yticks(np.arange(0, 1, 0.1))
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Female Negative', 'Female Positve', 'Male Negative', 'Male Positive'))

plt.ylabel('Mean Leaning Sentiment Positive:Negative Ratio')
plt.title('Positive and Negative Sentiment by Leaning and Gender')
plt.show()
