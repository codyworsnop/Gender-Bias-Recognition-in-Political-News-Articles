import csv
from collections import Counter
files = ["output_words_5_50.txt","output_words_1_50.txt", "output_words_2_50.txt", "output_words_3_50.txt" , "output_words_4_50.txt"]

male_words = []
female_words = []
for file in files:
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ' ')
        count = 0
        for row in csv_reader:
            if count >0 and count < 26:
                male_words.append(row[0])
            if count > 26:
                female_words.append(row[0])
            count +=1

counts = Counter(male_words)
print(counts)
counts = Counter(female_words)
print()
print(counts)

print("ADJECTIVES")
files = ["output_words_0_50_adj.txt","output_words_1_50_adj.txt", "output_words_2_50_adj.txt", "output_words_3_50_adj.txt" , "output_words_4_50_adj.txt"]

male_words = []
female_words = []
for file in files:
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ' ')
        count = 0
        for row in csv_reader:
            if count >0 and count < 26:
                male_words.append(row[0])
            if count > 26:
                female_words.append(row[0])
            count +=1

counts = Counter(male_words)
print(counts)
counts = Counter(female_words)
print()
print(counts)




