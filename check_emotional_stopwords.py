import time

stop_words = []
custom = []
female_words = []
male_words = []
f = open('stopwords_saracody.py')
current_list = 0
for line in f:
  line = line.strip()
  if line.startswith("'"):
    line = line[1:-2]
    if current_list == 0:
      stop_words.append(line)
    elif current_list == 1:
      custom.append(line)
    elif current_list == 2:
      male_words.append(line)
    elif current_list == 3:
      female_words.append(line)
    else:
      print("SOMETHING WENT WRONG")
  elif line.startswith('#'):
    if "custom" in line:
      current_list+=1
    elif "male" in line:
      current_list +=1
    elif "female" in line:
      current_list += 1 

f.close()


NEG_WORDS = []
POS_WORDS = []

f = open('opinion-lexicon-English/negative-words.txt', 'rb')
for line in f:
  if not line.startswith(bytes(';')):
    line = line.strip()
    NEG_WORDS.append(line)
    #print(type(line))
f.close()
NEG_WORDS = set(NEG_WORDS)

f = open('opinion-lexicon-English/positive-words.txt', 'rb')
for line in f:
  if not line.startswith(';'):
    line = line.strip()
    POS_WORDS.append(line)
f.close()
POS_WORDS = set(POS_WORDS)

print("STOP WORDS")
tot = 0
pos_cnt = 0
neg_cnt = 0
for item in stop_words:
  if item in POS_WORDS:
    pos_cnt += 1
    print("Pos:", item)
  elif item in NEG_WORDS:
    neg_cnt += 1
    print("Neg:", item)
  tot += 1

print("Stop Words", tot, pos_cnt, neg_cnt)

print("CUSTOM")
tot = 0
pos_cnt = 0
neg_cnt = 0
for item in custom:
  if item in POS_WORDS:
    pos_cnt += 1
    print("Pos:", item)
  elif item in NEG_WORDS:
    neg_cnt += 1
    print("Neg:", item)
  tot += 1

print("Custom", tot, pos_cnt, neg_cnt)

print("MALE")
tot = 0
pos_cnt = 0
neg_cnt = 0
for item in male_words:
  if item in POS_WORDS:
    pos_cnt += 1
    print("Pos:", item)
  elif item in NEG_WORDS:
    neg_cnt += 1
    print("Neg:", item)
  tot += 1

print("Male", tot, pos_cnt, neg_cnt)


print("FEMALE")
tot = 0
pos_cnt = 0
neg_cnt = 0
for item in female_words:
  if item in POS_WORDS:
    pos_cnt += 1
    print("Pos:", item)
  elif item in NEG_WORDS:
    neg_cnt += 1
    print("Neg:", item)
  tot += 1

print("Female", tot, pos_cnt, neg_cnt)
