import sys
import json
import matplotlib.pyplot as plt
import codecs

from collections import defaultdict

MIN_COUNT = 100

dictPath = sys.argv[1]
N = int(sys.argv[2])
C = int(sys.argv[3])

# create dictionary of hashtags and tweet counts
counts = defaultdict(int)
total = 0
below_thresh = 0
with codecs.open(dictPath,'r',encoding='utf-8') as d:
    for line in d:
        row = json.loads(line)
        counts[row[0]] = row[1]
        total += row[1]
        if row[1] < MIN_COUNT:
            below_thresh += row[1]

# sort in descending order
sorted_tags = sorted(counts, key=lambda k:counts[k], reverse=True)
sorted_counts = [counts[ii] for ii in sorted_tags]

print("Total number of hashtags = {}".format(len(sorted_counts)))

# print top N
print("Top - ")
top_sum = 0
for i in range(N):
    print(u'{} - {}'.format(sorted_tags[i],sorted_counts[i]))
    top_sum += sorted_counts[i]

next_sum = 0
for i in range(N,N+C):
    if sorted_counts[i] >= MIN_COUNT:
        next_sum += sorted_counts[i]

print("Total number of tweets = {}".format(total))
print("Below threshold = {}".format(below_thresh))
print("Sum of top {} = {}".format(N, top_sum))
print("Sum of next {} = {}".format(C, next_sum))
print("Remaining = {}".format(total-top_sum-next_sum-below_thresh))

# plot values
plt.figure()
plt.plot(sorted_counts)
full = plt.axvline(x=sorted_counts.index(MIN_COUNT),c='r',linestyle='dashed')
maxx = plt.axvline(x=sorted_counts.index(4407),c='k')
small = plt.axvline(x=sorted_counts.index(1099),c='k',linestyle='dashed')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Hashtag')
plt.ylabel('Number of tweets')
plt.title('Frequency of tweets per hashtag')
plt.legend([maxx,full,small],['max limit','full dataset','small dataset'])
plt.show()

# print bottom N
#print("Bottom - ")
#bot_sum = 0
#for i in range(1,N+1):
#    print(u'{} - {}'.format(sorted_tags[-i],sorted_counts[-i]))
#    bot_sum += sorted_counts[-i]
#print("Sum of bottom {} = {}".format(N, bot_sum))

# tweets above count 5
#s = 0
#for c in sorted_counts:
#	if c > 50:
#		s+=c
#print("Number of tweets above threshold = {} (total {})".format(s,total))

