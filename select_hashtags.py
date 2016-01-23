import sys
import io
import json

dictPath = sys.argv[1]
MIN_COUNT = int(sys.argv[2])
MAX_COUNT = int(sys.argv[3])
outPath = sys.argv[4]

# read in the tweets and filter
ntags = 0
with io.open(dictPath,'r',encoding='utf-8') as f_in, io.open('%s/data.txt'%outPath,'w',encoding='utf-8') as f_out:
    for line in f_in:
        current = json.loads(line)
        count = len(current[1])
        if count >= MIN_COUNT and count < MAX_COUNT:
            ntags += 1
            for item in current[1]:
                f_out.write('%s\t%s\n' %(current[0],item))
open('%s/ntags.txt'%outPath,'w').write('%d'%ntags)
