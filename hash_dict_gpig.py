from guineapig import *
import json

#parse params
params = GPig.getArgvParams()
in_file= params['input']

# helper functions
def parseLine(line):
    ht, text = line.rstrip('\n').split('\t')
    hts = ht.split(',')
    return (hts, text)

def split_hashtags(row):
    (ht, text) = row
    for h in ht:
        yield (h,text)

# main
class HashDict(Planner):
    hashtags = ReadLines(in_file) | ReplaceEach(by=lambda line:parseLine(line)) | Flatten(by=lambda row:split_hashtags(row)) | Distinct()
    hashdict = Group(hashtags, by=lambda (h,t):h, retaining=lambda (h,t):t) | Format(by=lambda row:json.dumps(row))
    hashcount = Group(hashtags, by=lambda (h,t):h, reducingTo=ReduceToCount()) | Format(by=lambda row:json.dumps(row))

    splittags = Format(hashtags, by=lambda (h,t):'%s\t%s'%(h,t))

if __name__ == "__main__":
    planner = HashDict()
    planner.main(sys.argv)
