import sys
import io

data_path = sys.argv[1]
out_path = sys.argv[2]

with io.open(data_path,'r',encoding='utf-8') as f, io.open(out_path,'w',encoding='utf-8') as f_o:
    it = 0
    for line in f:
        it += 1
        if it%100000 == 0:
            print("On line {}".format(it))
        row = line.rstrip('\n').split('\t')
        if len(row) != 2:
            continue
        if len(row[1].split()) < 2:
            continue
        f_o.write(line)
