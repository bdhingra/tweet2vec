from guineapig import *

# parse params
params = GPig.getArgvParams()
in_file = params['input']

class CombineTags(Planner):
    combined = ReadLines(in_file) | ReplaceEach(by=lambda line:line.rstrip('\n').split('\t')) | Group(by=lambda (h,t):t, retaining=lambda (h,t):h) | Format(by=lambda (t,hlist):'%s\t%s'%(','.join([h for h in hlist]),t))

if __name__ == "__main__":
    planner = CombineTags()
    planner.main(sys.argv)
