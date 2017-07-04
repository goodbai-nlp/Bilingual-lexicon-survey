import sys

f_map=dict()
e_map=dict()
freq_map=dict()
ptable=dict()

def load_map(dic,file):
  input=open(file,"r")
  id = 1
  for line in input:
    entries=line.strip().split()
    dic[str(id)]=entries[1]
    id += 1

def load_freq(dic,file):
  input=open(file,"r")
  for line in input:
    entries=line.strip().split()
    freq_map[entries[0]]=entries[1]


load_map(f_map,sys.argv[1])
load_map(e_map,sys.argv[2])
# start to convert


input=open(sys.argv[3],"r")
output = open(sys.argv[4],"w")

for line in input:
  entries=line.strip().split(" ||| ")
  print >>output, f_map[entries[0]]+" ||| "+e_map[entries[1]]+" ||| "+entries[2]
