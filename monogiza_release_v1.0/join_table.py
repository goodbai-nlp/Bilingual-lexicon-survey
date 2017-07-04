import sys

table = dict()

for line in sys.stdin:
  entries = line.strip().split(" ||| ")
  pair = " ||| ".join(entries [0:2])
  scores = entries[2].split()
  if pair not in table:
    table[pair]= [float(scores[0]),float(scores[1]),1.0]
  else:
    table[pair][0] += float(scores[0])
    table[pair][1] += float(scores[1])
    table[pair][2] += 1

for key in table.keys():
  if table[key][2]>1:# and ((table[key][1]/table[key][2])>=0.1 or (table[key][0]/table[key][2])>= 0.1):
    print key + " ||| " +str(table[key][0]/table[key][2]) + " "+ str(table[key][1]/table[key][2]) #+" "+str(table[key][2])
