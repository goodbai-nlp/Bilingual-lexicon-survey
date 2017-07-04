import sys

gold = dict()
test = dict()
cipher_vocab = dict()

cipher_input = open(sys.argv[1],"r")

for line in cipher_input:
  tokens = line.strip().split()
  cipher_vocab[tokens[1]] = 1

gold_input = open(sys.argv[2],"r")

for line in gold_input:
  entries = line.strip().split(" ")
  score = float(entries[2]) #[float(x) for x in entries[2].split()]
  if entries[1] not in gold:
    gold[entries[1]] = dict()
  #if score >= 0.1:
  gold[entries[1]][entries[0]]=score
  #if scores[1] > gold[entries[0]][1]:
  #  gold[entries[0]]= [entries[1],scores[1]]


test_input = open(sys.argv[3],"r")
total = 0.0
hit = 0.0
total_score = 0.0
miss = 0.0

#print gold["la"]

for line in test_input:
  entries = line.strip().split(" ||| ")
  scores = [float(x) for x in entries[2].split()]
  if entries[0] not in test:
    test[entries[0]] = dict()
  test[entries[0]][entries[1]]=scores[1]

actual = 0
unk_out = open("unk.log", "w")
for type in cipher_vocab.keys():
  if type in test and type not in gold:
    print>>unk_out,type
    for candidate in sorted(test[type], key=test[type].get,reverse=True):
        print>>unk_out,candidate,
    print>>unk_out," "
  if type in gold:
    total +=1
    if type not in test:
      miss += 1
  #elif type in test:
  #   print "new",type,test[type][0]
  if type in test and type in gold:
     count=1
     best_score=0.0
     best_candidate=""
     has_hit=False
     for candidate in sorted(test[type],key=test[type].get,reverse=True):
       if candidate in gold[type] and count <= 5 :
          if gold[type][candidate]>best_score:
             best_score=gold[type][candidate]
             best_candidate=candidate
          has_hit=True
         # total_score += gold[type][candidate]
         # print >>sys.stderr,type,candidate          
       count=count+1
     if has_hit:
       hit=hit+1
       total_score += best_score
       print >>sys.stderr,type,best_candidate
     #else:
     #  print "wrong:",type

print "vocab:",len(cipher_vocab)
print "miss:",miss
print "correct:", hit
print "total:",total
print "actual:",len(test)
print "precision:",hit/total
print "avg score:",total_score/hit
  
