import sys

vocab=dict()

vocab_file=open(sys.argv[2], "r")
id=1
for line in vocab_file:
    entries = line.strip().split()
    vocab[entries[1]] = id
    id += 1

counts=dict()
text_file=open(sys.argv[1],"r")
for line in text_file:
    entries = line.strip().split()
    for i in range(1,len(entries) - 1):
        bigram = str(vocab[entries[i - 1]]) + " " + str(vocab[entries[i]])
        if bigram in counts:
            counts[bigram] += 1
        else:
            counts[bigram] = 1

for bigram in counts:
    print str(counts[bigram]) + "\t" + bigram
