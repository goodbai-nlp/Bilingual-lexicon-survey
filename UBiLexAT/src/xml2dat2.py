# -*- coding: utf-8 -*-

# version 2
# select 55000 length comparable docs from 150000 docs
# preprocessing step 
# extracting english and chinese parallel corpus from wiki xml file
# outputs en.dat and zh.dat, as an input to LDA

import re
import json
import sys
import os
from nltk.corpus import stopwords
from collections import Counter


# number of doc pair need to extract for training
pair_num_trn = 50000
# number of doc pair need to extract for inference
pair_num_inf = 5000

print "Generating %d doc pairs for training, and %d pairs for inference...\n" % (pair_num_trn, pair_num_inf)

#######################################################################################
print '#############################################'
print 'Extracting contents from XML file'
print '#############################################'
en = []
zh = []
# extracting content from xml file
with open('wikicomp-2014_enzh.xml', 'r') as f:
    doc_num = 0
    # indicate whether current line a part of content or not
    content = False
    # indicate which language the current content is
    en_current = True
    # record previous two lines
    line_pre = ""
    line_prepre = ""
    # have to read 1 line a time because the file is large
    for line in f:
        # if current line a part of content 
        if content:
            # check if the end of the content
            found = re.search(r'</content>', line)
            if found:
                # if enough doc found, quit
                if doc_num == 150000*2:
                    break
                content = False
                en_current = not en_current
                continue
            # add current line to the list
            if en_current == True:
                en[-1] += line
            else:
                zh[-1] += line
        else:
            found = re.search(r'<content>', line)
            # if find the start of content
            # change state of variables
            if found:
                doc_num += 1
                content = True
                # also add article name to content
                article_name = re.search(r'name="(.+)"', line_prepre)
                if en_current == True: 
                    en.append('')
                    en[-1] += article_name.group(1)+' '
                else:
                    zh.append('')
                    zh[-1] += article_name.group(1)+' '
        line_prepre = line_pre
        line_pre = line
#######################################################################################
# only select docs in the doc_ids set

# load doc ids with comparable length
with open('doc_ids.json', 'r') as f:
    doc_ids = json.loads(f.read())

en = [en[i] for i in doc_ids]
zh = [zh[i] for i in doc_ids]
#######################################################################################
# remove xml elements from extracted content
print '#############################################'
print 'Removing XML elements from extracted contents'
print '#############################################'
# remove link 
en = [re.sub(r'<link.+?>(.*?)</link>', r'\1', s) for s in en]
zh = [re.sub(r'<link.+?>(.*?)</link>', r'\1', s) for s in zh]
# remove table
en = [re.sub(r'<table>[\w\W]+?</table>', '', s) for s in en]
zh = [re.sub(r'<table>[\w\W]+?</table>', '', s) for s in zh]
# remove math
en = [re.sub(r'<math>[\w\W]+?</math>', '', s) for s in en]
zh = [re.sub(r'<math>[\w\W]+?</math>', '', s) for s in zh]
# remove &quot
en = [re.sub(r'&?quot;', ' ', s) for s in en]
zh = [re.sub(r'&?quot;', ' ', s) for s in zh]
# remove &apos
en = [re.sub(r'&?apos;', ' ', s) for s in en]
zh = [re.sub(r'&?apos;', ' ', s) for s in zh]
# remove &amp
en = [re.sub(r'&?amp;', ' ', s) for s in en]
zh = [re.sub(r'&?amp;', ' ', s) for s in zh]
# remove reference and the thing after it
en = [re.sub(r'<p><h>References[\w\W]*</p>$', '', s) for s in en]
zh = [re.sub(r'<p><h>參考文獻[\w\W]*</p>$', '', s) for s in zh]
zh = [re.sub(r'<p><h>参考文献[\w\W]*</p>$', '', s) for s in zh]
zh = [re.sub(r'<p><h>參考資料[\w\W]*</p>$', '', s) for s in zh]
zh = [re.sub(r'<p><h>資料來源[\w\W]*</p>$', '', s) for s in zh]
zh = [re.sub(r'<p><h>外部链接[\w\W]*</p>$', '', s) for s in zh]
zh = [re.sub(r'<p><h>参见[\w\W]*</p>$', '', s) for s in zh]
zh = [re.sub(r'<p><h>参考[\w\W]*</p>$', '', s) for s in zh]
# remove <p>
en = [re.sub(r'<p>', '', s) for s in en]
zh = [re.sub(r'<p>', '', s) for s in zh]
# remove </p>
en = [re.sub(r'</p>', '', s) for s in en]
zh = [re.sub(r'</p>', '', s) for s in zh]
# remove <h>
en = [re.sub(r'<h>', '', s) for s in en]
zh = [re.sub(r'<h>', '', s) for s in zh]
# remove </h>
en = [re.sub(r'</h>', '', s) for s in en]
zh = [re.sub(r'</h>', '', s) for s in zh]
# remove http address
en = [re.sub(r'http://.*', '', s) for s in en]
zh = [re.sub(r'http://.*', '', s) for s in zh]
#######################################################################################
# transform traditional chinese to simplified chinese
print '#############################################'
print 'Transforming traditional chinese to simplified chinese'
print '#############################################'
# opencc is used
for i in range(len(zh)):
    with open('test_in.txt', 'w') as f:
        f.write(zh[i])
    os.system('cat test_in.txt | opencc -o test_out.txt -c zht2zhs.ini')
    with open('test_out.txt', 'r') as f:
        zh[i] = f.read()

os.system('rm test_in.txt')
os.system('rm test_out.txt')
#######################################################################################
# do words segmentation for chinese
print '#############################################'
print 'Doing words segmentation for chinese'
print '#############################################'
# stanford segmenter is used
with open('zhs.txt', 'w') as f:
    for doc in zh:
        f.write(doc)
        f.write('#######\n')
# do words segmentationby calling stanford segmenter
os.system('../stanford-segmenter-2014-08-27/segment.sh ctb zhs.txt UTF-8 0 > zhs_seg.txt')

os.system('rm zhs.txt')

# read in segmented texts
with open('zhs_seg.txt', 'r') as f:
    zh = f.read()
# transform to unicode
zh = zh.decode('utf-8')

zh = zh.split('#######')
zh = zh[:-1]

os.system('rm zhs_seg.txt')
#######################################################################################
# filter for chinese
print '#############################################'
print 'Filtering chinese'
print '#############################################'

# only keep chinese words and space
def keep(w):
    if w == ' ':
        return True
    if w >=u'\u4e00' and w <= u'\u9fa5':
        return True
    return False

# remove characters other than chinese words and space
zh = [[w if keep(w) else ' ' for w in s] for s in zh]

# import chinese stopwords
with open('zh_stopwords.json', 'r') as f:
    zh_stopwords = json.loads(f.read())
zh_stopwords = set(zh_stopwords)
# remove stopwords
zh = [[w for w in s if not w in zh_stopwords] for s in zh]

zh = ["".join(s) for s in zh]

# remove unnecessary space
zh = [re.sub(r'\s+', ' ', s) for s in zh]
zh = [s.strip() for s in zh]

zh = [s.split() for s in zh]

# remove low-frequency words
zh_words = [w for s in zh for w in s]
zh_counter = Counter(zh_words)
for key in zh_counter.keys():
    if zh_counter[key] < 5:
        del zh_counter[key]
zh_words_set = set(zh_counter.keys())
zh = [[w for w in s if w in zh_words_set] for s in zh]

zh = [' '.join(s) for s in zh]

## in case there is empty document
zh = [s if len(s)>0 else u'空缺' for s in zh]

# split to training and inference files
zh_trn = zh[:pair_num_trn]
zh_inf = zh[pair_num_trn:]

## save training file in json format
#with open('zh_'+str(pair_num_trn)+'.json', 'w') as f:
#    f.write(json.dumps(zh_trn))
## save inference file in json format
#with open('zh_'+str(pair_num_inf)+'_inf'+'.json', 'w') as f:
#    f.write(json.dumps(zh_inf))

# each document per line
zh_trn = '\n'.join(zh_trn)
zh_inf = '\n'.join(zh_inf)

# save training file in plain text
with open('zh_'+str(pair_num_trn)+'.dat', 'w') as f:
    f.write(str(pair_num_trn)+'\n')
    f.write(zh_trn.encode('utf-8'))
# save inference file in plain text
with open('zh_'+str(pair_num_inf)+'_inf'+'.dat', 'w') as f:
    f.write(str(pair_num_inf)+'\n')
    f.write(zh_inf.encode('utf-8'))
######################################################################################
# filter for english
print '#############################################'
print 'Filtering english'
print '#############################################'

# remove non-character
en = [re.sub(r'[\W\d_]', ' ', s) for s in en]
# remove unnecessary space
en = [re.sub(r'\s+', ' ', s) for s in en]
# transform to lowercase
en = [s.lower() for s in en]

# remove stopwords
en_stopwords = set(stopwords.words('english'))
en = [s.split() for s in en]
en = [[w for w in s if not w in en_stopwords] for s in en]
# remove words with length 1 and 2
en = [[w for w in s if len(w)>2] for s in en]

# remove low-frequency words
en_words = [w for s in en for w in s]
en_counter = Counter(en_words)
for key in en_counter.keys():
    if en_counter[key] < 5:
        del en_counter[key]
en_words_set = set(en_counter.keys())
en = [[w for w in s if w in en_words_set] for s in en]

en = [' '.join(s) for s in en]

# in case there is empty document
en = [s if len(s)>0 else u'missing' for s in en]

# split to training and inference files
en_trn = en[:pair_num_trn]
en_inf = en[pair_num_trn:]

#with open('en_'+str(pair_num_trn)+'.json', 'w') as f:
#    f.write(json.dumps(en_trn))
#with open('en_'+str(pair_num_inf)+'_inf'+'.json', 'w') as f:
#    f.write(json.dumps(en_inf))

## each document per line
#en_trn1 = en_trn[:len(en_trn)/2]
#en_trn1 = '\n'.join(en_trn1)
#en_trn1 += '\n'
#en_trn2 = en_trn[len(en_trn)/2:]
#en_trn2 = '\n'.join(en_trn2)
#en_inf = '\n'.join(en_inf)

with open('en_'+str(pair_num_trn)+'.dat', 'w')as f:
    f.write(str(pair_num_trn)+'\n')
    for doc in en_trn:
        f.write(doc.encode('utf-8'))
        f.write('\n')
with open('en_'+str(pair_num_inf)+'_inf'+'.dat', 'w')as f:
    f.write(str(pair_num_inf)+'\n')
    for doc in en_inf:
        f.write(doc.encode('utf-8'))
        f.write('\n')
