#!/usr/bin/env bash

CIPHER_BIGRAM=bigram.f
PLAIN_BIGRAM=bigram.e
#SEED=$RANDOM #seed for random number generator
LM=train.e.lm_0
SORTED_LIST=slice.e.dat_0
SORTED_LIST_SIZE=2000
SRILM_DIR=3rdparty/srilm/bin/i686
VOCAB_SIZE=10000
if [ `getconf LONG_BIT` == 64 ]
then 
    SRILM_DIR=3rdparty/srilm/bin/i686-m64
fi
OUT_TTABLE=ttable.id # output ttable file name

if [ $1 == "--help" ]
then 
    ./slice_with_embeddings --help
    exit
fi

tr -d '\r' < $1 | tr ' ' '\n' | sort | uniq -c | sort -nr > vocab.f
tr -d '\r' < $2 | tr ' ' '\n' | sort | uniq -c | sort -nr > vocab.e
PLAIN_VOCAB_SIZE=`wc -l < vocab.e`
CIPHER_VOCAB_SIZE=`wc -l < vocab.f`

if [ $PLAIN_VOCAB_SIZE -gt $CIPHER_VOCAB_SIZE ]
then
    VOCAB_SIZE=$PLAIN_VOCAB_SIZE
else
    VOCAB_SIZE=$CIPHER_VOCAB_SIZE
fi

if [ $PLAIN_VOCAB_SIZE -lt 2000 ]
then
    SORTED_LIST_SIZE=$PLAIN_VOCAB_SIZE
fi
echo "Plaintext vocabulary size: $SORTED_LIST_SIZE"

python get_bigrams.py $1 vocab.f > bigram.f
python get_bigrams.py $2 vocab.e > bigram.e

echo "Building language model from plaintext bigram $2 ..."
$SRILM_DIR/ngram-count -text $PLAIN_BIGRAM -text-has-weights -order 2 -lm $LM

echo "Building sorted list from language model $LM ..."
java -jar -Xmx10g Build_List.jar $LM $SORTED_LIST_SIZE > $SORTED_LIST


echo "Starting decipherment ..."

./slice_with_embeddings --output_ttable $OUT_TTABLE --lm $LM --sorted_list $SORTED_LIST --sorted_list_size $SORTED_LIST_SIZE --cipher_bigrams $CIPHER_BIGRAM --vocab_size $VOCAB_SIZE ${@:4}

echo "Writing final ttable to $3"
python extract_ptable.py vocab.f vocab.e ttable.id $3
