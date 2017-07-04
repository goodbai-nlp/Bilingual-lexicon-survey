
CIPHER_BIGRAM=/home/nlg-02/qingdou/cipher_data/graph_data/ch_bigram.300
SEED=0 #seed for random number generator
LM=/home/nlg-02/qingdou/cipher_data/graph_data/train.id.lm
SORTED_LIST=/home/nlg-02/qingdou/cipher_data/graph_data/slice_en.dat
SORTED_LIST_SIZE=2000
DIMENSION=50
OPTITR=5  # number of iterations for gradient descent
LEARNING_RATE=10 # step size
REG=0.001 # regularizer
INTERVAL=20000 # how many sampling iterations to be done between two M step
ALPHA=1.0
VOCAB_SIZE=10001  # vocab size, used to set mapping matrix size
BASE_T=0.002
FAST_MODE=1  # 0 for performing true slice sampling, 1 for doing approximated slice sampling
ITR=80000 # total number of iterations
NUM_THREADS=1 # number of threads for sampler
PLAIN_EMBEDDINGS=/home/nlg-05/qingdou/GITHUB/DECIPHERMENT_WITH_EMBEDDINGS/vectors.s$DIMENSION.10k.en
CIPHER_EMBEDDINGS=/home/nlg-05/qingdou/GITHUB/DECIPHERMENT_WITH_EMBEDDINGS/vectors.s$DIMENSION.10k.es
OUT_TTABLE=ptable.final # output ttable file name
OUT_MAPPING=mapping.final  # output mapping matrix file
USE_UNIFORM_BASE=0 #
USE_EMBEDDINGS=0 #
BASE_FILE=/home/nlg-02/qingdou/cipher_data/graph_data/prior.txt
MAP_SEED=/scratch/m.seed # use seed mapping matrix, you can use mapping matrix learned from previous decipherment
TTABLE_SEED=/scratch/ttable.seed # seed table used to initialize the first sample


/home/nlg-05/qingdou/GITHUB/DECIPHERMENT_WITH_EMBEDDINGS/src/slice_with_embeddings_test --output_ttable $OUT_TTABLE --output_mapping $OUT_MAPPING --iteration $ITR --lm $LM --sorted_list $SORTED_LIST --sorted_list_size $SORTED_LIST_SIZE --reg $REG --learning_rate $LEARNING_RATE --cipher_bigrams $CIPHER_BIGRAM --random_seed $SEED --mapping_seed $MAP_SEED --seed_table $TTABLE_SEED --num_threads $NUM_THREADS --plain_embeddings $PLAIN_EMBEDDINGS --fast_mode $FAST_MODE --base_threshold $BASE_T --use_uniform_base $USE_UNIFORM_BASE --use_embeddings $USE_EMBEDDINGS --base_file $BASE_FILE --cipher_embeddings $CIPHER_EMBEDDINGS --dimension $DIMENSION --m_iteration $OPTITR --interval $INTERVAL --alpha $ALPHA --vocab_size $VOCAB_SIZE &> slice.graph_decipher.log
 
