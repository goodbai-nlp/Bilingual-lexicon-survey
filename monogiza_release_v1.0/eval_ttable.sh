GOLD_TABLE=$1
TEST_TABLE=$2

python extract_ptable.py vocab.giga.es.top10k vocab.giga.en.top10k $TEST_TABLE >ptable.wfreq

python eval_ttable.top5.py vocab.giga.es.top5k $GOLD_TABLE ptable.lex

