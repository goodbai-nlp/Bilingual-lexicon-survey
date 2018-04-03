for((index=32;index<=256;index*=2))
do
	for((i=0; i<8; ++i))
	do 
		python3 unsupervised.py --src_emb data/word2vec.zh --tgt_emb data/word2vec.en --number $i --batch_size $index 
	done
done	
