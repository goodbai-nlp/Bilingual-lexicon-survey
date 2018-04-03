## facebook 中英抽词典任务复现
---
run.sh 用于调参
## 环境需要
+ pytorch gpu 0.3
+ faiss 
## 可调的参数
+ torch.cuda.set_device(3) #记得设置gpu
+ --map_beta 正则化系数, 可tune范围 {0.01 0.001 0.0001 0.00001}
+ --dis_hid_dim 判别器隐层大小 可tune范围 {200 500 600} 建议500不动
+ --dis_dropout 判别器隐层dropout 可tune范围 {0,0.1,0.2} 可以少调
+ --dis_input_dropout 判别器输入层dropout 可tune范围 {0,0.1,0.2 0.3 0.4 0.5}
+ --dis_smooth 判别器平滑 可tune范围 {0,0.1,0.2,0.3}
+ --epoch_size 可tune范围 {100000 - 1000000} 中间可以适当插值
+ --batch_size 可tune范围 {32,64,128,256,512}
+ --lr_shrink  这个可以调一调,当学习率衰减速度太快时
+ --normalize_embeddings 经验是设成 "center",已默认,可以尝试改成空 ""
+ --number 用这个值来防止生成的图片重复 


## 其他
+ 每次模型的运行结果会总结为三张图,分别是判别器的准确率,无监督学习的loss,以及测试集上的准确率
+ 由于模型不是很稳定,每次调参后建议多跑几轮
