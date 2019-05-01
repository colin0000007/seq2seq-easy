# seq2seq-easy

修改自https://zhuanlan.zhihu.com/p/27608348  
我认为的原代码的问题：  
（1）decoder不需要参数共享，预测和训练decode只是操作不一样，但都是使用同一个计算图中的rnn layer和projection layer的参数  
（2）原代码中预测必须把输入的数据shape 为[训练时的batchsize,source_max_len]。造成的原因是预测过程的start_tokens的batch_size  
固定为了训练时的batch_size，只需要定义一个tensor替换这个batch_size，预测时获取tensor传入数据的真实batch即可。  
## 修改  
(1) 输入tensor按照我自己写的，稍微不同  
(2) 单独将inference定义到一个函数  
(3)纠正了预测时必须使用训练的batch_size的问题  

## 依然存在的问题  
(1)代码还是有点乱  
后期可能会修改得并规范，并且加入attention，bidirectional等等。  

## 关于使用
数据的source是一个字母序列。target是一个将字母倒序的序列。  
涉及到加载数据路径在dataPreprocessing的make_source_target_list方法  
自定义包的导入可能需要修改。 

