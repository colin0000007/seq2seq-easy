#encoding:UTF-8
import tensorflow as tf
from tensorflow.python.keras.layers.core import Dense
from Myseq2seq.dataPreprocessing import batch_generator
from Myseq2seq.dataPreprocessing import source_list,target_list,source_seq_list,target_seq_list,target_max_len,source_max_len,target_token_2_id
import numpy as np
'''
所以数据直接从dataPreprocessing中导入
'''
pad = "<pad>"
start_token = "<s>"
end_token = "</s>"
#定义encoder的输入tensor，加入了词嵌入
def input_tensor(batch_size,source_time_step,target_time_step,source_vocab_size,source_embedding_size,target_vocab_size,target_embedding_size):
    #[time_step,batch_size]这样是为了实则设置time_major=true，这样可以使得效率提高
    #这里设置位None，意思时不指定batch_size用到多少就是多少？还不确定
    source_batch = tf.placeholder(tf.int32,[None,source_time_step],name="source_batch")
    #对source做词嵌入，词向量矩阵的shape为[source的词库大小,嵌入维度]
    #矩阵中每一行就是一个词向量
    source_embedding = tf.get_variable(shape=[source_vocab_size,source_embedding_size],name='source_embedding')
    #使用embedding_lookup从embedding矩阵中查询词向量从而将X的每一个单词的index转换一个词向量
    embedded_source_batch = tf.nn.embedding_lookup(source_embedding,source_batch)
    #最后返回的embedded_X.shape = [time_step,batch_size,embedding_size]
    #target类似
    target_batch_x = tf.placeholder(tf.int32,[batch_size,target_time_step])
    target_batch_y = tf.placeholder(tf.int32,[batch_size,target_time_step])
    target_embedding = tf.get_variable(shape=[target_vocab_size,source_embedding_size],name='target_embedding')
    embedded_target_batch_x = tf.nn.embedding_lookup(target_embedding,target_batch_x)
    source_batch_seq_len = tf.placeholder(tf.int32,[None],name="source_batch_seq_len")
    target_batch_seq_len = tf.placeholder(tf.int32,[batch_size],name="target_batch_seq_len")
    return embedded_source_batch,embedded_target_batch_x,source_batch,target_batch_x,target_batch_y,source_batch_seq_len,target_batch_seq_len,target_embedding

#参照tensorflow/nmt的官方教程
def build_encoder(embedded_source_batch,source_batch_seq_len,rnn_layer_size=2,rnn_num_units=128):
    #定义rnn cell的获取
    def get_rnn_cell():
        return tf.nn.rnn_cell.LSTMCell(num_units=rnn_num_units)
    #定义encoder的rnn_layer
    rnn_layer = tf.nn.rnn_cell.MultiRNNCell([get_rnn_cell() for _ in range(rnn_layer_size)])
    #将rnn沿时间序列展开
    #   encoder_outputs: [batch_size,max_time, num_units]
    #   encoder_state: [batch_size, num_units]
    #   sequence_length:传入一个list保存每个样本的序列的真实长度，教程中说这样做有助于提高效率
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    rnn_layer, embedded_source_batch,
    sequence_length=source_batch_seq_len, time_major=False,dtype=tf.float32)
    #注意返回的encoder_outputs返回了每一个序列节点的output shape为[max_time, batch_size, num_units]
    #返回的encoder_state返回了每个样本最后一个序列节点的隐状态/cell值，猜测是这样，是否是待定
    return encoder_outputs, encoder_state

'''
decoder怎样使用source的信息或者说语义向量
最简单的办法就是把样本最后一个序列节点的hidden state传递给decoder
'''    

def build_decoder(encoder_outputs, encoder_state,embedded_target_batch_x,target_batch_seq_len,target_embedding,start_token_id,end_token_id,rnn_layer_size=2,rnn_num_units=128):
    def get_rnn_cell():
        return tf.nn.rnn_cell.LSTMCell(num_units=rnn_num_units)
    rnn_layer = tf.nn.rnn_cell.MultiRNNCell([get_rnn_cell() for _ in range(rnn_layer_size)])
    #decoder：需要helper，rnn cell，helper被分离开可以使用不同的解码策略，比如预测时beam search，贪婪算法
    #这里projection_layer就是一个全连接层，encoder_state连接到encoder_embede后的向量维度不能和target词汇数量一致所以需要映射层
    #不知道为什么这里使用tensorflow.python.keras.layers.core.Dense一直报错
    #换成tf.layers.Dense后解决 但是我在另一份能够运行的seq2seq代码中发现使用上面的Dense并不会报错
    projection_layer = tf.layers.Dense(units=target_dic_size,kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    #with tf.variable_scope("decode"):
        #训练用到的training helper
        #这里train helper
    training_helper = tf.contrib.seq2seq.TrainingHelper(
        embedded_target_batch_x, target_batch_seq_len, time_major=False)
    decoder = tf.contrib.seq2seq.BasicDecoder(
        rnn_layer, training_helper, encoder_state,
        output_layer=projection_layer)
    #将序列展开
    decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder,output_time_major=False)
    #decoder_output的shape:[batch_size,序列长度,target_vocab_size]
    #decoder参数共享
    #with tf.variable_scope("decode", reuse=True):
        #按照官方GreedyEmbeddingHelper的文档，需要传入一个shape为batchsize大小的start_token，类型为int32
        #好像这里的batch size就决定了预测时的batch 大小，必须符合才行
    #原文中的inference部分已经被inference函数替换，强烈不建议使用
    '''
    start_tokens = tf.constant(start_token_id, dtype=tf.int32, shape=[batch_size], name='start_tokens')
    predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(target_embedding,start_tokens,end_token_id)
    predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
        rnn_layer, predicting_helper, encoder_state,
        output_layer=projection_layer)
    predicting_decoder_output, _ ,_= tf.contrib.seq2seq.dynamic_decode(predicting_decoder,output_time_major=False,maximum_iterations=target_max_len)
    '''
    return decoder_output,rnn_layer,projection_layer

#推理阶段，也就是预测阶段
def inference(start_token_id,target_embedding,end_token_id,encoder_state,decoder_rnn_layer,decoder_projection_layer,target_max_len):
    inuput_batch = tf.placeholder(dtype=tf.int32, shape=[1],name="input_batch")
    start_tokens = tf.tile(tf.constant(value=start_token_id, dtype=tf.int32,shape=[1]), multiples = inuput_batch, name="start_tokens_2")
    #start_tokens = tf.placeholder(dtype=tf.int32, shape=[None], name="start_tokens_2")
    predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(target_embedding,start_tokens,end_token_id)
    predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_rnn_layer, predicting_helper, encoder_state,
        output_layer=decoder_projection_layer)
    predicting_decoder_output, _ ,_= tf.contrib.seq2seq.dynamic_decode(predicting_decoder,output_time_major=False,maximum_iterations=target_max_len)
    tf.identity(predicting_decoder_output.sample_id, name='predictions2')
    
def seq2seq_model(embedded_source_batch,source_batch_seq_len,embedded_target_batch_x,target_batch_seq_len,target_embedding,encoder_rnn_layer_size=2,encoder_rnn_num_units=128,decoder_rnn_layer_size=2,decoder_rnn_num_units=128):
    encoder_outputs, encoder_state = build_encoder(embedded_source_batch, source_batch_seq_len, encoder_rnn_layer_size, encoder_rnn_num_units)
    decoder_output,predicting_decoder_output,rnn_layer,projection_layer = build_decoder(encoder_outputs, encoder_state, embedded_target_batch_x, target_batch_seq_len, target_embedding, decoder_rnn_layer_size, decoder_rnn_num_units)
    print("encoder_outputs:",encoder_outputs)
    print("encoder_state:",encoder_state)
    print("decoder_output:",decoder_output)
    print("predicting_decoder_output",predicting_decoder_output)
    return decoder_output,predicting_decoder_output,rnn_layer,projection_layer,encoder_state

#一系列超参数
lr = 0.001
batch_size = 128
epochs = 60
source_embedding_size = 15
target_embedding_size = 15
source_dic_size = len(source_list)
target_dic_size = len(target_list)
encoder_rnn_layer_size = 2
encoder_rnn_num_units = 128
decoder_rnn_layer_size = 2
decoder_rnn_num_units = 128


def build_graph():
    #构造计算图
    train_graph = tf.Graph()
    with train_graph.as_default():
        #1.tensor声明
        embedded_source_batch,embedded_target_batch_x,source_batch,target_batch_x,target_batch_y,source_batch_seq_len,target_batch_seq_len,target_embedding = input_tensor(batch_size, source_max_len, target_max_len, source_dic_size, source_embedding_size, target_dic_size, target_embedding_size)
        #2.构造seq2seq产生的output tensor
        decoder_output,rnn_layer,projection_layer,encoder_state = seq2seq_model(embedded_source_batch, source_batch_seq_len, embedded_target_batch_x, target_batch_seq_len, target_embedding, encoder_rnn_layer_size, encoder_rnn_num_units, decoder_rnn_layer_size, decoder_rnn_num_units)
        #1和2这个步骤必须在同一个graph下声明
        #对这2个decoder的输出获取不同的tensor并且取名字
        training_logits = tf.identity(decoder_output.rnn_output, 'logits')
        #predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')
        print("predicting_logits.shape:",predicting_logits.shape)
        print("training_logits.shape:",training_logits.shape)
        #尝试是否能成功
        inference(1, target_embedding, 2, encoder_state, rnn_layer, projection_layer, target_max_len)
        #mask的作用是：计算loss时忽略pad的部分，这部分的loss不需要算，提升性能，
        masks = tf.sequence_mask(target_batch_seq_len, target_max_len, dtype=tf.float32, name='masks')
        with tf.name_scope("optimization"):
            #loss
            #这里尝试使用下tensorflow-nmt官方教程中的tf.nn.sparse_softmax_cross_entropy_with_logits
            #不测试了，不能使用mask
            loss = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                target_batch_y,
                masks)
            optimizer = tf.train.AdamOptimizer(lr)
            # Gradient Clipping
            gradients = optimizer.compute_gradients(loss)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
    return train_graph,loss,train_op,source_batch,target_batch_x,target_batch_y,source_batch_seq_len,target_batch_seq_len

#训练
def train():
    train_graph,loss,train_op,source_batch,target_batch_x,target_batch_y,source_batch_seq_len,target_batch_seq_len = build_graph()
    #训练步骤
    checkpoint = "./my_seq2seq_model.ckpt"
    with tf.Session(graph=train_graph) as sess:
        #tensor初始化
        sess.run(tf.global_variables_initializer())
        #获取数据生成器
        generator = batch_generator(source_seq_list, target_seq_list, batch_size, epochs)
        for src_batch,tgt_batch_x,tgt_batch_y,src_batch_seq_len,tgt_batch_seq_len,batch_num,epoch_num in generator:
            b_loss,_ = sess.run([loss,train_op],feed_dict={
                source_batch:src_batch
                ,target_batch_x:tgt_batch_x
                ,target_batch_y:tgt_batch_y#np.array(tgt_batch_y).reshape(target_max_len,batch_size)
                ,source_batch_seq_len:src_batch_seq_len
                ,target_batch_seq_len:tgt_batch_seq_len
                })
            print("epoch:",epoch_num,"/",epochs,"batch:",batch_num,"loss:",b_loss)
        # 保存模型
        saver = tf.train.Saver()
        saver.save(sess, checkpoint)
if __name__ == "__main__":
    train()      