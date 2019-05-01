#encoding=UTF-8
import random
import numpy as np
pad = "<pad>"
start_token = "<s>"
end_toekn = "</s>"
source_max_len = None
source_list = None
source_token_2_id = None
target_max_len= None
target_list = None
target_token_2_id = None

'''
数据预处理，生成序列的最大长度，序列词典等操作
'''
def preprocessing(source_seq_list,target_seq_list):
    source_max_len = max([len(seq) for seq in source_seq_list])
    target_max_len = max([len(seq) for seq in target_seq_list])
    source_list = list({token for seq in source_seq_list for token in seq})
    target_list = list({token for seq in target_seq_list for token in seq})
    source_list.insert(0, pad)
    target_list.insert(0, pad)
    target_list.insert(1,start_token)
    target_list.insert(2,end_toekn)
    source_token_2_id = dict(zip(source_list,[i for i in range(len(source_list))]))
    target_token_2_id = dict(zip(target_list,[i for i in range(len(target_list))]))
    #加上2是因为target_batch_x首需要<s>
    #target_batch_y末需要</s>
    target_max_len = target_max_len + 1
    print("data preprocessing:")
    print("source_seq_max_len:",source_max_len,"\nsource_dic_len:",len(source_list),"\nsource_dict:",source_list)
    print()
    print("target_seq_max_len:",target_max_len,"\ntarget_dic_len:",len(target_list),"\ntarget_dict:",target_list)
    return source_max_len,source_list,source_token_2_id,target_max_len,target_list,target_token_2_id

#将seq列表转换位int id
def source_seq_list_2_ids(source_seq_list):
    source_seq_int = [[source_token_2_id[c] for c in seq] for seq in source_seq_list]
    #保存source序列的真实长度
    source_seq_len_real = []
    #不足max的补充<pad>
    for seq in source_seq_int:
        source_seq_len_real.append(len(seq))
        for i in range(source_max_len - len(seq)):
            seq.append(source_token_2_id[pad])
    return source_seq_int,source_seq_len_real

def target_seq_list_2_ids(target_seq_list):
    target_seq_int = [[target_token_2_id[c] for c in seq] for seq in target_seq_list]
    target_seq_len_real = []
    for seq in target_seq_int:
        #首加上<s>剩余位置补充<pad>
        seq.insert(0, target_token_2_id[start_token])
        target_seq_len_real.append(len(seq))
        for i in range(target_max_len - len(seq)):
            seq.append(target_token_2_id[pad]) 
    return target_seq_int,target_seq_len_real
#source_max_len,source_dic_list,source_token_2_id,target_max_len,target_dic_list,target_token_2_id,source_seq_list, target_seq_list,
def batch_generator(source_seq_list,target_seq_list,batch_size=128,epochs=20):
    #将seq中的字符变成int，并补充<pad> <s> </s>
    source_seq_int,source_seq_len_real = source_seq_list_2_ids(source_seq_list)
    target_seq_int,target_seq_len_real = target_seq_list_2_ids(target_seq_list)
    num_sample = len(source_seq_int)
    num_batch = num_sample // batch_size
    print("num_samples:",num_sample)
    print("batch_size:",batch_size)
    print("num_batch:",num_batch)
    indices = [i for i in range(num_sample)]
    source_seq_int = np.array(source_seq_int)
    target_seq_int = np.array(target_seq_int)
    for i in range(epochs):
        #对样本索引打乱
        random.shuffle(indices)
        for j in range(num_batch):
            source_batch = [source_seq_int[index] for index in indices[j*batch_size:(j+1)*batch_size]]
            source_batch_seq_len = [source_seq_len_real[index] for index in indices[j*batch_size:(j+1)*batch_size]]
            target_batch_seq_len = [target_seq_len_real[index] for index in indices[j*batch_size:(j+1)*batch_size]]
            '''
                                    合适的应该是：
                    <s> A B C D  <pad> <pad>
                     A  B C D </s> <pad> <pad>
            '''
            target_batch_x = [target_seq_int[index] for index in indices[j*batch_size:(j+1)*batch_size]]
            #将target_batch_x向前移动一格，末尾补充pad，也就是用上一个字预测下一个字
            #好像不需要这个。。。
            
            #target_batch_y = [[seq[n] for n in range(1,target_max_len)] for seq in target_batch_x]
            target_batch_y = [[target_seq_int[index,k] for k in range(1,target_seq_len_real[index])] for index in indices[j*batch_size:(j+1)*batch_size]]
            for seq in target_batch_y:
                #每个seq后面添加</s>
                seq.append(target_token_2_id[end_toekn])
                #剩余位置添加<pad>
                ty_len = len(seq)
                for _ in range(target_max_len-ty_len):
                    seq.append(target_token_2_id[pad])
            
            yield source_batch,target_batch_x,target_batch_y,source_batch_seq_len,target_batch_seq_len,j,i

#将source文件和target文件加载进来，存储到2个list中
#只需要重写make_source_target_list这个方法就可以加载其他数据
def make_source_target_list(source_path="../data/letters_source2.txt",target_path="../data/letters_target2.txt"):
    with open(source_path,"r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        source_seq_list = [[token for token in seq] for seq in lines]
    with open(target_path,"r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        target_seq_list = [[token for token in seq] for seq in lines]
    return source_seq_list,target_seq_list


if source_max_len == None:
    #这样处理是为了被其他py导入变量时不会重复生成
    source_seq_list, target_seq_list = make_source_target_list()
    source_max_len,source_list,source_token_2_id,target_max_len,target_list,target_token_2_id = preprocessing(source_seq_list, target_seq_list)
    print("generate variables!")

'''
gen = batch_generator(source_seq_list,target_seq_list)
for source_batch,target_batch_x,target_batch_y,source_batch_seq_len,target_batch_seq_len,j,i in gen:
    print("source_batch:",source_batch)
    print("target_batch_x:",target_batch_x)
    print("target_batch_y:",target_batch_y)
    print("source_batch_seq_len:",source_batch_seq_len)
    print("target_batch_seq_len:",target_batch_seq_len)
    break
'''
