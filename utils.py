import numpy as np
from sklearn import preprocessing


#用于将数据标准化
def data_preprocessing(data_in,mean=None, std=None):
    if mean is None and std is None:
        mean_out = np.mean(data_in,axis=0)
        std_out = np.std(data_in,axis=0)
        return mean_out, std_out, preprocessing.scale(data_in)
    else:
        if mean is None or std is None:
            raise ValueError("mean or std is lacked")
        data_out = (data_in-mean)/std
        return data_out


#用于将数据标准化
def data_standardization(data_in, mean=None, std=None):
    if mean is None and std is None:
        mean_out = np.mean(data_in,axis=0)
        std_out = np.std(data_in,axis=0)
        return mean_out, std_out, preprocessing.scale(data_in)
    else:
        if mean is None or std is None:
            raise ValueError("mean or std is lacked")
        data_out = (data_in-mean)/std
        return data_out


#用于将数据归一化
def data_normalization(data_in, min_in=None, max_in=None):
    if min_in is None and max_in is None:
        min_out = np.min(data_in,axis=0)
        max_out = np.max(data_in,axis=0)
        return min_out, max_out, preprocessing.minmax_scale(data_in)
    else:
        if min_in is None or max_in is None:
            raise ValueError("mean or std is lacked")
        data_out = (data_in-min_in)/(max_in-min_in)
        return data_out
    
    
    
#将原始数据序列化
def data_sequence(data_in, n_seq):
    #返回数据格式 n_seq* n_batch* n_dim 
    data_seq = []
    n_len = np.size(data_in, axis=0)
    for i in range(n_seq):
        data_seq_mid = data_in[i:n_len-(n_seq-i),:]
        data_seq.append(data_seq_mid)
    data_out = np.array(data_seq)
    return data_out


def data_sequence_matrix(data_in, n_seq):
    #返回数据格式 n_batch* (n_seq*n_dim) 
    data_seq = []
    n_len = np.size(data_in, axis=0)
    for i in range(n_seq):
        data_seq_mid = data_in[i:n_len-(n_seq-i),:]
        data_seq.append(data_seq_mid)
        
    data_out = np.concatenate(data_seq, axis=1)
    return data_out




def data_sequence_pf(data_in, n_seq_p, n_seq_f):
    #返回过去时间段的序列数据与未来时间段的序列数据
    '''
        数据格式
        data_past： n_batch* (n_seq_p*n_dim)
        data_future： n_batch* (n_seq_f*n_dim)
    '''
    
    n_seq = n_seq_p+n_seq_f
    n_len = np.size(data_in, axis=0)
    
    data_seq_p = []
    for i in range(n_seq_p):
        data_seq_mid = data_in[i:n_len-(n_seq-i),:]
        data_seq_p.append(data_seq_mid)
    data_past = np.concatenate(data_seq_p, axis=1)
    
    data_seq_f = []
    for i in range(n_seq_p,n_seq):
        data_seq_mid = data_in[i:n_len-(n_seq-i),:]
        data_seq_f.append(data_seq_mid)
    data_future = np.concatenate(data_seq_f, axis=1)
    
    
    
    return data_past, data_future




















