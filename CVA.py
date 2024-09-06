'''
    CVA
'''

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

from utils.utils import data_normalization, data_standardization
from utils.utils import data_sequence, data_sequence_pf, data_sequence_matrix





def binary_class_accuracy(y_true, y_pred, target_class):
    relevant = y_true == target_class
    correct = (y_true[relevant] == y_pred[relevant]).sum()
    return correct / relevant.sum()




n_seq = 15
dim = 23
nc = 25


file_path = './data/Training.mat'  # 替换为你的文件路径
mat = scipy.io.loadmat(file_path)

train_num = ['T2','T3']
test_num = ['T1']

data_train_xp = []
data_train_xf = []
for tn in train_num:
    data_normal = mat[tn][:,:dim]
    data_xp, data_xf  = data_sequence_pf(data_normal, n_seq_p=n_seq, n_seq_f=n_seq)
    
    data_train_xp.append(data_xp)
    data_train_xf.append(data_xf)
    
data_train_xp = np.concatenate(data_train_xp, axis=0)
data_train_xf = np.concatenate(data_train_xf, axis=0)
    
    


data_test_seq = []
for tn in test_num:
    data_test = mat[tn][:,:dim]
    data_test_seq.append(data_sequence_matrix(data_test, n_seq=n_seq))
        
data_test_seq = np.concatenate(data_test_seq, axis=0)


mean_p, std_p, data_train_xp = data_standardization(data_train_xp)
mean_f, std_f, data_train_xf = data_standardization(data_train_xf)

data_test = data_standardization(data_test_seq, mean=mean_p, std=std_p)


print('Train Data shape:', data_train_xp.shape)
print('Test Data shape:', data_test.shape)


sigma_pp = np.dot(data_train_xp.T,data_train_xp)/(data_train_xp.shape[0]-1)
sigma_ff = np.dot(data_train_xf.T,data_train_xf)/(data_train_xf.shape[0]-1)
Hfp = np.dot(data_train_xf.T,data_train_xp)/(data_train_xf.shape[0]-1)

Rp = np.linalg.cholesky(sigma_pp).T
Rf = np.linalg.cholesky(sigma_ff).T


# lhs = np.tril(np.linalg.inv(Rf.T),k=0)
# rhs = np.triu(np.linalg.inv(Rp),k=0)

lhs = np.linalg.inv(Rf.T)
rhs = np.linalg.inv(Rp)

H = lhs @ Hfp @ rhs
U, S, V = np.linalg.svd(H)

V1 = U[:,:nc]
J = np.dot(V1.T,np.linalg.inv(Rp.T))
L = np.dot((np.eye(S.shape[0])-np.dot(V1, V1.T)), np.linalg.inv(Rp.T))

print('====== CVA Established =======')



data_all = np.concatenate((data_train_xp, data_test), axis=0)

z = np.dot(J, data_all.T)
e = np.dot(L, data_all.T)

data_spe = np.sum(e**2, axis=0)
data_t2 = np.sum(z**2, axis=0)


from statsmodels.nonparametric.api import KDEUnivariate

kde_spe=KDEUnivariate(data_spe[:data_train_xp.shape[0]])
kde_spe.fit(kernel='gau',bw=0.2, gridsize=100)#

index=np.argmin(abs(kde_spe.cdf-0.99))
limit_spe=kde_spe.support[index]


kde_T2=KDEUnivariate(data_t2[:data_train_xp.shape[0]])
kde_T2.fit(kernel='gau',bw=0.2, gridsize=100)#
index = np.argmin(abs(kde_T2.cdf-0.99))
limit_T2=kde_T2.support[index]


plt.figure()
plt.subplot(2,1,1)
plt.plot(data_spe[:])
plt.axvline(x=data_train_xp.shape[0])
plt.axhline(y=limit_spe)

plt.subplot(2,1,2)
plt.plot(data_t2[:])
plt.axvline(x=data_train_xp.shape[0])
plt.axhline(y=limit_T2)



y_true = np.zeros(data_test.shape[0])
y_pred = data_spe[data_train_xp.shape[0]:]>limit_spe
acc0 = binary_class_accuracy(y_true, y_pred, target_class=0)

y_true = np.zeros(data_test.shape[0])
y_pred = data_t2[data_train_xp.shape[0]:]>limit_T2
acc1 = binary_class_accuracy(y_true, y_pred, target_class=0)

print(f'limits--- spe:{limit_spe}----T2:{limit_T2}')
print(f'======= Training SPE FAR:{1-acc0:.4f} ========')
print(f'======= Training T2 FAR:{1-acc1:.4f} ========')





model = [L, J]




def preprocess(x_in, mean_out, std_out):
    
    x_in = data_sequence_matrix(x_in[:,:dim], n_seq=n_seq)
    x_in = data_standardization(x_in, mean=mean_out, std=std_out)
    
    return x_in    




def f1_index(x_in, model):
    
    e = np.dot(model[0], x_in.T)
    data_spe = np.sum(e**2, axis=0)
    return data_spe



def f2_index(x_in, model):
    
    z = np.dot(model[1], x_in.T)
    data_t2 = np.sum(z**2, axis=0)
    return data_t2




method_dict = {
    'spe': [f1_index, limit_spe],
    't2':[f2_index, limit_T2]    
    }
method = 'spe'




from data import data_eval
res = data_eval.mff_test_process(preprocess, mean_p, std_p, model,\
                                 method_dict[method][0], limit=method_dict[method][1], bias=n_seq-1)

    
    
print('=========== AVG ============')
set_cho = ['Set1_1','Set1_2','Set1_3','Set3_1','Set3_3','Set4_2','Set6_1','Set6_2']
FDR, FAR = data_eval.mff_cho_performance_avg(res, set_cho)


