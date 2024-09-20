'''
    VI-CVA
'''

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

from utils.utils import data_normalization, data_standardization
from utils.utils import data_sequence, data_sequence_pf, data_sequence_matrix





def binary_class_accuracy(y_true, y_pred, target_class):
    # 选择目标类别的样本
    relevant = y_true == target_class
    # 计算目标类别中的准确预测
    correct = (y_true[relevant] == y_pred[relevant]).sum()
    return correct / relevant.sum()




n_seq = 4
dim = 23
D = 25

file_path = './data/Training.mat'  # 替换为你的文件路径
mat = scipy.io.loadmat(file_path)

train_num = ['T2','T3']
test_num = ['T1']

data_train_xp = []
data_train_xf = []
for tn in train_num:
    data_normal = mat[tn][::4,:dim]
    data_xp, data_xf  = data_sequence_pf(data_normal, n_seq_p=n_seq, n_seq_f=n_seq)
    
    data_train_xp.append(data_xp)
    data_train_xf.append(data_xf)
    
data_train_xp = np.concatenate(data_train_xp, axis=0)
data_train_xf = np.concatenate(data_train_xf, axis=0)
    
    
data_test_seq = []
data_test_xp = []
data_test_xf = []
for tn in test_num:
    data_test = mat[tn][::4,:dim]
    data_xp, data_xf  = data_sequence_pf(data_test, n_seq_p=n_seq, n_seq_f=n_seq)
    data_test_xp.append(data_xp)
    data_test_xf.append(data_xf)


data_test_xp = np.concatenate(data_test_xp, axis=0)
data_test_xf = np.concatenate(data_test_xf, axis=0)
    
       


mean_p, std_p, data_train_xp = data_standardization(data_train_xp)
mean_f, std_f, data_train_xf = data_standardization(data_train_xf)

data_test_xp = data_standardization(data_test_xp, mean=mean_p, std=std_p)
data_test_xf = data_standardization(data_test_xf, mean=mean_f, std=std_f)




Ml = data_train_xp.shape[1]
Ms1 = data_train_xf.shape[1]
N = data_train_xp.shape[0]

print('Train Data shape:', data_train_xp.shape)
print('Test Data shape:', data_test.shape)


## 先验分布
## P z_t
p_mu_zt = np.zeros([D,1])
p_sig_zt = np.eye(D)

## P\epsilon
p_ep_taum = 0.01

## P\delta
p_de_psim = 0.01

## P\tau
p_tau_j = 1
p_tau_k = 2

## P\psi
p_psi_j = 1
p_psi_k = 2

## P W
p_W_alpha = 0.1*np.eye(D)

## P H
p_H_beta = 0.1*np.eye(D)

## P\alpha
p_alpha_a = 1
p_alpha_B = np.eye(D)
p_alpha_invB = np.linalg.inv(p_alpha_B)

## P\beta
p_beta_a = 1
p_beta_B = np.eye(D)
p_beta_invB = np.linalg.inv(p_beta_B)


## 变分分布 初始化
## Q z_t
q_mu_z = np.random.normal(0,0.01,size=[N,Ml,1])
q_sig_z = np.array([np.eye(Ml) for _ in range(N)]).astype(np.float32)


## Q w
q_mu_w = np.random.normal(0,0.01,size=[Ml,D,1])
q_sig_w = np.array([np.eye(D) for _ in range(Ml)]).astype(np.float32)

## Q h
q_mu_h = np.random.normal(0,0.01,size=[Ms1,D,1])
q_sig_h = np.array([np.eye(D) for _ in range(Ms1)]).astype(np.float32)


## Q \tau
q_lambda_tau = np.array([1 for _ in range(Ml)])
q_nu_tau = np.array([2 for _ in range(Ml)])

## Q \psi
q_lambda_psi = np.array([1 for _ in range(Ms1)])
q_nu_psi = np.array([2 for _ in range(Ms1)])


## Q \alpha
q_nu_alpha = np.array([1 for _ in range(Ml)])
q_V_alpha = np.array([np.eye(D) for _ in range(Ml)]).astype(np.float32)

## Q \beta
q_nu_beta = np.array([1 for _ in range(Ms1)])
q_V_beta = np.array([np.eye(D) for _ in range(Ms1)]).astype(np.float32)


print('=== parameter initialization ===')



norm_mu = [[np.sum(q_mu_h**2), np.sum(q_mu_w**2)]]

for _ in range(10):
    
    ## update z_t
    
    E_tau = q_lambda_tau/q_nu_tau
    E_w2 = np.linalg.inv(q_sig_w) + np.array([q_mu_w[i,:,:] @ q_mu_w[i,:,:].T for i in range(Ml)])
    
    
    E_psi = q_lambda_psi/q_nu_psi
    E_h2 = np.linalg.inv(q_sig_h) + np.array([q_mu_h[i,:,:] @ q_mu_h[i,:,:].T for i in range(Ms1)])
    
    
    
    q_Lambda_zt = np.sum(E_tau.reshape(-1,1,1)*E_w2,axis=0)+np.sum(E_psi.reshape(-1,1,1)*E_h2,axis=0)+np.eye(D)
    q_Lambda_z = np.array([q_Lambda_zt for _ in range(N)]).astype(np.float32)
    
    
    
    dp = np.expand_dims(data_train_xp.T, axis=[2,3])
    df = np.expand_dims(data_train_xf.T, axis=[2,3])
    
    p1 = E_tau.reshape(-1,1,1)*q_mu_w
    p1 = np.expand_dims(p1, axis=1)
    
    p2 = E_psi.reshape(-1,1,1)*q_mu_h
    p2 = np.expand_dims(p2, axis=1)
    
    pp1 = np.sum( np.array([dp[i,:,:,0]@p1[i,:,:,0] for i in range(Ml)]), axis=0)
    pp2 = np.sum( np.array([df[i,:,:,0]@p2[i,:,:,0] for i in range(Ms1)]), axis=0)
    
    q_invLambda_zt = np.linalg.inv(q_Lambda_zt)
    q_mu_z = np.array([ q_invLambda_zt @ (pp1[i,:].reshape(-1,1)+pp2[i,:].reshape(-1,1)) for i in range(N)])
    
    
    
    ## update H,W
    
    E_tau = q_lambda_tau/q_nu_tau
    E_psi = q_lambda_psi/q_nu_psi
    
    E_z2 = np.array([q_invLambda_zt + q_mu_z[i,:,:] @ q_mu_z[i,:,:].T for i in range(N)])
    
    
    E_alpha = q_nu_alpha.reshape(-1,1,1)*q_V_alpha
    q_sig_w = np.sum(np.array([E_tau[i]*E_z2 for i in range(Ml)]),axis=1)+E_alpha
    q_mu_w = np.array([np.linalg.inv(q_sig_w[i,:,:])@(E_tau[i]*np.sum(q_mu_z*dp[i,:,:,:],axis=0)) for i in range(Ml)])
    
    
    E_beta = q_nu_beta.reshape(-1,1,1)*q_V_beta
    q_sig_h = np.sum(np.array([E_psi[i]*E_z2 for i in range(Ms1)]),axis=1)+E_beta
    q_mu_h = np.array([np.linalg.inv(q_sig_h[i,:,:])@(E_psi[i]*np.sum(q_mu_z*df[i,:,:,:],axis=0)) for i in range(Ms1)])
    
    
    ## update \tau,\psi
    
    E_2z = np.expand_dims(np.array([np.trace(q_invLambda_zt)+q_mu_z[i,:,:].T@ q_mu_z[i,:,:] for i in range(N)]),axis=0)
    
    E_2w = np.expand_dims(np.array([np.trace(np.linalg.inv(q_sig_w[i,:,:]))+q_mu_w[i,:,:].T@ q_mu_w[i,:,:] for i in range(Ml)]),axis=1)
    E_wz = np.expand_dims(np.sum(np.array([q_mu_z[:,:,:]*q_mu_w[i,:,:] for i in range(Ml)]),axis=2),axis=3)
    
    
    pp_tau = dp**2-2*dp*E_wz+E_2w@E_2z
    
    q_lambda_tau = p_tau_j+1/2*N
    q_lambda_tau = np.array([q_lambda_tau for _ in range(Ml)])
    
    q_nu_tau = p_tau_k+1/2*np.sum(pp_tau, axis=1)
    q_nu_tau = np.squeeze(q_nu_tau)
    
    
    
    
    E_2h = np.expand_dims(np.array([np.trace(np.linalg.inv(q_sig_h[i,:,:]))+q_mu_h[i,:,:].T@ q_mu_h[i,:,:]for i in range(Ms1)]),axis=1)
    E_hz = np.expand_dims(np.sum(np.array([q_mu_z[:,:,:]*q_mu_h[i,:,:] for i in range(Ms1)]),axis=2),axis=3)
    
    pp_psi = df**2-2*df*E_hz+E_2h@E_2z
    
    q_lambda_psi = p_psi_j+1/2*N
    q_lambda_psi = np.array([q_lambda_psi for _ in range(Ms1)])
    
    q_nu_psi = p_psi_k+1/2*np.sum(pp_psi, axis=1)
    q_nu_psi = np.squeeze(q_nu_psi)
    
        
    
    ## update \alpha,\beta
    
    
    q_nu_alpha = np.array([p_alpha_a+1 for _ in range(Ml)])
    q_V_alpha = np.array([np.linalg.inv(p_alpha_invB+E_w2[i,:,:]) for i in range(Ml)])
    
    
    q_nu_beta = np.array([p_beta_a+1 for _ in range(Ms1)])
    q_V_beta = np.array([np.linalg.inv(p_beta_invB+E_h2[i,:,:]) for i in range(Ms1)])
    
    
    
    
    
    
    
    norm_mu.append([np.sum(q_mu_h**2), np.sum(q_mu_w**2)])


norm_mu = np.array(norm_mu)


plt.figure()
plt.plot(norm_mu[:,0])
plt.plot(norm_mu[:,1])


print('=========== model training completed =======')



model_tau = np.diag(E_tau)
model_psi = np.diag(E_psi)
model_mu_h = np.squeeze(q_mu_h)
model_mu_w = np.squeeze(q_mu_w)

model = [q_Lambda_zt, model_mu_h, model_mu_w, model_tau, model_psi]



data_all_xp = np.concatenate((data_train_xp, data_test_xp), axis=0)
data_all_xf = np.concatenate((data_train_xf, data_test_xf), axis=0)

z_p = np.dot(model_mu_w.T,np.dot(model_tau, data_all_xp.T))
z_f = np.dot(model_mu_h.T,np.dot(model_psi, data_all_xf.T))

z = q_Lambda_zt@(z_p+z_f)
data_t2 = np.sum(z**2, axis=0)


delta = data_all_xf.T-np.dot(model_mu_h, z)

data_spe = np.sum(np.diag(model_psi).reshape(-1,1)*delta**2,axis=0)





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


y_true = np.zeros(data_test_xp.shape[0])
y_pred = data_spe[data_train_xp.shape[0]:]>limit_spe
acc0 = binary_class_accuracy(y_true, y_pred, target_class=0)

y_true = np.zeros(data_test_xp.shape[0])
y_pred = data_t2[data_train_xp.shape[0]:]>limit_T2
acc1 = binary_class_accuracy(y_true, y_pred, target_class=0)

print(f'limits--- spe:{limit_spe}----T2:{limit_T2}')
print(f'======= Training SPE FAR:{1-acc0:.4f} ========')
print(f'======= Training T2 FAR:{1-acc1:.4f} ========')




def preprocess(x_in, mean_out, std_out):
    
    mean_p = mean_out[0]
    mean_f = mean_out[1]
    std_p = std_out[0]
    std_f = std_out[1]
        
    x_in_p, x_in_f = data_sequence_pf(x_in[:,:dim], n_seq_p=n_seq, n_seq_f=n_seq)
    x_in_p = data_standardization(x_in_p, mean=mean_p, std=std_p)
    x_in_f = data_standardization(x_in_f, mean=mean_f, std=std_f)
    x_in = [x_in_p, x_in_f]
    return x_in





def f1_index(x_in, model):
    
    
    q_Lambda_zt = model[0]
    model_mu_h = model[1]
    model_mu_w = model[2]
    model_tau = model[3]
    model_psi = model[4]
    x_in_p = x_in[0]
    x_in_f = x_in[1]

    z_p = np.dot(model_mu_w.T,np.dot(model_tau, x_in_p.T))
    z_f = np.dot(model_mu_h.T,np.dot(model_psi, x_in_f.T))
    
    z = q_Lambda_zt@(z_p+z_f)
    data_t2 = np.sum(z**2, axis=0)

    return data_t2



def f2_index(x_in, model):
    
    q_Lambda_zt = model[0]
    model_mu_h = model[1]
    model_mu_w = model[2]
    model_tau = model[3]
    model_psi = model[4]
    x_in_p = x_in[0]
    x_in_f = x_in[1]

    z_p = np.dot(model_mu_w.T,np.dot(model_tau, x_in_p.T))
    z_f = np.dot(model_mu_h.T,np.dot(model_psi, x_in_f.T))
    
    z = q_Lambda_zt@(z_p+z_f)
    
    delta = data_all_xf.T-np.dot(model_mu_h, z)

    data_spe = np.sum(np.diag(model_psi).reshape(-1,1)*delta**2,axis=0)

    
    return data_spe




method_dict = {
    'spe': [f1_index, limit_spe],
    't2':[f2_index, limit_T2]    
    }
method = 'spe'




from data import data_eval
res = data_eval.mff_test_process_ds(preprocess, [mean_p, mean_f], [std_p,std_f], model,\
                                 method_dict[method][0], limit=method_dict[method][1], ds_rate=4, bias=2*n_seq-1)

    
    
print('=========== AVG ============')
set_cho = ['Set1_1','Set1_2','Set1_3',\
    'Set2_1','Set2_2','Set2_3',\
        'Set3_1','Set3_2','Set3_3',\
            'Set4_1','Set4_2','Set4_3',\
                'Set5_1','Set5_2',\
                    'Set6_1','Set6_2']
FDR, FAR = data_eval.mff_cho_performance_avg(res, set_cho)













