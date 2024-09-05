import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

def data_generator(num_sample, std=0.3, fault=None,f_time = 200):
    '''
        生成数值例子数据
    '''
    x1 = [0]
    x2 = [0]
    x3 = [0]
    x4 = [0]
    x5 = [0]

    if fault == None:
        for i in range(num_sample):
            x1.append(np.sin(x1[i]**2)-0.4*np.sqrt((0.1*x5[i]-0.3*x4[i])**2+0.2)+npr.normal(0,std))
            x2.append(np.exp(-x1[i])-0.4*x2[i]-0.2*x4[i]+npr.normal(0,std))
            x3.append(x1[i]*x2[i]+np.cos(x3[i])-0.1*x5[i]+npr.normal(0,std))
            x4.append(np.sqrt(x2[i]**2+x3[i]**2)-0.4*x4[i]+npr.normal(0,std))
            x5.append(x3[i]*x4[i]+np.sin(x5[i])-1.2*x1[i]+npr.normal(0,std))
            data = np.array([x1,x2,x3,x4,x5])
            # data = np.array([x1,x2,x3,x4])
        return data

    if fault:
        for i in range(num_sample):
            if i <= f_time:
                x1.append(np.sin(x1[i]**2)-0.4*np.sqrt((0.1*x5[i]-0.3*x4[i])**2+0.2)+npr.normal(0,std))
                x2.append(np.exp(-x1[i])-0.4*x2[i]-0.2*x4[i]+npr.normal(0,std))
                x3.append(x1[i]*x2[i]+np.cos(x3[i])-0.1*x5[i]+npr.normal(0,std))
                x4.append(np.sqrt(x2[i]**2+x3[i]**2)-0.4*x4[i]+npr.normal(0,std))
                x5.append(x3[i]*x4[i]+np.sin(x5[i])-1.2*x1[i]+npr.normal(0,std))
            else:
                x1.append(np.sin(x1[i]**2)-0.4*np.sqrt((0.1*x5[i]-0.3*x4[i])**2+0.2)+npr.normal(0,std)+fault[0])
                x2.append(np.exp(-x1[i])-0.4*x2[i]-0.2*x4[i]+npr.normal(0,std)+fault[1])
                x3.append(x1[i]*x2[i]+np.cos(x3[i])-0.1*x5[i]+npr.normal(0,std)+fault[2])
                x4.append(np.sqrt(x2[i]**2+x3[i]**2)-0.4*x4[i]+npr.normal(0,std)+fault[3])
                x5.append(x3[i]*x4[i]+np.sin(x5[i])-1.2*x1[i]+npr.normal(0,std)+fault[4])
            data = np.array([x1,x2,x3,x4,x5])
            # data = np.array([x1,x2,x3,x4])1111
        return data

def data_linear_generator(num_sample, std=0.3, fault=None, f_time=200):
    '''
        生成数值例子数据
    '''
    x1 = [0]
    x2 = [0]
    x3 = [0]
    x4 = [0]
    x5 = [0]

    if fault == None:
        for i in range(num_sample):
            x1.append(0.62*x1[i]+npr.normal(0,std))
            x2.append(0.68*x2[i]+npr.normal(0,std))
            x3.append(x1[i]+x2[i]+0.82*x3[i]+npr.normal(0,std))
            x4.append(x2[i]+x3[i]-0.72*x4[i]+npr.normal(0,std))
            data = np.array([x1,x2,x3,x4])
        return data

    if fault:
        for i in range(num_sample):
            if i <= f_time:
                x1.append(0.62*x1[i]+npr.normal(0,std))
                x2.append(0.98*x2[i]+npr.normal(0,std))
                x3.append(x1[i]+x2[i]+0.82*x3[i]+npr.normal(0,std))
                x4.append(x2[i]+x3[i]-0.72*x4[i]+npr.normal(0,std))
            else:
                x1.append(0.62*x1[i]+npr.normal(0,std)+fault[0])
                x2.append(0.68*x2[i]+npr.normal(0,std)+fault[1])
                x3.append(x1[i]+x2[i]+0.82*x3[i]+npr.normal(0,std)+fault[2])
                x4.append(x2[i]+x3[i]-0.72*x4[i]+npr.normal(0,std)+fault[3])
            data = np.array([x1,x2,x3,x4])
        return data

def data_high_order_generator(num_sample, std=0.3, fault=None, f_time=200):
    '''
        生成动态阶数高点(4)的数值例子数据
    '''
    x1 = [0,0,0,0]
    x2 = [0,0,0,0]
    x3 = [0,0,0,0]
    x4 = [0,0,0,0]

    if fault == None:
        for i in range(3,num_sample):
            x1.append(np.sin(x1[i]**2)-0.7*np.exp(-(x1[i-1]**2+x1[i-2]**2))-0.4*np.sqrt((0.1*x2[i-1]-0.3*x4[i-2])**2+0.2)+npr.normal(0,std))
            x2.append(np.exp(-x1[i])-0.4*x2[i]-0.4*np.sin(x2[i-1]-0.4*x3[i-2])-0.2*x4[i-2]+npr.normal(0,std))
            x3.append(x1[i-1]*x2[i-2]+np.cos(x3[i])-0.2*x3[i-1]-np.exp(-x4[i-1])+npr.normal(0,std))
            x4.append(-0.4*x4[i-3]*np.sin(-x4[i-2])+np.sqrt(x2[i]**2+x3[i-2]**2)-0.4*x4[i]*x1[i-1]+npr.normal(0,std))
        
        data = np.array([x1,x2,x3,x4])
        return data

    if fault:
        for i in range(3,num_sample):
            if i <= f_time:
                x1.append(np.sin(x1[i]**2)-0.7*np.exp(-(x1[i-1]**2+x1[i-2]**2))-0.4*np.sqrt((0.1*x2[i-1]-0.3*x4[i-2])**2+0.2)+npr.normal(0,std))
                x2.append(np.exp(-x1[i])-0.4*x2[i]-0.4*np.sin(x2[i-1]-0.4*x3[i-2])-0.2*x4[i-2]+npr.normal(0,std))
                x3.append(x1[i-1]*x2[i-2]+np.cos(x3[i])-0.2*x3[i-1]-np.exp(-x4[i-1])+npr.normal(0,std))
                x4.append(-0.4*x4[i-3]*np.sin(-x4[i-2])+np.sqrt(x2[i]**2+x3[i-2]**2)-0.4*x4[i]*x1[i-1]+npr.normal(0,std))
    
            else:
                x1.append(np.sin(x1[i]**2)-0.7*np.exp(-(x1[i-1]**2+x1[i-2]**2))-0.4*np.sqrt((0.1*x2[i-1]-0.3*x4[i-2])**2+0.2)+npr.normal(0,std)+fault[0])
                x2.append(np.exp(-x1[i])-0.4*x2[i]-0.4*np.sin(x2[i-1]-0.4*x3[i-2])-0.2*x4[i-2]+npr.normal(0,std)+fault[1])
                x3.append(x1[i-1]*x2[i-2]+np.cos(x3[i])-0.2*x3[i-1]-np.exp(-x4[i-1])+npr.normal(0,std)+fault[2])
                x4.append(-0.4*x4[i-3]*np.sin(-x4[i-2])+np.sqrt(x2[i]**2+x3[i-2]**2)-0.4*x4[i]*x1[i-1]+npr.normal(0,std)+fault[3])

            data = np.array([x1,x2,x3,x4])
        return data

# if __name__ == '__main__':
#    data = data_high_order_generator(1000,0.3,fault=[0,0,0,0])
# #    print(data.shape)
#    plt.figure()
#    plt.plot(data[0:4,:].T)
#    plt.show()

    
    
    