# 学校:北京航空航天大学
# 姓名:李文雯
# 日期:2023.4.8

import matplotlib as mpl
import math
import numpy as np
import pandas as pd
import random
from scipy.stats import norm

# 读取数据
def read_data(data_path):
    data_ = pd.read_csv(data_path)
    data = []
    for i in range(data_.size):
        data.append(data_.values[i].tolist()[0])
    return data

def Gaussian(x, mu, sigma):
     return (1 / (np.sqrt(2 * math.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# E_step
def E_step(data, n, theta, mu, sigma):
    data_len = len(data)
    gamma = np.zeros((data_len, n))
    for i in range(data_len):
        # print(i)
        for n_ in range(n):
            # print('*')
            gamma[i][n_] = theta[n_] * Gaussian(data[i], mu[n_], sigma[n_])
        gamma[i] = gamma[i] / sum(gamma[i])
    return gamma


# M_step
def M_step(data, n, gamma):
    mu = [0, 0]
    sigma = [0, 0]
    theta = [0, 0]
    for n in range(n):
        mu[n] = np.dot(data, gamma[:, n]) / np.sum(gamma[:, n])
        sigma[n] = math.sqrt(np.dot((np.array(data) - mu[n]) ** 2, gamma[:, n]) / np.sum(gamma[:, n]))
        theta[n] = np.mean(gamma[:, n])
    return mu, sigma, theta

def my_init(data_path,theta_init=None,sigma_init=None,mu_init=None):
    data_ = read_data(data_path)
    n_class_ = 2
    len_ = len(data_)
    sigma_ = [1 for i in range(n_class_)] if sigma_init is None else sigma_init
    theta_ = [1 / n_class_ for i in range(n_class_)] if theta_init is None else theta_init
    if mu_init is None:
        mu_ = []
        for i in range(n_class_):
            sample_count = int(len(data_) * theta_[i])
            sample = random.sample(data_, sample_count)
            mu_.append(sum(sample) / sample_count)
    else:
        mu_ = mu_init
    print('初始值:mu,sigma,theta')
    print(mu_)
    print(sigma_)
    print(theta_)
    return data_,n_class_,len_,sigma_,theta_,mu_

def my_EM(data_, n_class_, len_, sigma_, theta_, mu_, max_iter = 2000 ,thre = 0.5*1e-10):
    iter = 0
    while iter <= max_iter:
        iter += 1
        gamma = E_step(data = data_, n = n_class_, theta = theta_, mu = mu_, sigma = sigma_)
        mu, sigma, theta = M_step(data = data_, n = n_class_, gamma = gamma)
        if np.max(np.array(mu) - np.array(mu_)) < thre and \
                np.max(np.array(sigma) - np.array(sigma_)) < thre and\
                np.max(np.array(theta) - np.array(theta_)) < thre:
            mu_ = mu
            sigma_ = sigma
            theta_ = theta
            break
        else:
            mu_ = mu
            sigma_ = sigma
            theta_ = theta

    print(iter,'次迭代后满足阈值条件')

    return mu, sigma, theta

def my_assess(mu, sigma, theta):
    print('男生评估:')
    print("占比为{:.4f}，相对偏差为{:.2%}，身高均值为{:.4f}，相对偏差为{:.2%}，身高方差为{:.4f}，相对偏差为{:.2%}".format(
            theta[0], (theta[0] - 0.75) / 0.75, mu[0], (mu[0] - 176) / 176, sigma[0], (sigma[0] - 5) / 5))

    print('女生评估:')
    print("占比为{:.4f}，相对偏差为{:.2%}，身高均值为{:.4f}，相对偏差为{:.2%}，身高方差为{:.4f}，相对偏差为{:.2%}".format(
            theta[1], (theta[1] - 0.25) / 0.25, mu[1], (mu[1] - 164) / 164, sigma[1],(sigma[1] - 3) / 3))

def my_main(data_path,theta_init=None,sigma_init=None,mu_init=None):
    data_,n_class_,len_,sigma_,theta_,mu_ = \
        my_init(data_path, theta_init=theta_init, sigma_init=sigma_init, mu_init=mu_init)
    mu, sigma, theta = \
        my_EM(data_=data_, n_class_=n_class_, len_=len_, sigma_=sigma_, theta_=theta_, mu_=mu_)
    my_assess(mu, sigma, theta)

if __name__ == '__main__':
    print('不指定初始值，采用默认计算的mu,sigma,theta')
    my_main(data_path='./height_data.csv')
    print('指定初始值')
    my_main(data_path='./height_data.csv',theta_init=[0.5,0.5],mu_init=[173,162],sigma_init=[1,1])




