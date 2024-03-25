
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
def set_plot_basicinf(ylabel,xlabel,fontsize=14):
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize)
    
def generate_vecsek_data(N, L):
    """
    生成二维Vecsek模型的初始状态数据

    :param N: 粒子数
    :param L: 系统尺寸
    :return: 初始状态数据，形状为(N, 4)，第一列为粒子的x坐标，第二列为粒子的y坐标,3,4列复平面坐标
    """

    # 随机生成粒子的初始位置
    pos = np.random.uniform(0, L, size=(N, 2))

    # 随机生成粒子的初始方向（角度）
    theta = np.random.uniform(-np.pi, np.pi, size=N)

    # 将角度转换为向量
    v = np.column_stack((np.cos(theta), np.sin(theta)))

    # 返回初始状态数据
    return np.column_stack((pos, v))


def vecsek_update(data, eta, r_vision,L):
    """
    更新Vecsek模型的运动状态

    :param data: 当前状态数据，形状为(N, 4)，第一列和第二列为粒子的x和y坐标，第三列和第四列为粒子的x和y方向的向量
    :param eta: 系统参数
    :param r_vision: 视野半径
    :return: 更新后的状态数据
    """
#     print('t的位置:',data[0,:2])
#     print('t的速度:',data[0,2:])
    data[:, :2] += data[:, 2:]
    
#     print('t+1的位置:',data[0,:2])
#     de
    # 将超出边界的粒子移动到另一边界
    data[:, :2] %= L
    
    dists = cdist(data[:, :2], data[:, :2])
    # 在视野范围内的粒子对当前粒子产生相互作用
    mask = (dists <= r_vision) & (dists > 0)
    mask = mask.astype(int) + np.eye(data.shape[0])
    
    cos_theta = (mask@data[:, 2:3])[:,0]/np.sum(mask,0)
    sin_theta = (mask@data[:, 3:4])[:,0]/np.sum(mask,0)

    theta_mean = np.arctan2(sin_theta, cos_theta) #把噪声加在角度上，是最原始的，相变会更平滑
#     theta_mean = (mask@theta_mean)[:,0]/np.sum(mask,0)
#     print(theta_mean)
#     de
    # 随机扰动所有粒子的方向向量
    noise = eta * np.random.uniform(-np.pi,np.pi,size=theta_mean.shape)#把噪声加在角度上，是最原始的，相变会更平滑
    #噪声在-pi到pi的均匀分布，eta是强度，就可以设置为0,1之间，和cavada文献的参数保持一致
#     print('noise',noise,theta_mean)
#     de
    theta_mean += noise
#     print(theta_mean)
#     print(theta_mean)
    
    # 更新所有粒子的方向向量
    data[:, 2] = np.cos(theta_mean)
    data[:, 3] = np.sin(theta_mean)


    # 返回更新后的状态数据
    return data,mask

def vecsek_update(data, eta, r_vision,L):
    """
    更新Vecsek模型的运动状态

    :param data: 当前状态数据，形状为(N, 4)，第一列和第二列为粒子的x和y坐标，第三列和第四列为粒子的x和y方向的向量
    :param eta: 系统参数
    :param r_vision: 视野半径
    :return: 更新后的状态数据
    """
#     print('t的位置:',data[0,:2])
#     print('t的速度:',data[0,2:])
    data[:, :2] += data[:, 2:]
    
#     print('t+1的位置:',data[0,:2])
#     de
    # 将超出边界的粒子移动到另一边界
    data[:, :2] %= L
    
    dists = cdist(data[:, :2], data[:, :2])
    # 在视野范围内的粒子对当前粒子产生相互作用
    mask = (dists <= r_vision) & (dists > 0)
    mask = mask.astype(int) + np.eye(data.shape[0])
    
    cos_theta = (mask@data[:, 2:3])[:,0]/np.sum(mask,0)
    sin_theta = (mask@data[:, 3:4])[:,0]/np.sum(mask,0)

    theta_mean = np.arctan2(sin_theta, cos_theta)
#     theta_mean = (mask@theta_mean)[:,0]/np.sum(mask,0)
    
    # 随机扰动所有粒子的方向向量
    noise = eta * np.random.uniform(-1/2,1/2,size=theta_mean.shape)
#     print('noise',noise,theta_mean)
#     de
    theta_mean += noise
#     print(theta_mean)
#     print(theta_mean)
    
    # 更新所有粒子的方向向量
    data[:, 2] = np.cos(theta_mean)
    data[:, 3] = np.sin(theta_mean)


    # 返回更新后的状态数据
    return data,mask