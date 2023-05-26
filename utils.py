# -*- coding: utf-8 -*-
"""
 Created on Fri May 26 2023 16:03:59
 Modified on 2023-5-26 16:03:59
 
 @auther: HJ https://github.com/zhaohaojie1998
"""
# 图像处理 + 绘图输出 + 工具
import cv2
import time
import math
import numpy as np
import matplotlib.pyplot as plt


# matlab计时器
def tic(): 
    '''计时开始'''
    if 'global_tic_time' not in globals():
        global global_tic_time
        global_tic_time = []
    global_tic_time.append(time.time())
    
def toc(name='', *, CN=True, digit=6): 
    '''计时结束'''
    if 'global_tic_time' not in globals() or not global_tic_time: # 未设置全局变量或全局变量为[]
        print('未设置tic' if CN else 'tic not set')  
        return
    name = name+' ' if (name and not CN) else name
    if CN:
        print('%s历时 %f 秒。\n' % (name, round(time.time() - global_tic_time.pop(), digit)))
    else:
        print('%sElapsed time is %f seconds.\n' % (name, round(time.time() - global_tic_time.pop(), digit)))


# 图像处理
class GridMap:
    """从图片中提取栅格地图"""

    def __init__(
        self, 
        img_path: str,
        thresh: int,
        high: int,
        width: int,
    ):
        """提取栅格地图

        Parameters
        ----------
        img_path : str
            原图片路径
        thresh : int
            图片二值化阈值, 大于阈值的部分被置为255, 小于部分被置为0
        high : int
            栅格地图高度
        width : int
            栅格地图宽度
        """
        # 存储路径
        self.__map_path = 'map.png' # 栅格地图路径
        self.__path_path = 'path.png' # 路径规划结果路径

        # 图像处理 #  NOTE cv2 按 HWC 存储图片
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)                     # 读取原图 H,W,C
        thresh, map_img = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY) # 地图二值化
        map_img = cv2.resize(map_img, (width, high))                           # 设置地图尺寸
        cv2.imwrite(self.__map_path, map_img)                                  # 存储二值地图

        # 栅格地图属性
        self.map_array = np.array(map_img)
        """ndarray地图, H*W, 0代表障碍物"""
        self.high = high
        """ndarray地图高度"""
        self.width = width
        """ndarray地图宽度"""


    def show_path(self, path_list, *, save = False):
        """路径规划结果绘制

        Parameters
        ----------
        path_list : list[Node]
            路径节点组成的列表, 要求Node有x,y属性
        save : bool, optional
            是否保存结果图片
        """

        if not path_list:
            print("\n传入空列表, 无法绘图\n")
            return
        if not hasattr(path_list[0], "x") or not hasattr(path_list[0], "y"):
            print("\n路径节点中没有坐标x或坐标y属性, 无法绘图\n")
            return

        x, y = [], []
        for p in path_list:
            x.append(p.x)
            y.append(p.y)
       
        fig, ax = plt.subplots()
        map_ = cv2.imread(self.__map_path)
        map_ = cv2.resize(map_, (self.width, self.high)) 
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # R G B
        #img = img[:, :, ::-1] # R G B
        map_ = map_[::-1] # 画出来的鸡哥是反的, 需要转过来
        ax.imshow(map_, extent=[0, self.width, 0, self.high]) # extent[x_min, x_max, y_min, y_max]
        ax.plot(x, y, c = 'r', label='path', linewidth=2)
        ax.scatter(x[-1], y[-1], c='c', marker='o', label='start', s=40, linewidth=2)
        ax.scatter(x[0], y[0], c='c', marker='x', label='end', s=40, linewidth=2)
        ax.invert_yaxis() # 反转y轴
        ax.legend().set_draggable(True)
        plt.show()
        if save:
            plt.savefig(self.__path_path)



# 节点
class Node:
    pass




# 节点存储队列
class NodeQueue:
    pass











__all__ = ['tic', 'toc', 'GridMap']