# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 16:45:58 2023

@author: HJ
"""

# Dijkstra算法 
from typing import Union
import cv2
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
from dataclasses import dataclass, field
from queue import PriorityQueue
Number = Union[int, float]

    

# 地图读取
IMAGE_PATH = 'image.jpg' # 原图路径
MAP_PATH = 'map.png'     # cv加工后的地图存储路径

THRESH = 172             # 图片二值化阈值, 大于阈值的部分被置为255, 小于部分被置为0


# 障碍地图参数设置     #  NOTE cv2 按 HWC 存储图片
HIGHT = 70          # 地图高度
WIDTH = 120          # 地图宽度
START = (58, 54)   # 起点坐标 y轴向下为正
END = (59, 26)     # 终点坐标 y轴向下为正


# 障碍地图提取
image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)                   # 读取原图 H,W,C
THRESH, map_img = cv2.threshold(image, THRESH, 255, cv2.THRESH_BINARY) # 地图二值化
map_img = cv2.resize(map_img, (WIDTH, HIGHT))                          # 设置地图尺寸
cv2.imwrite(MAP_PATH, map_img)                                         # 存储二值地图









""" ---------------------------- Dijkstra算法 ---------------------------- """


# 节点
@dataclass(eq=False)
class Node:
    """节点数据类

    一些骚操作
    ---------
    0.索引节点坐标:
        x, y = Node
        x, y = Node[0], Node[1]
        x, y = Node.node[0], Node.node[1]
    1.求节点和另一个节点/坐标之间的曼哈顿距离:
        distance = Node - (x, y)
        distance = Node - Node_1
    2.移动节点:
        NewNode = Node + (dx, dy)
    3.判断两个节点坐标是否相同 (不比较父节点坐标):
        Node == (x, y)
        Node == Node_1
    4.判断坐标/节点是否在Node组成的列表中:
        Node in [Node, ...]
        (x, y) in [Node, ...]
    4.两个节点代价比较 (要求两边必须都是Node):
        Node_1 <= Node_2
        Node_1 < Node_2

    常规操作
    ---------
    0.判断节点是否在地图中:
        Node.in_map(your_map_ndarray)
    1.判断节点是否碰到障碍 (要求地图中0表示障碍物):
        Node.is_collided(your_map_ndarray)

    """

    # 节点
    node: tuple[int, int]               # 当前节点的 x, y 坐标

    # 节点其它信息
    cost: Number = 0                    # 父节点到当前节点的代价信息
    parent_node: tuple[int, int] = None # 父节点 x, y 坐标信息

    # 像list/tuple一样索引x,y坐标
    def __getitem__(self, idx): # Node[idx] = Node.node[idx]
        return self.node[idx]

    # 计算两个节点的曼哈顿距离
    def __sub__(self, other) -> int: # distance = self - other
        return abs(self[0] - other[0]) + abs(self[1] - other[1])
        
    # 移动节点坐标 x,y, 生成新节点
    def __add__(self, other) -> "Node": # new_node = self + other
        cost = self.cost + math.sqrt(other[0]**2 + other[1]**2) # 欧式距离
        #cost = self.cost + (self - other) # 曼哈顿距离
        return Node(node=(self[0]+other[0], self[1]+other[1]), parent_node=self[:], cost=cost)
    
    # 两个节点坐标是否相等 (避免重复添加节点, 所以不比较父节点)
    def __eq__(self, other): # Node in queue 调用 __eq__ 方法
        if isinstance(other, (Node, tuple, list)):
            return other[0] == self[0] and other[1] == self[1] 
        return False
    
    # 两个节点代价比较 (自动实现 >和 >= 了)
    def __lt__(self, other): # 优先queue中弹出代价最小的数据 调用 __lt__ 和 __le__ 方法
        return self.cost < other.cost
    def __le__(self, other):
        return self.cost <= other.cost
    
    # 当前节点是否在网格地图中
    def in_map(self, map_: np.ndarray): # map_ H*W
        return (0 <= self[0] < map_.shape[1]) and (0 <= self[1] < map_.shape[0]) # 右边不能取等!!!
    
    # 当前节点是否碰到障碍物 
    def is_collided(self, map_: np.ndarray): # map_ H*W
        return map_[self[1], self[0]] == 0 # map(y, x)


    



# Dijkstra算法
class Dijkstra:
    """Dijkstra算法"""

    def __init__(
        self,
        start_pos = START,
        end_pos = END,
        map_img = map_img,
        move_step = 1,
        move_direction = 8,
        run = True,
    ):
        """Dijkstra算法

        Parameters
        ----------
        start_pos : tuple/list
            起点坐标
        end_pos : tuple/list
            终点坐标
        map_img : Mat
            二值化地图, 0表示障碍物, 255表示空白
        move_step : int
            移动步数, 默认3
        move_direction : int (8 or 4)
            移动方向, 默认8个方向
        run : bool
            是否在算法实例化之后运行算法, 否则需要手动调用
        """
        self.__tic_list = [] # 计时器

        # 网格化地图
        self.map_ = np.array(map_img) # H * W

        self.width = self.map_.shape[1]
        self.high = self.map_.shape[0]

        # 起点终点
        self.start_pos = start_pos # 初始位置
        self.end_pos = end_pos     # 结束位置

        self.start_node = Node(start_pos, 0, start_pos) # 初始节点
        
        # Error Check
        self.end_node = Node(end_pos)
        if not isinstance(sum(start_pos), int) or not isinstance(sum(end_pos), int):
            raise ValueError("x坐标和y坐标必须为int")
        if not self.start_node.in_map(self.map_) or not self.end_node.in_map(self.map_):
            raise ValueError(f"x坐标范围0~{self.width-1}, y坐标范围0~{self.height-1}")
        if self.start_node.is_collided(self.map_):
            raise ValueError(f"起点x坐标或y坐标在障碍物上")
        if self.end_node.is_collided(self.map_):
            raise ValueError(f"终点x坐标或y坐标在障碍物上")
       
        # 算法初始化
        self.reset(move_step, move_direction)
        
        # 算法运行
        if run:
            self.__call__()


    def reset(self, move_step=3, move_direction=8):
        """重置算法"""
        self.__reset_flag = False
        self.move_step = move_step                # 移动步长(搜索后期会减小)
        self.move_direction = move_direction      # 移动方向 8 个
        self.close_list = []                      # 存储已经走过的位置及其G值 
        self.open_list = PriorityQueue()          # 存储当前位置周围可行的位置及其F值
        self.path_list = []                       # 存储路径(CloseList里的数据无序)

    
    @staticmethod
    def _move(move_step:int):
        """移动点"""
        move = (
            [0, move_step], # 上
            [0, -move_step], # 下
            [-move_step, 0], # 左
            [move_step, 0], # 右
            [move_step, move_step], # 右上
            [move_step, -move_step], # 右下
            [-move_step, move_step], # 左上
            [-move_step, -move_step], # 左下
            )
        return move # 坐标增量
        

    def _update_open_list(self, curr_node: Node):
        """open_list添加可行点"""
        
        for add in self._move(self.move_step)[:self.move_direction]:
            # 更新可行位置
            next_node = curr_node + add

            # 新位置是否在地图外边
            if not next_node.in_map(self.map_):
                continue
            # 新位置是否碰到障碍物
            if next_node.is_collided(self.map_):
                continue
            # 新位置是否已经存在
            if next_node in self.open_list.queue or next_node in self.close_list:
                continue # in是值比较, 只看(x,y)是否在列表, 不看id是否在列表

            # 计算所添加的结点的代价
            # G = next_node.cost                  # 已走的代价
          
            # open-list添加结点
            self.open_list.put(next_node)
            
            # 当剩余距离小时, 走慢一点
            if next_node - self.end_pos < 20:
                self.move_step = 1


    def __call__(self):
        """Dijkstra路径搜索"""
        assert not self.__reset_flag, "call之前需要reset"
        print("搜索中\n")

        # 初始化优先队列
        self.open_list.put(self.start_node) # 初始化 OpenList
        
        # 正向搜索节点(CloseList里的数据无序)
        self._tic
        while not self.open_list.empty():
            # 吐出 F 最小的节点
            curr_node = self.open_list.get() # OpenList里的 cost 为 F = G
            
            # 更新 OpenList
            self._update_open_list(curr_node)

            # 更新 CloseList
            self.close_list.append(curr_node)

            # 结束迭代
            if curr_node == self.end_pos:
                break
        print("路径节点搜索完成\n")
        self._toc
    
        # 节点组合成路径(将CloseList里的数据整合)
        self._tic
        next_ = self.close_list[-1] # P0, G, P1
        start_ = self.close_list[0] # P0, G, P1
        while next_ != start_:
            for i, curr_ in enumerate(self.close_list):
                if curr_.node == next_.parent_node:     # 如果当前节点是目标节点的父节点
                    next_ = curr_                       # 更新目标节点
                    self.path_list.append(curr_.node)   # 将当前节点坐标加入路径
                    self.close_list.pop(i)              # 弹出当前节点, 避免重复遍历
        print("路径节点整合完成\n")
        self._toc
       
        # 绘图输出
        self.show()

        # 需要重置
        self.__reset_flag = True
        

    def show(self):
        """绘图输出"""
        # 没路径先搜索路径
        if not self.path_list:
            self.__call__() 
            return

        # 绘图输出
        x, y = [], []
        for p in self.path_list:
            x.append(p[0])
            y.append(p[1])
       
        fig, ax = plt.subplots()
        map_ = cv2.imread(MAP_PATH)
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


    @property
    def _tic(self):
        """matlab计时器tic"""
        self.__tic_list.append(time.time())


    @property
    def _toc(self):
        """matlab计时器toc"""
        if self.__tic_list:
            t = time.time() - self.__tic_list.pop()
            print(f"历时: {t}s\n")







# debug
if __name__ == '__main__':
    s = Dijkstra()

            


            


        












