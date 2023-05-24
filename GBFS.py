# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 16:45:58 2023

@author: HJ
"""

# 贪婪最佳优先算法(Greedy Best First Search, GBFS)算法 
# A*: F = G + H
# GBFS: F = H
# https://zhuanlan.zhihu.com/p/346666812
from typing import Union
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
from dataclasses import dataclass, field
    

# 地图读取
IMAGE_PATH = 'image.jpg' # 原图路径
MAP_PATH = 'map.png'     # cv加工后的地图存储路径

THRESH = 172             # 图片二值化阈值, 大于阈值的部分被置为255, 小于部分被置为0


# 障碍地图参数设置     #  NOTE cv2 按 HWC 存储图片
HIGHT = 350          # 地图高度
WIDTH = 600          # 地图宽度
START = (290, 270)   # 起点坐标 y轴向下为正
END = (298, 150)     # 终点坐标 y轴向下为正


# 障碍地图提取
image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)                   # 读取原图 H,W,C
THRESH, map_img = cv2.threshold(image, THRESH, 255, cv2.THRESH_BINARY) # 地图二值化
map_img = cv2.resize(map_img, (WIDTH, HIGHT))                          # 设置地图尺寸
cv2.imwrite(MAP_PATH, map_img)                                         # 存储二值地图

















""" ---------------------------- Greedy Best First Search算法 ---------------------------- """
# F = H

Number = Union[int, float]


# 位置坐标数据类
@dataclass # 默认就有 __eq__ 方法
class Position:
    """位置Position对象\n
    两个Position对象相减得到两个Position之间的曼哈顿距离\n
    Position对象加一个坐标得到一个新的Position对象, 即对Position的坐标进行更新\n

    """
    x: int
    y: int

    # 计算两个坐标之间的曼哈顿距离
    def __sub__(self, other) -> int:
        # int = self - other
        if isinstance(other, Position):
            return abs(self.x - other.x) + abs(self.y - other.y)
        elif isinstance(other, (tuple, list)):
            return abs(self.x - other[0]) + abs(self.y - other[1])
        raise ValueError("other必须为坐标或Position")
    def __rsub__(self, other):
        # int = other - self
        return self.__sub__(other)
      
    # 更新坐标 x、y
    def __add__(self, other):
        # new_pos = self + other
        if isinstance(other, Position):
            return Position(self.x + other.x, self.y + other.y)
        elif isinstance(other, (tuple, list)):
            return Position(self.x + other[0], self.y + other[1])
        raise ValueError("other必须为坐标或Position")
    
    # 数据类型检查
    def check(self):
        if not isinstance(self.x, int) or not isinstance(self.y, int):
            raise ValueError("x,y坐标必须为int类型")




# 节点数据类
@dataclass
class NodeList:
    """节点存储列表: OpenList / CloseList"""
    pos_list: list[Position] = field(default_factory=list) # 节点位置坐标存储列表
    cost_list: list[Number] = field(default_factory=list)  # CloseList的G代价, OpenList的F代价
    parent_list: list[Position] = field(default_factory=list) # 父节点坐标
    #NOTE: 可变对象不能作为默认参数, 需要field

    def __bool__(self):
        """判断: while NodeList:"""
        return bool(self.pos_list)
    
    def __contains__(self, item):
        """包含: pos in NodeList"""
        return item in self.pos_list 
        #NOTE: in是值比较, 只看value是否在列表, 不看id是否在列表

    def __len__(self):
        """长度: len(NodeList)"""
        return len(self.pos_list)
    
    def __getitem__(self, idx):
        """索引: NodeList[i]""" #NOTE: idx = 0:2:1 等时 自动转换成 idx=slice(0,2,1)
        return self.pos_list[idx], self.cost_list[idx], self.parent_list[idx]
    
    # def __iter__(self):
    #     """实现__getitem__自动实现__iter__了"""
    #     return zip(self.pos_list, self.cost_list, self.parent_list)
    
    def append(self, pos: Position, cost: Number, parent: Position):
        self.pos_list.append(pos)
        self.cost_list.append(cost)
        self.parent_list.append(parent)

    def pop(self, idx):
        return self.pos_list.pop(idx), self.cost_list.pop(idx), self.parent_list.pop(idx)
    
    def getmin(self):
        """获取cost最小的节点, 并在NodeList中删除"""
        # 用优先队列方便取cost最小的元素, 但不好判断位置坐标pos是否在队列中
        idx = self.cost_list.index(min(self.cost_list))
        pos, cost, parent = self.pop(idx)
        return pos, cost, parent
    



# GBFS算法
class GBFS:
    """GBFS算法"""

    def __init__(
        self,
        start_pos = START,
        end_pos = END,
        map_img = map_img,
        move_step = 3,
        move_direction = 8,
        run = True,
    ):
        """GBFS算法

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
        self.start_pos = Position(*start_pos) # 初始位置
        self.end_pos = Position(*end_pos)     # 结束位置
       
        # Error Check
        self.start_pos.check()
        self.end_pos.check()
        if not self._in_map(self.start_pos) or not self._in_map(self.end_pos):
            raise ValueError(f"x坐标范围0~{self.width-1}, y坐标范围0~{self.height-1}")
        if self._is_collided(self.start_pos):
            raise ValueError(f"起点x坐标或y坐标在障碍物上")
        if self._is_collided(self.end_pos):
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
        self.close_list = NodeList()              # 存储已经走过的位置及其G值 
        self.open_list = NodeList()               # 存储当前位置周围可行的位置及其F值
        self.path_list = []                       # 存储路径(CloseList里的数据无序)


    def _in_map(self, pos: Position):
        """点是否在网格地图中"""
        return (0 <= pos.x < self.width) and (0 <= pos.y < self.high) # 右边不能取等!!!
    

    def _is_collided(self, pos: Position):
        """点是否和障碍物碰撞"""
        return self.map_[pos.y, pos.x] == 0
    

    def _move(self):
        """移动点"""
        @lru_cache(maxsize=3) # 避免参数相同时重复计算
        def _move(move_step:int, move_direction:int):
            move = (
                [0, move_step], # 上
                [0, -move_step],  # 下
                [-move_step, 0],  # 左
                [move_step, 0],  # 右
                [move_step, move_step], # 右上
                [move_step, -move_step],  # 右下
                [-move_step, move_step],  # 左上
                [-move_step, -move_step], # 左下
                )
            return move[0:move_direction] # 坐标增量+代价
        return _move(self.move_step, self.move_direction)


    def _update_open_list(self, curr_pos: Position):
        """open_list添加可行点"""
        for add in self._move():
            # 更新可行位置
            next_pos = curr_pos + add

            # 新位置是否在地图外边
            if not self._in_map(next_pos):
                continue
            # 新位置是否碰到障碍物
            if self._is_collided(next_pos):
                continue
            # 新位置是否已经存在
            if next_pos in self.open_list or next_pos in self.close_list:
                continue # in是值比较, 只看(x,y)是否在列表, 不看id是否在列表

            # 计算所添加的结点的代价
            H = next_pos - self.end_pos # 剩余距离估计
           
            # open-list添加结点
            self.open_list.append(next_pos, H, curr_pos)
            
            # 当剩余距离小时, 走慢一点
            if H < 20:
                self.move_step = 1


    def __call__(self):
        """GBFS路径搜索"""
        assert not self.__reset_flag, "call之前需要reset"

        # 初始化列表
        self.open_list.append(self.start_pos, 0, self.start_pos) # 初始化 OpenList

        # 正向搜索节点(CloseList里的数据无序)
        self._tic
        while self.open_list:
            # 寻找 OpenList 代价最小的点, 并在OpenList中删除
            pos, _, parent = self.open_list.getmin()

            # 更新 OpenList
            self._update_open_list(pos)

            # 更新 CloseList
            self.close_list.append(pos, 0, parent) # G始终为0

            # 结束迭代
            if pos == self.end_pos:
                break
        print("路径节点搜索完成\n")
        self._toc
    
        # 节点组合成路径
        self._tic
        path = self.close_list[-1] # P0, G, P1
        start = self.close_list[0] # P0, G, P1
        while path != start:
            for i, path_curr in enumerate(self.close_list):
                if path_curr[0] == path[2]:             # 如果当前节点是目标节点的父节点
                    path = path_curr                    # 更新目标节点
                    self.path_list.append(path_curr[0]) # 将当前节点加入路径
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
            x.append(p.x)
            y.append(p.y)
       
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
    s = GBFS()

            


            


        












