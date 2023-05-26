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
import math
import numpy as np
from functools import lru_cache
from dataclasses import dataclass, field
from utils import tic, toc, GridMap
    

# 地图读取
IMAGE_PATH = 'image.jpg' # 原图路径
THRESH = 172             # 图片二值化阈值, 大于阈值的部分被置为255, 小于部分被置为0
HIGHT = 350              # 地图高度
WIDTH = 600              # 地图宽度

MAP = GridMap(IMAGE_PATH, THRESH, HIGHT, WIDTH) # 栅格地图对象

# 起点终点     
START = (290, 270)   # 起点坐标 y轴向下为正
END = (298, 150)     # 终点坐标 y轴向下为正







""" ---------------------------- Greedy Best First Search算法 ---------------------------- """
# F = H

Number = Union[int, float]


@dataclass(eq=False)
class Node:
    """节点"""

    x: int
    y: int
    cost: Number = 0
    parent: "Node" = None

    def __sub__(self, other) -> int:
        """计算节点与坐标的曼哈顿距离"""
        if isinstance(other, Node):
            return abs(self.x - other.x) + abs(self.y - other.y)
        elif isinstance(other, (tuple, list)):
            return abs(self.x - other[0]) + abs(self.y - other[1])
        raise ValueError("other必须为坐标或Node")
    
    def __add__(self, other: Union[tuple, list]) -> "Node":
        """生成新节点"""
        x = self.x + other[0]
        y = self.y + other[1]
        cost = self.cost + math.sqrt(other[0]**2 + other[1]**2) # 欧式距离
        return Node(x, y, cost, self)
        
    def __eq__(self, other):
        """坐标x,y比较 -> node in close_list"""
        if isinstance(other, Node):
            return self.x == other.x and self.y == other.y
        elif isinstance(other, (tuple, list)):
            return self.x == other[0] and self.y == other[1]
        return False
    
    def __le__(self, other: "Node"):
        """代价<=比较 -> min(open_list)"""
        return self.cost <= other.cost
    
    def __lt__(self, other: "Node"):
        """代价<比较 -> min(open_list)"""
        return self.cost < other.cost



@dataclass
class NodeList:
    """节点存储队列: OpenList / CloseList"""

    queue: list[Node] = field(default_factory=list)

    def __bool__(self):
        """判断: while NodeList:"""
        return bool(self.queue)
    
    def __contains__(self, item):
        """包含: pos in NodeList"""
        return item in self.queue
        #NOTE: in是值比较, 只看value是否在列表, 不看id是否在列表

    def __len__(self):
        """长度: len(NodeList)"""
        return len(self.queue)
    
    def __getitem__(self, idx):
        """索引: NodeList[i]"""
        return self.queue[idx]
    
    # List操作
    def append(self, node: Node):
        """CloseList 添加节点"""
        self.queue.append(node)

    def pop(self, idx = -1):
        """CloseList 弹出节点"""
        return self.queue.pop(idx)
    
    # PriorityQueue操作
    def get(self):
        """OpenList 弹出代价最小节点"""
        idx = self.queue.index(min(self.queue)) 
        return self.queue.pop(idx) # 获取cost最小的节点, 并在NodeList中删除
        
    def put(self, node: Node):
        """OpenList 加入/更新节点"""
        if node in self.queue:
            idx = self.queue.index(node)
            if node.cost < self.queue[idx].cost:     # 新节点代价更小
                self.queue[idx].cost = node.cost     # 更新代价
                self.queue[idx].parent = node.parent # 更新父节点
        else:
            self.queue.append(node)

    def empty(self):
        """OpenList 是否为空"""
        return len(self.queue) == 0



# 贪婪最佳优先搜索算法
class GBFS:
    """GBFS算法"""

    def __init__(
        self,
        start_pos = START,
        end_pos = END,
        map_array = MAP.map_array,
        move_step = 3,
        move_direction = 8,
    ):
        """GBFS算法

        Parameters
        ----------
        start_pos : tuple/list
            起点坐标
        end_pos : tuple/list
            终点坐标
        map_array : ndarray
            二值化地图, 0表示障碍物, 255表示空白, H*W维
        move_step : int
            移动步数, 默认3
        move_direction : int (8 or 4)
            移动方向, 默认8个方向
        """
        # 网格化地图
        self.map_array = map_array # H * W

        self.width = self.map_array.shape[1]
        self.high = self.map_array.shape[0]

        # 起点终点
        self.start = Node(*start_pos) # 初始位置
        self.end = Node(*end_pos)     # 结束位置
       
        # Error Check
        if not self._in_map(self.start) or not self._in_map(self.end):
            raise ValueError(f"x坐标范围0~{self.width-1}, y坐标范围0~{self.height-1}")
        if self._is_collided(self.start):
            raise ValueError(f"起点x坐标或y坐标在障碍物上")
        if self._is_collided(self.end):
            raise ValueError(f"终点x坐标或y坐标在障碍物上")
       
        # 算法初始化
        self.reset(move_step, move_direction)


    def reset(self, move_step=3, move_direction=8):
        """重置算法"""
        self.__reset_flag = False
        self.move_step = move_step                # 移动步长(搜索后期会减小)
        self.move_direction = move_direction      # 移动方向 8 个
        self.close_list = NodeList()              # 存储已经走过的位置及其G值 
        self.open_list = NodeList()               # 存储当前位置周围可行的位置及其F值
        self.path_list = []                       # 存储路径(CloseList里的数据无序)

    
    def search(self):
        """搜索路径"""
        return self.__call__()


    def _in_map(self, node: Node):
        """点是否在网格地图中"""
        return (0 <= node.x < self.width) and (0 <= node.y < self.high) # 右边不能取等!!!
    

    def _is_collided(self, node: Node):
        """点是否和障碍物碰撞"""
        return self.map_array[node.y, node.x] == 0
    

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


    def _update_open_list(self, curr: Node):
        """open_list添加可行点"""
        for add in self._move():
            # 更新可行位置
            next_ = curr + add

            # 新位置是否在地图外边
            if not self._in_map(next_):
                continue
            # 新位置是否碰到障碍物
            if self._is_collided(next_):
                continue
            # 新位置是否在 CloseList 中
            if next_ in self.close_list:
                continue # in是值比较, 只看(x,y)是否在列表, 不看id是否在列表

            # 计算所添加的结点的代价
            H = next_ - self.end # 剩余距离估计
            next_.cost = H # G = 0
           
            # open-list添加结点
            self.open_list.put(next_)
            
            # 当剩余距离小时, 走慢一点
            if H < 20:
                self.move_step = 1


    def __call__(self):
        """GBFS路径搜索"""
        assert not self.__reset_flag, "call之前需要reset"
        print("搜索中\n")

        # 初始化列表
        self.open_list.put(self.start) # 初始化 OpenList

        # 正向搜索节点(CloseList里的数据无序)
        tic()
        while not self.open_list.empty():
            # 寻找 OpenList 代价最小的点, 并在OpenList中删除
            curr = self.open_list.get()
            # 更新 OpenList
            self._update_open_list(curr)
            # 更新 CloseList
            self.close_list.append(curr) # G始终为0
            # 结束迭代
            if curr == self.end:
                break
        print("路径节点搜索完成\n")
        toc()
    
        # 节点组合成路径
        tic()
        start = self.close_list[0]
        next_ = self.close_list[-1]
        while next_ != start:
            for i, curr in enumerate(self.close_list):
                if curr == next_.parent:             # 如果当前节点是目标节点的父节点
                    next_ = curr                     # 更新目标节点
                    self.path_list.append(curr)      # 将当前节点加入路径
                    self.close_list.pop(i)           # 弹出当前节点, 避免重复遍历
                    break
        print("路径节点整合完成\n")
        toc()

        # 需要重置
        self.__reset_flag = True
        
        return self.path_list








# debug
if __name__ == '__main__':
    p = GBFS()()
    MAP.show_path(p)
