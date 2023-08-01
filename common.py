# -*- coding: utf-8 -*-
"""
 Created on Fri May 26 2023 16:03:59
 Modified on 2023-5-26 16:03:59
 
 @auther: HJ https://github.com/zhaohaojie1998
"""
# 算法共同组成部分
from typing import Union
import cv2
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
from dataclasses import dataclass, field
Number = Union[int, float]


__all__ = ['tic', 'toc', 'limit_angle', 'GridMap', 'PriorityQueuePro', 'ListQueue', 'SetQueue', 'Node']




# 坐标节点
@dataclass(eq=False)
class Node:
    """节点"""

    x: int
    y: int
    cost: Number = 0      # 父节点到节点的代价
    parent: "Node" = None # 父节点指针

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
        """坐标x,y比较 -> node in list"""
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
    
    def __hash__(self) -> int:
        """使可变对象可hash, 能放入set中 -> node in set"""
        return hash((self.x, self.y)) # tuple可hash
        # data in set 时间复杂度为 O(1), 但data必须可hash
        # data in list 时间复杂度 O(n)





# Set版优先队列
@dataclass
class SetQueue:
    """节点优先存储队列 set 版"""

    queue: set[Node] = field(default_factory=set)

    # Queue容器增强
    def __bool__(self):
        """判断: while Queue:"""
        return bool(self.queue)
    
    def __contains__(self, item):
        """包含: pos in Queue"""
        return item in self.queue
        #NOTE: in是值比较, 只看hash是否在集合, 不看id是否在集合

    def __len__(self):
        """长度: len(Queue)"""
        return len(self.queue)
    
    # PriorityQueue操作
    def get(self):
        """Queue 弹出代价最小节点"""
        node = min(self.queue)  # O(n)?
        self.queue.remove(node) # O(1)
        return node
        
    def put(self, node: Node):
        """Queue 加入/更新节点"""
        if node in self.queue:              # O(1)
            qlist = list(self.queue)        # 索引元素, set无法索引需转换
            idx = qlist.index(node)         # O(n)
            if node.cost < qlist[idx].cost: # 新节点代价更小则加入新节点
                self.queue.remove(node)     # O(1)
                self.queue.add(node)        # O(1) 移除node和加入node的hash相同, 但cost和parent不同
        else:
            self.queue.add(node)            # O(1)

    def empty(self):
        """Queue 是否为空"""
        return len(self.queue) == 0
    




# List版优先队列
@dataclass
class ListQueue:
    """节点优先存储队列 list 版"""

    queue: list[Node] = field(default_factory=list)

    # Queue容器增强
    def __bool__(self):
        """判断: while Queue:"""
        return bool(self.queue)
    
    def __contains__(self, item):
        """包含: pos in Queue"""
        return item in self.queue
        #NOTE: in是值比较, 只看value是否在列表, 不看id是否在列表

    def __len__(self):
        """长度: len(Queue)"""
        return len(self.queue)
    
    def __getitem__(self, idx):
        """索引: Queue[i]"""
        return self.queue[idx]
    
    # List操作
    def append(self, node: Node):
        """List 添加节点"""
        self.queue.append(node) # O(1)

    def pop(self, idx = -1):
        """List 弹出节点"""
        return self.queue.pop(idx) # O(1) ~ O(n)
    
    # PriorityQueue操作
    def get(self):
        """Queue 弹出代价最小节点"""
        idx = self.queue.index(min(self.queue)) # O(n) + O(n)
        return self.queue.pop(idx)              # O(1) ~ O(n)
        
    def put(self, node: Node):
        """Queue 加入/更新节点"""
        if node in self.queue:                       # O(n)
            idx = self.queue.index(node)             # O(n)
            if node.cost < self.queue[idx].cost:     # 新节点代价更小
                self.queue[idx].cost = node.cost     # O(1) 更新代价 
                self.queue[idx].parent = node.parent # O(1) 更新父节点 
        else:
            self.queue.append(node)                  # O(1)
        
        # NOTE try语法虽然时间复杂度更小, 但频繁抛出异常速度反而更慢
        # try:
        #     idx = self.queue.index(node)             # O(n)
        #     if node.cost < self.queue[idx].cost:     # 新节点代价更小
        #         self.queue[idx].cost = node.cost     # O(1) 更新代价 
        #         self.queue[idx].parent = node.parent # O(1) 更新父节点 
        # except ValueError:
        #     self.queue.append(node)                  # O(1)

    def empty(self):
        """Queue 是否为空"""
        return len(self.queue) == 0




# 原版优先队列增强(原版也是list实现, 但get更快, put更慢)
class PriorityQueuePro(PriorityQueue):
    """节点优先存储队列 原版"""

    # PriorityQueue操作
    def put(self, item, block=True, timeout=None):
        """Queue 加入/更新节点"""
        if item in self.queue:                # O(n)
            return # 修改数据会破坏二叉树结构, 就不存了
        else:
            super().put(item, block, timeout) # O(logn)

    # Queue容器增强
    def __bool__(self):
        """判断: while Queue:"""
        return bool(self.queue)
    
    def __contains__(self, item):
        """包含: pos in Queue"""
        return item in self.queue
        #NOTE: in是值比较, 只看value是否在列表, 不看id是否在列表

    def __len__(self):
        """长度: len(Queue)"""
        return len(self.queue)
    
    def __getitem__(self, idx):
        """索引: Queue[i]"""
        return self.queue[idx]





# 图像处理生成网格地图
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




# 角度归一化
def limit_angle(x, mode=1):
    """ 
    mode1 : (-inf, inf) -> (-π, π] 
    mode2 : (-inf, inf) -> [0, 2π)
    """
    x = x - x//(2*math.pi) * 2*math.pi # any -> [0, 2π)
    if mode == 1 and x > math.pi:
        return x - 2*math.pi           # [0, 2π) -> (-π, π]
    return x







