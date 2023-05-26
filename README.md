# 路径规划算法

## 算法:

| 算法                                                         | file        | 类别                |
| ------------------------------------------------------------ | ----------- | ------------------- |
| A星算法<br />(A*)                                            | A_star.py   | 启发搜索<br />F=G+H |
| 贪婪最佳优先搜索算法<br />(Greedy Best First Search, GBFS)   | GBFS.py     | 启发搜索<br />F=H   |
| 迪杰斯特拉搜索算法<br />(Dijkstra)                           | Dijkstra.py | 启发搜索<br />F=G   |
| 深度优先搜索算法<br />(Depth First Search, DFS)              | DFS.py      | 遍历搜索            |
| 广度优先搜索算法<br />(Breadth First Search, BFS)            | BFS.py      | 遍历搜索            |
| 概率路图算法<br />(Probabilistic Road Map, PRM)              |             | 采样                |
| 快速随机扩展树算法<br />(Rapidly-exploring Random Tree, RRT) |             | 采样                |

## 使用方法:

在草纸上随便画点障碍物，拍照上传替换鲲鲲图片 image.jpg，在 A_star.py 等脚本中设置起点终点等参数，运行即可.

程序并没有设置复杂的继承/依赖关系，只需要如 utils.py + A_star.py + image.jpg 三个文件在同一目录就能运行.

## 效果:

**复杂障碍物地图下的路径规划结果（只能看一眼，不然会爆炸）**

###### A*算法：（介于最优和快速之间）

List耗时0.63s，PriorityQueue耗时0.48s，由于List存储结构能动态更新OpenList中Node的cost和parent信息，路径会更优

List存储结构：

![](图片/astar.png)

PriorityQueue存储结构：

![](图片/astar_1.png)

###### Dijkstra算法：（最优路径，耗时较大）

List耗时81s，PriorityQueue耗时15s,  OpenList的数据结构对Dij算法的结果影响不太大，采用优先队列更快

![](图片/dij.png)

###### GBFS算法：（速度最快，路径较差）

List耗时0.19s，速度本来就很快，没必要上PriorityQueue了

![](图片/gbfs.png)

###### BFS算法：（最优路径，耗时较大）

Deque耗时52s，结果比Dij更好点

![](图片/bfs.png)

###### DFS算法：（最烂路径，速度最慢）

低分辨率鸡哥地图耗时7.5s，高分辨率鸡哥地图搜索太慢没跑出来

![](图片/dfs.png)

## Requirement:

python  >= 3.9

opencv-python >= 4.7.0.72

matplotlib >= 3.5.1

numpy >= 1.22.3

###### 广告:

[DRL-for-Path-Planning: 深度强化学习路径规划, SAC路径规划](https://github.com/zhaohaojie1998/DRL-for-Path-Planning)

[Grey-Wolf-Optimizer-for-Path-Planning: 灰狼优化算法路径规划、多智能体/多无人机航迹规划](https://github.com/zhaohaojie1998/Grey-Wolf-Optimizer-for-Path-Planning)
