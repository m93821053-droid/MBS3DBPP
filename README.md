
# 🧩 3D 装箱问题求解器（基于 PPO + Transformer）

本项目实现了一个基于**近端策略优化（PPO）**和**Transformer 架构**的智能体，用于解决**多规格多容器三维装箱问题（Multi-bin-size 3D Bin Packing）**。
在该任务中，智能体需要从给定的多种不同箱子（bin）尺寸中任意选择，并决策给定的每个物品的放置位置与旋转方式，以最大化最终选择使用的多个箱子的空间利用率。

---

## 📌 主要特点

- **分层动作空间**：依次决策选择箱子类型、物品、放置坐标和旋转角度（6 种）。
- **高效状态表示**：使用高度图 + 距离特征（支持悬空检测与支撑约束）。
- **并行采样**：多进程同时与环境交互，大幅提升数据收集效率。
- **可视化输出**：内置 Plotly 3D 绘图，可调用直观展示装箱布局。

---
## 📚 核心参考文献

本项目的算法设计主要基于以下工作：
```bibtex
@article{QUE2023119153,
title = {Solving 3D packing problem using Transformer network and reinforcement learning},
journal = {Expert Systems with Applications},
volume = {214},
pages = {119153},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2022.119153},
url = {https://www.sciencedirect.com/science/article/pii/S0957417422021716},
author = {Quanqing Que and Fang Yang and Defu Zhang},
keywords = {3D packing problem, Deep reinforcement learning, Transformer},
abstract = {The three-dimensional packing problem (3D-PP) is a classic NP-hard problem in operations research and computer science. One of the most popular ways to solve the problem is heuristic methods with a search strategy. However, approaches based on machine learning have recently received widespread attention because of their efficiency. In this work, we propose a deep reinforcement learning (DRL) model to solve 3D-PP. Our method employs Transformer architecture as the policy network and uses Proximal Policy Optimization (PPO) to train the network. Compared with previous approaches using DRL, our method presents a novel state representation of packing environment, and introduces plane features for representing the length and width information of container. Our method achieves the new state-of-the-art results for using DRL to solve 3D-PP. The code of our method will be released to facilitate future research.}
}
```
## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/yourname/3D-Bin-Packing-PPO.git
cd 3D-Bin-Packing-PPO
```

### 2. 安装依赖

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 生成测试数据集（可选）

```bash
python scripts/generate_dataset.py
```
具体参数可在`generate_dataset.py`中设置

### 4. 训练模型

#### 使用默认配置（硬编码 + 自动加载 YAML）
```bash
python src/train.py
```

#### 修改训练参数
- 编辑 `src/arguments.py` 

训练过程中：
- 模型权重保存至 `save/` 目录（按参数自动命名子文件夹）
- 日志保存至 `log/` 和 TensorBoard 记录（`runs/`）
- 空间利用率（UR）实时记录在 `save/.../result_ur.txt`

### 5. 测试训练好的模型

编辑 `src/test.py`，指定模型路径：
```python
load_file_path = r"./save/100_100_100_10_10_50_1_5_0_0_4_None_2_50_0/actor.pth"
```
然后运行：
```bash
python src/test.py
```

输出示例：
```
average use ratio: [85.23, 87.45, 88.12, 88.98, 89.34]
average time: 0.832s
```

---

## 📊 结果可视化

测试完成后，可调用 `plot.py` 绘制特定装箱布局：

```python
from src.plot import plotResult

# packing_result 是一个列表，每个元素为 [l, w, h, x, y, z]
plotResult(packing_result, bin_size_x=100, bin_size_y=100, bin_size_z=100)
```


---

## ⚙️ 参数配置详解

所有可调参数均集中在 `configs/default.yaml` 中（若无此文件，则回退到 `arguments.py` 的硬编码默认值）。

| 分类 | 参数 | 说明 |
|------|------|------|
| **训练** | `gamma` | 折扣因子 |
| | `lr_actor` / `lr_critic` | 网络学习率 |
| | `batch_size` | 每次更新的样本量 |
| | `target_step` | 每轮收集的步数 |
| | `repeat_times` | 每轮更新次数 |
| | `trunc_step` | 截断步长，控制奖励回传长度 |
| **环境** | `bin_type_list` | 可用箱子类型列表，如 `[[50,50,50], [80,80,80], ...]` |
| | `box_num` | 每个实例的盒子个数 |
| | `min_factor` / `max_factor` | 盒子尺寸相对于箱子尺寸的缩放范围 |
| | `orientation` | 可选旋转数量（2 或 6） |
| | `support_constraint` | 是否启用支撑约束（盒子需被完全支撑） |
| **网络** | `d_model` | Transformer 嵌入维度 |
| | `n_head` | 多头注意力头数 |
| | `nlayers` | 编码器层数 |
| **并行** | `process_num` | 并行环境数量（建议 ≤ GPU 数×2） |

---

## 📁 项目结构

```
3D-Bin-Packing-PPO/
├── scripts/
│   └── generate_dataset.py       # 生成测试数据集
├── src/
│   ├── agent.py                  # PPO 代理（动作选择、存储、更新）
│   ├── Agent_explore_env.py      # 多进程环境交互
│   ├── arguments.py              # 参数定义 + YAML 加载器
│   ├── CuttingBox.py             # 盒子尺寸生成器
│   ├── environment.py            # 3D 装箱环境核心逻辑
│   ├── network.py                # Actor/Critic 网络（Transformer）
│   ├── plot.py                   # 3D 可视化
│   ├── test.py                   # 模型测试入口
│   └── train.py                  # 训练入口（多进程采样）
├── datasets/                     # 存放 HDF5 数据集（自动生成）
├── save/                         # 模型和日志保存目录（训练自动创建）
├── log/                          # 训练日志
├── runs/                         # TensorBoard 记录
├── requirements.txt
├── LICENSE
└── README.md
```

---


## 📄 许可证

本项目采用 [MIT License](LICENSE)，欢迎自由使用和修改。

---

**Happy Packing! 🎯**
```