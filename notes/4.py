你是一个python、tsp、强化学习专家，帮我review和优化下面方案 有无遗漏和错误点
# 1. 算法方案如下：

## 1.1数据集
tsp数据有两种方案：随机生成坐标，适配了10、20、30、50个城市；tsplib95库 适配了52城市数据
实现参考tsp_env.py，要求根据实际方案进行修改。
要求：
1.env返回给agent的状态 都要求包含 state状态：current_city_onehot + visited_mask + order_embedding + distances_from_current（归一化可改为基于当前城市到未访问城市的最远距离） + 当前step在episode第几步，然后具体agent自行选择
2.在 _load_tsplib_data 中添加异常处理，若 .opt.tour 文件不存在，则丢弃该数据
3.随机生成时：添加检查逻辑，确保坐标唯一性（如通过检查距离矩阵对角线外的零值）。若重复，重新生成坐标。
4.get_optimal_solution 使用 nearest neighbor 算法作为默认解，但未验证其质量。
改进：对于随机生成的实例，可引入更强的启发式算法（如 2-opt 或 Lin-Kernighan）作为参考解，选取最优结果
5.生成的数据和最优路径生成完成后，保存到文件，下次直接读文件 （单独的数据集生成模块）


# 1.2 算法方案版本
QDN
DQN_LSTM
输入：LSTM接收历史状态序列，每步状态为[visited_mask]（形状[N]，N为城市数）或扩展状态（如[current_city_onehot, visited_mask, order_embedding, distances_from_current]）。使用完整状态以提供更多上下文，形状为[步数, 特征维度]，
网络结构：
LSTM层：
层数：2层LSTM，平衡表达能力和过拟合风险。
隐藏单元数：256个隐藏单元，与MLP的中间层（256）对齐，确保容量一致。
输入维度：根据状态定义（如N或N+其他特征维度）。
输出维度：256，输出历史上下文的嵌入向量。

后续层：LSTM输出（[batch_size, 256]）接MLP（256-128-N），输出Q值（N个动作，对应选择下一个城市）。
实现细节：
初始化：初始化LSTM权重，减少训练初期的不稳定性。
Dropout（0.2~0.3）
序列处理：每个episode从头开始，初始隐藏状态设为零；若使用mini-batch，需按序列长度0 padding。
伪代码：

class DQN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, output_dim=N):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, state_seq, hidden=None):
        lstm_out, hidden = self.lstm(state_seq, hidden)  # state_seq: [batch, seq_len, input_dim]
        q_values = self.mlp(lstm_out[:, -1, :])  # 取最后时间步输出
        q_values[visited_mask.bool()] = -1e9  # Mask已访问城市
        return q_values
优化点：

若城市数较大（如50），可增加隐藏单元（如512）或层数（如3），但需监控过拟合。
考虑添加Dropout（0.2~0.3）在LSTM和MLP间，增强泛化能力。
若计算资源有限，可测试单层LSTM以降低开销。


Reinforce
ActorCritic
PPO
PPO：定义剪切参数（建议0.2）、GAE参数（λ=0.95，γ=0.99）

# 2.实验方案
消融实验（Ablation Study）设计方案

实验组织结构：
基准算法：DQN_optimal（包含所有组件）
消融目标：验证每个组件的必要性和贡献度


state状态：current_city_onehot + visited_mask + order_embedding + distances_from_current（按当前城市的距离动态归一化（如除以当前城市到其他城市的最大距离），或使用对数变换以压缩动态范围。） + 当前step在episode第几步

消融实验一：状态表示组件分析



消融组合：
1. Ablation-1：仅current_city_onehot + visited_mask（移除order_embedding和distances）
2. Ablation-2：current_city_onehot + visited_mask + order_embedding（移除distances）
3. Ablation-3：current_city_onehot + visited_mask + distances_from_current（移除order_embedding）
4. Ablation-4：仅visited_mask（移除current_city_onehot，对应原DQN_visited）

实验控制：
* 相同网络结构：MLP（512-256-128）
* 相同训练参数：学习率、批量大小、探索策略
* 相同数据集：10、20、30、50城市各100个实例
* 重复实验：每种配置5次独立运行


# 3.代码相关
- 1.日志要求保存文件的同时也输出控制台，定义标准日志格式（如 [时间戳][算法][城市数][状态类型][episode][step]:xxxx）
- 2.算法版本使用个不同的agent来实现
- 3.有一个总入口函数 能一键启动（另外，代码模块之间 不要使用sys.argv），参考代码：
#!/bin/bash
printf 'y\n' | ./clean.sh
caffeinate -s python run.py
- 4.要求有单独的配置文件模块，里面可以配置 训练和数据集生成的相关选项，比如状态选择等；模型的结构、参数也要可配置，单独一个模型配置文件；配置支持定义每种ablation对应的状态组合。配置文件使用yaml的格式
- 5.有数据集生成的模块，训练的时候不需要再生成数据，并且seed可以全局配置
- 6.实验中日志、模型、csv都要保存，要求以实验名字/日期/的结构。其中使用分层CSV存储（如results/实验名/日期/算法名/），避免单一文件过大。打印日志的时候需要打印实验的层级信息，见伪代码中for的层级信息 和训练信息
- 7.项目中依赖的库要放到requirements.txt中，gitignore（要求提出claude相关的配置）
- 8.代码方案主逻辑：

(1)网络中visited_mask要特殊处理：
Mask 要在网络末尾乘 logits，否则训练不稳定。
logits[visited_mask.bool()] = -1e9
logp = F.log_softmax(logits, dim=-1)
(2)神经网络 共用 backbone + 条件输入
(3)评估器统一写一次
(4)reward单独模块

Per-instance模式（独立训练）：
每个实例独立训练网络
在测试集实例上测试
运行次数：城市数 × 状态表示类型 × 实例数 × 重复次数

Cross-instance模式（泛化训练）：
在训练集上共享训练一个网络
在测试集上测试
运行次数：城市数 × 状态表示类型 × 重复次数

伪代码：
算法组=[算法A，....]
状态表示类型 = ['full', 'ablation_1', 'ablation_2', 'ablation_3', 'ablation_4']
城市数列表 = [10, 20, 30, 50]
总实例数 = 100
训练实例数 = 80
测试实例数 = 20
重复运行次数 = 5
for 算法A in 算法组：
    模式一：Per-instance模式
    for 城市数 in 城市数列表:
        for 状态表示 in 状态表示类型:

            # 数据集划分
            all_instances = load_instances(城市数, 总实例数)
            train_instances = all_instances[:训练实例数]
            test_instances = all_instances[训练实例数:]

            # 在训练集上独立训练每个实例
            for 实例ID in range(训练实例数):
                for 运行次数 in range(重复运行次数):
                    网络 = 重新初始化()
                    训练结果 = 从零训练(train_instances[实例ID], 状态表示)
                    存储训练结果(mode='per_instance_train')

                # 在测试集上zero-shot测试
                for 实例ID in range(测试实例数):
                    for 运行次数 in range(重复运行次数):
                        网络 = 重新初始化()  # 每次测试都用新网络
                        测试结果 = zero_shot_test(test_instances[实例ID], 状态表示)
                        存储测试结果(mode='per_instance_test')

    模式二：Cross-instance模式
    for 城市数 in 城市数列表:
        for 状态表示 in 状态表示类型:
            # 数据集划分
            all_instances = load_instances(城市数, 总实例数)
            train_instances = all_instances[:训练实例数]
            test_instances = all_instances[训练实例数:]

            for 运行次数 in range(重复运行次数):
                网络 = 重新初始化()

                # 在训练集上共享训练
                for episode in total_episodes:
                    sample_instance = random.choice(train_instances)
                    更新网络(sample_instance, 状态表示)

                # 在测试集上zero-shot测试
                for test_id in range(测试实例数):
                    测试结果 = zero_shot_inference(网络, test_instances[test_id])
                    存储测试结果(mode='cross_instance_test')

每组运行需要保存如下csv数据
CSV表头字段说明：
algorithm: 算法名称，取值[DQN_visited, DQN_LSTM, Reinforce, ActorCritic, DQN_order, DQN_optimal]
city_num: 城市数量，取值[10, 20, 30, 50]
mode: 训练模式，取值[per_instance, cross_instance]
instance_id: 实例ID，范围0-99，用于标识100个TSP实例
run_id: 运行次数，范围0-4，表示5次独立运行
state_type: 状态表示类型，取值[full, ablation_1, ablation_2, ablation_3, ablation_4]
train_test: 数据集类型，取值[train, test]
episode: 训练或测试的episode编号，从0开始
step: episode内的step编号，从0开始
state: 状态表示，JSON字符串，包含current_city_onehot、visited_mask、order_embedding、distances_from_current（根据state_type选择）
done: 是否为episode的最后一步，取值[0, 1]
reward: 该step的奖励值，浮点数
loss: 该step的损失值，浮点数（DQN类算法记录step级别loss，on-policy算法如Reinforce为空）
total_reward: 当前epsiode到当前step的reward总和

Cross-instance模式的目标是训练一个泛化网络，处理不同TSP实例。随机采样实例可能导致训练偏向某些实例，mini-batch训练可提高效率和稳定性：
Batch大小：建议batch_size=4~8，平衡计算效率和梯度稳定性。城市规模较大（如50）时，可减小到4以降低内存需求。
实例选择：每个batch从训练集（80实例）中随机采样k个实例（k=batch_size），确保多样性。
序列处理：对于LSTM（如DQN_LSTM），batch中各实例的序列长度可能不同，需padding到最大长度（如N步，N为城市数）。
经验回放：对于DQN类算法，使用经验回放缓冲区（Replay Buffer）存储经验，容量建议为10000~50000，采样batch_size=64的经验进行更新。




------tsp_env.py----
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist as distance_matrix
from heapq import heappop, heappush, heapify
import random
from typing import Optional, Dict, Any, Tuple, List
import os
import urllib.request
import zipfile

from confs.path import project_root

try:
    import tsplib95
except ImportError:
    print("Warning: tsplib95 not installed. Only random mode will work.")
    print("Install with: pip install tsplib95")
    tsplib95 = None


class TSPEnvironment:
    """
    Traveling Salesman Problem environment for reinforcement learning.

    The agent starts at city 0 and must visit all cities exactly once before returning to the start.
    """

    def __init__(self,
                 n_cities: int = 10,
                 coordinates: Optional[np.ndarray] = None,
                 seed: Optional[int] = None,
                 use_tsplib: bool = False,
                 tsplib_name: Optional[str] = None,
                 tsplib_path: str = project_root+"/tsplib_data"):
        """
        Initialize TSP environment.

        Args:
            n_cities: Number of cities (used for random mode or if tsplib problem has different size)
            coordinates: Optional pre-defined city coordinates (n_cities x 2)
            seed: Random seed for reproducibility
            use_tsplib: Whether to use TSPLIB95 dataset
            tsplib_name: Name of TSPLIB problem (e.g., 'berlin52', 'eil51')
            tsplib_path: Path to TSPLIB data files
        """
        self.n_cities = n_cities
        self.seed = seed
        self.use_tsplib = use_tsplib
        self.tsplib_name = tsplib_name
        self.tsplib_path = tsplib_path
        self.optimal_distance = None

        # Download and prepare TSPLIB data if needed
        if use_tsplib and tsplib95 is not None:
            self._prepare_tsplib_data()

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Initialize city coordinates
        if coordinates is not None:
            assert coordinates.shape == (n_cities, 2), f"Coordinates must be {n_cities}x2"
            self.coordinates = coordinates.copy()
        elif use_tsplib and tsplib95 is not None and tsplib_name:
            self.coordinates, self.optimal_path = self._load_tsplib_data()
            self.n_cities = len(self.coordinates)
        else:
            self.coordinates = np.random.uniform(0, 1, (n_cities, 2))

        # Precompute distance matrix
        self.distance_matrix = self._compute_distance_matrix()
        self.optimal_path, self.optimal_distance = self.get_optimal_solution()

        # State variables
        self.current_city = 0
        self.visited = set([0])  # Start at city 0
        self.path = [0]
        self.total_distance = 0.0
        self.done = False

    def _prepare_tsplib_data(self):
        """Download and prepare TSPLIB data if not exists."""
        import gzip
        import shutil

        if not os.path.exists(self.tsplib_path):
            os.makedirs(self.tsplib_path)

        # List of common EUC_2D problems to download
        euc_2d_problems = [
            'berlin52', 'eil51', 'eil76', 'eil101', 'ch130', 'ch150',
            'a280', 'pr76', 'rat195', 'kroA100', 'kroB100', 'kroC100',
            'kroD100', 'kroE100', 'rd100', 'st70'
        ]

        base_url = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/"

        for problem in euc_2d_problems:
            tsp_file = f"{problem}.tsp"
            tsp_gz_file = f"{problem}.tsp.gz"
            opt_file = f"{problem}.opt.tour"
            opt_gz_file = f"{problem}.opt.tour.gz"

            tsp_path = os.path.join(self.tsplib_path, tsp_file)
            tsp_gz_path = os.path.join(self.tsplib_path, tsp_gz_file)
            opt_path = os.path.join(self.tsplib_path, opt_file)
            opt_gz_path = os.path.join(self.tsplib_path, opt_gz_file)

            # Download and extract .tsp file
            if not os.path.exists(tsp_path):
                try:
                    # Download the gzipped file
                    url = f"{base_url}{tsp_gz_file}"
                    urllib.request.urlretrieve(url, tsp_gz_path)
                    print(f"Downloaded {tsp_gz_file}   from {url}")

                    # Extract the gzipped file
                    with gzip.open(tsp_gz_path, 'rb') as f_in:
                        with open(tsp_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    print(f"Extracted {tsp_file}")

                    # Optionally remove the gz file after extraction
                    os.remove(tsp_gz_path)
                except Exception as e:
                    print(f"Failed to download or extract {tsp_gz_file}: {e}  {url}")

            # Download and extract .opt.tour file
            if not os.path.exists(opt_path):
                try:
                    # Download the gzipped file
                    url = f"{base_url}{opt_gz_file}"
                    urllib.request.urlretrieve(url, opt_gz_path)
                    print(f"Downloaded {opt_gz_file} from {url}")

                    # Extract the gzipped file
                    with gzip.open(opt_gz_path, 'rb') as f_in:
                        with open(opt_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    print(f"Extracted {opt_file}")

                    # Optionally remove the gz file after extraction
                    os.remove(opt_gz_path)
                except Exception as e:
                    print(f"Failed to download or extract {opt_gz_file}: {e}")

    def _load_tsplib_data(self) -> Tuple[np.ndarray, Optional[float]]:
        """Load coordinates from TSPLIB dataset."""
        tsp_file = os.path.join(self.tsplib_path, f"{self.tsplib_name}.tsp")
        opt_file = os.path.join(self.tsplib_path, f"{self.tsplib_name}.opt.tour")

        if not os.path.exists(tsp_file):
            raise FileNotFoundError(f"TSPLIB file not found: {tsp_file}")

        # Load TSP problem
        problem = tsplib95.load(tsp_file)

        # Check if it's EUC_2D type
        if problem.edge_weight_type != 'EUC_2D':
            raise ValueError(f"Only EUC_2D problems are supported, got {problem.edge_weight_type}")

        # Extract coordinates
        coordinates = []
        for node in sorted(problem.node_coords.keys()):
            coordinates.append(problem.node_coords[node])
        coordinates = np.array(coordinates)

        # Normalize coordinates to [0, 1]
        min_coords = np.min(coordinates, axis=0)
        max_coords = np.max(coordinates, axis=0)
        coord_range = max_coords - min_coords
        normalized_coords = (coordinates - min_coords) / coord_range

        # Load optimal tour and distance if available
        optimal_distance = None
        if os.path.exists(opt_file):
            try:
                solution = tsplib95.load(opt_file)
                # Calculate optimal distance with normalized coordinates
                opt_tour = solution.tours[0]


                # Convert to 0-based indexing and ensure starts with 0
                tour = [city - 1 for city in opt_tour]
                if tour[0] != 0:
                    # Rotate tour to start with city 0
                    start_idx = tour.index(0)
                    tour = tour[start_idx:] + tour[:start_idx]
                tour.append(0)  # Return to start

            except Exception as e:
                print(f"Failed to load optimal solution: {e}")

        return normalized_coords, tour

    def _compute_distance_matrix(self) -> np.ndarray:
        return distance_matrix(self.coordinates, self.coordinates)

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Reset environment to initial state.

        Args:
            seed: Optional seed for new random coordinates

        Returns:
            Initial state information
        """
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)
            random.seed(seed)

            # Only regenerate coordinates if not using TSPLIB
            if not self.use_tsplib:
                self.coordinates = np.random.uniform(0, 1, (self.n_cities, 2))
                self.distance_matrix = self._compute_distance_matrix()
                self.optimal_path, self.optimal_distance = self.get_optimal_solution()

        # Reset state
        self.current_city = 0
        self.visited = set([0])
        self.path = [0]
        self.total_distance = 0.0
        self.done = False

        return self._get_state()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take action in environment.

        Args:
            action: Next city to visit (0 to n_cities-1)

        Returns:
            state: New state
            reward: Reward for this action
            done: Whether episode is finished
            info: Additional information
        """
        if self.done:
            raise ValueError("Environment is done. Call reset() to start new episode.")

        # Check if action is valid (not already visited, unless returning to start when all visited)
        if len(self.visited) == self.n_cities:
            # All cities visited, must return to start
            if action != 0:
                # Invalid action - force return to start
                action = 0
        else:
            # Still cities to visit
            if action in self.visited:
                # Invalid action - choose random unvisited city
                unvisited = [i for i in range(self.n_cities) if i not in self.visited]
                action = random.choice(unvisited)

        # Calculate reward (negative distance)
        distance = self.distance_matrix[self.current_city, action]
        reward = -distance
        self.total_distance += distance

        # Update state
        self.current_city = action
        self.visited.add(action)
        self.path.append(action)

        # Check if episode is done
        if len(self.visited) == self.n_cities and action == 0:
            self.done = True  # path = n_city + 1 (return to the start)

        info = {
            'total_distance': self.total_distance,
            'path': self.path.copy(),
            'visited_all': len(self.visited) == self.n_cities,
            'is_valid_path': self._is_valid_path(),
            'optimal_distance': self.optimal_distance,
            'gap_to_optimal': (self.total_distance - self.optimal_distance) / self.optimal_distance if self.optimal_distance else None
        }

        return self._get_state(), reward, self.done, info

    def _get_state(self) -> Dict[str, Any]:
        """Get current state representation."""
        # --------lstm
        # Convert path to sequence of one-hot vectors
        sequence_length = len(self.path)
        sequence_onehot = np.zeros((sequence_length, self.n_cities))

        for i, city in enumerate(self.path):
            sequence_onehot[i, city] = 1.0
        # --------lstm

        # Basic one-hot encoding of current city
        current_city_onehot = np.zeros(self.n_cities)
        current_city_onehot[self.current_city] = 1.0

        # Visited mask
        visited_mask = np.zeros(self.n_cities)
        for city in self.visited:
            visited_mask[city] = 1.0

        sequence_length = len(self.path)
        sequence_onehot = np.zeros((sequence_length, self.n_cities))

        for i, city in enumerate(self.path):
            sequence_onehot[i, city] = 1.0
        # Distance to all cities from current position
        distances_from_current = self.distance_matrix[self.current_city].copy()

        # Normalize distances to [0, 1]
        max_dist = np.max(self.distance_matrix)
        distances_from_current = distances_from_current / max_dist if max_dist > 0 else distances_from_current

        # Order embedding: use visit order as values
        order_embedding = np.zeros(self.n_cities)
        for i, city in enumerate(self.path):
            order_embedding[city] = min((i+1)/self.n_cities, 1)

        return {
            'current_city_onehot': current_city_onehot,   #  1:current city  0:other order_embedding   DQNOptimal
            'sequence_onehot': sequence_onehot,
            'sequence_length': sequence_length,
            'visited_mask': visited_mask,  # 1:visited 0: unvisited         basic  DQNLSTM  reinforce  ActorCritic  DQNOptimal
            'order_embedding': order_embedding,  # visited index / n_cities        order_embedding   DQNOptimal
            'distances_from_current': distances_from_current, #distance to others       DQNOptimal
            'current_city': self.current_city,
            'visited': self.visited.copy(),
            'path_sequence': self.path.copy(),
            'coordinates': self.coordinates.copy()
        }

    def _is_valid_path(self) -> bool:
        """Check if current path is valid (visits all cities exactly once)."""
        if not self.done:
            return False
        return len(set(self.path[:-1])) == self.n_cities and self.path[-1] == 0

    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions from current state."""
        if len(self.visited) == self.n_cities:
            # Must return to start
            return [0]
        else:
            # Can visit any unvisited city
            return [i for i in range(self.n_cities) if i not in self.visited]

    def get_action_mask(self) -> np.ndarray:
        """Get mask for valid actions (1 for valid, 0 for invalid)."""
        mask = np.zeros(self.n_cities)
        valid_actions = self.get_valid_actions()
        mask[valid_actions] = 1.0
        return mask

    def render(self, save_path: Optional[str] = None, show: bool = True) -> None:
        """
        Visualize current state of TSP.

        Args:
            save_path: Optional path to save figure
            show: Whether to display figure
        """
        plt.figure(figsize=(8, 8))

        # Plot cities
        plt.scatter(self.coordinates[:, 0], self.coordinates[:, 1],
                    c='red', s=100, zorder=3)

        # Label cities
        for i, (x, y) in enumerate(self.coordinates):
            plt.annotate(str(i), (x, y), xytext=(5, 5),
                         textcoords='offset points', fontsize=12)

        # Plot path
        if len(self.path) > 1:
            path_coords = self.coordinates[self.path]
            plt.plot(path_coords[:, 0], path_coords[:, 1],
                     'b-', linewidth=2, alpha=0.7, zorder=2)

            # Highlight current city
            current_coord = self.coordinates[self.current_city]
            plt.scatter(current_coord[0], current_coord[1],
                        c='green', s=200, marker='*', zorder=4)

        title = f'TSP Environment - {len(self.visited)}/{self.n_cities} cities visited\n'
        title += f'Current city: {self.current_city}, Total distance: {self.total_distance:.3f}'
        if self.use_tsplib and self.tsplib_name:
            title += f'\nDataset: {self.tsplib_name}'
            if self.optimal_distance:
                gap = (self.total_distance - self.optimal_distance) / self.optimal_distance * 100
                title += f' (Optimal: {self.optimal_distance:.3f}, Gap: {gap:.1f}%)'

        plt.title(title)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def get_optimal_solution(self) -> Tuple[List[int], float]:
        """
        Get optimal solution using nearest neighbor heuristic with a priority queue.
        Note: This is not guaranteed to be optimal, just a reasonable approximation.
        For TSPLIB problems, returns the known optimal if available.
        """
        # If using TSPLIB and optimal solution is known, try to load it
        if self.use_tsplib and self.tsplib_name and self.optimal_path:
            total_dist = 0
            for index, current in enumerate(self.optimal_path[:-1]):
                print('index=',index,current,self.optimal_path[index+1])
                total_dist += self.distance_matrix[current, self.optimal_path[index+1]]

            return self.optimal_path, total_dist


        # Fallback to nearest neighbor heuristic
        # Initialize heap with distances from starting city (0)
        heap = [(self.distance_matrix[0, i], i) for i in range(1, self.n_cities)]
        heapify(heap)
        path = [0]
        total_dist = 0.0
        visited = {0}  # Track visited cities

        current = 0
        while len(visited) < self.n_cities:
            # Get nearest unvisited city
            while heap:
                dist, next_city = heappop(heap)
                if next_city not in visited:
                    break
            else:
                break  # No more unvisited cities

            total_dist += dist
            path.append(next_city)
            visited.add(next_city)

            # Add distances to unvisited cities from the new current city
            current = next_city
            for i in range(1, self.n_cities):
                if i not in visited:
                    heappush(heap, (self.distance_matrix[current, i], i))

        # Return to start
        total_dist += self.distance_matrix[current, 0]
        path.append(0)

        return path, total_dist


# Usage examples:
def create_random_env():
    """Create environment with random coordinates."""
    return TSPEnvironment(n_cities=10, seed=42, use_tsplib=False)


def create_tsplib_env(problem_name='berlin52'):
    """Create environment with TSPLIB dataset."""
    return TSPEnvironment(use_tsplib=True, tsplib_name=problem_name)

