from sympy.strategies.branch import do_one

Optimisation of path generation for parcel delivery with multiple destinations via an Android application
todo：
1.lstm 输入one-hot效果差的原因，并且改进网络  done
2.适配tsplib，从而不需要计算最优路径；并且支持tsplib和random生成两种方案  done
3.comparison对比计算逻辑 检查。done


1.为什么特殊处理区分网络 与update。 done
2.3版本对应2.2版本区别 是否合并。done
3.分层，神经网络如何解决聚类问题 done
4.   版本2.2 vs 3(faster, sicne more status representation) done


1.如何做实验
2.近期强化学习的论文
3.近期强化学习解决tsp的论文
4.alphgo的视频和论文


下面是tsp状态设计方案，要求以模型维度（顺序：QDN_visited、DQN_LSTM  Reinforce  ActorCritic、DQN_order、 DQN_optimal），输出每个模型是输入状态，带上例子。回复的时候 不要使用markdown和python代码。给的输入例子，要求符合状态逻辑
'current_city_onehot': current_city_onehot,   # 状态输入说明 1:current city  0:other   模型： DQN_order   DQN_optimal
'visited_mask': visited_mask,  # 态输入说明 1:visited 0: unvisited         模型 QDN_visited  DQN_LSTM  Reinforce  ActorCritic  DQN_optimal
'order_embedding': order_embedding,  # 态输入说明（visited index+1） / n_cities      模型  DQN_order   DQN_optimal
'distances_from_current': distances_from_current, #态输入说明 distance to others   模型    DQN_optimal










针对TSP（旅行商问题，10-50城市，闭合路径），从基础版本逐步引入改进，如序列处理、状态扩展、Reward Shaping等，以解决状态混叠（state aliasing）、学习困难等问题。
版本序列基于：1. Reinforce版本、2（升级版总体，包括2.1 DQN+LSTM、2.2 修改状态、2.3 Actor-Critic+MLP）、3 最优版本。

分析基于方案文本，逐步优化焦点是提升性能（e.g., 减少gap：10城市<5%、50城市<20%），通过状态增强、序列记忆、Reward技巧等。基础版本效果差（gap>20-50%），逐步优化到近最优。

1.1 DQN版本（基础DQN，使用MLP）
状态设计：仅当前城市的one-hot编码，单一向量维度N（N=城市数，已访问位置为1，其余为0

Reward差异：无差异，使用基础Reward：每步r = -distance(current, next)；结束r += -distance(last, start)。总为负总路径长度，无shaping或调整。

缺点与改进点：

缺点（对比基础描述）：严重状态混叠（模型无法区分不同访问历史，导致重复访问、无效路径）；学习困难，效果差（10城市gap>20%、50城市>50%）；无记忆，依赖隐式序列学习但实际失效。

改进点：引入DQN（MLP作为Q网络，ε-greedy探索），比纯随机好，但仍基础。作为起点，易实现；若无效，可添加mask或shaping。优化到1.2通过切换算法框架。

1.2 Reinforce神经网络版本
状态设计：仅当前城市的one-hot编码，单一向量维度N（N=城市数，已访问位置为1，其余为0

Reward差异：无差异，与1.1相同（基础负距离，无shaping）。

缺点与改进点：
缺点（对比1.1）：继承状态混叠和学习困难；Reinforce（策略梯度）可能更不稳定（高方差），效果类似或略差于DQN，无off-policy优势。

改进点：从DQN切换到Reinforce，提供策略优化备选，但未解决核心混叠问题。作为基础变体，适合比较；下一步优化到#2通过引入序列或扩展来解决混叠。

2. 升级版（总体，包括2.1、2.2、2.3子版本）
这是一个中间升级阶段，焦点是解决混叠，通过LSTM序列、状态修改或Actor-Critic。整体从基础MLP/DQN转向更鲁棒框架。

2.1 DQN + LSTM版本
状态设计：输入为序列 of one-hot向量（状态设计：仅当前城市的one-hot编码，单一向量维度N（N=城市数，已访问位置为1，其余为0）（从起点到当前，形状[步数, N]），LSTM捕捉历史。示例：路径0→1→3（N=5），状态 = [[1,0,0,0,0], [0,1,0,0,0], [0,0,0,1,0]]；LSTM处理序列，输出隐藏状态区分ABCE vs ACBE。

Reward差异：有差异，引入shaping（potential基于剩余城市）+密度+结束调整，与基础负距离结合，帮助LSTM学习序列模式。

缺点与改进点：

缺点（对比1.2）：序列处理增加计算开销（长序列慢）；可能遗忘长历史（N=50时挑战）。

改进点：LSTM解决混叠（捕捉长期依赖），效果显著提升（10城市gap<5%、50<15%）；添加padding + truncated BPTT优化训练；共享LSTM高效。比前版更鲁棒，平滑升级。

2.2 修改状态版本
状态设计：在基础one-hot改成 浮点值表示顺序（当前=1，已访=1/访问步数，未访=0）。示例：路径A→B→C（假设城市0=A,1=B,2=C），状态向量后N维如[1/1 for A, 1/2 for B, 1/3 for C, 0 for others]；区分ABCE (B=1/2, C=1/3) vs ACBE (C=1/2, B=1/3)。

Reward差异：无明确差异，继承基础或升级版负距离（文本中未指定变化，但可结合shaping）。

缺点与改进点：

缺点（对比2.1）：顺序嵌入粗略，可能不足以完全解决长序列混叠；维度稍增（仍简单）。

改进点：直接嵌入顺序到one-hot，简单解决历史区分，而不需序列输入；比2.1计算更轻，适合MLP。优化焦点是状态增强，效果介于基础与LSTM间。

2.3 Actor-Critic + MLP版（切换到LSTM备选）
状态设计：序列 of one-hot（类似2.1，从起点到当前，形状[步数, N]，padding到N）

Reward差异：有差异，与2.1类似：基础负距离 + shaping（剩余城市潜力）+密度+结束调整，提升序列学习。

缺点与改进点：

缺点（对比2.2）：训练更慢（序列 + Actor-Critic复杂）；潜在梯度爆炸（用裁剪解决）。

改进点：从DQN切换到Actor-Critic + LSTM，结合策略/价值网络，更鲁棒；解决混叠，效果接近完整状态（10城市gap<5%、50<15%）。比前子版更全面，引入共享LSTM和熵损失稳定探索。

3. 最优版本（最终升级，DQN+MLP with 扩展状态 + Shaping）
状态设计：扩展one-hot + 距离向量 + 访问序号 + 进度（总维度~3N+1）。示例：对于N=5，当前在1，已访{0,1}，状态 = [one-hot当前: [0,1,0,0,0]] + [距离到所有: e.g., [0.5,0,0.3,0.7,0.2]] + [序号: e.g., [1,2,0,0,0]] + [进度: 0.4]；拼接为扁平向量，包含历史/距离。

Reward差异：有显著差异：基础r = -distance + Shaping（潜力函数: γ*(potential(current)-potential(next))，potential=剩余MST估计）+密度（+0.1 for 短边）+结束惩罚（基于总长偏差）。备选：-1/(distance+ε)。比前版更丰富，提供中间反馈。

缺点与改进点：

缺点（对比2.3）：维度更高（~3N），可能增加MLP复杂度；仍依赖MLP，无RNN高级记忆（但shaping补偿）。

改进点：综合扩展状态（嵌入历史/距离/顺序）+高级shaping，解决混叠/稀疏reward；效果最佳（10城市gap<5%、50<20%）；创新混合输入 + 可切换框架（DQN to Actor-Critic），比#2更稳健，针对大N优化探索。最终版逼近SOTA，通过shaping提升收敛2x。

总体对比与优化路径
逐步优化趋势：从1.1/1.2的基础（纯one-hot + 简单RL，混叠严重）→ #2的序列/状态增强（LSTM/嵌入解决历史区分，gap减至<15%）→ #3的最优（全面扩展 + shaping，gap<5-20%，接近原方案）。

共同改进：所有版本使用离散N动作、环境mask强制有效路径。Reward从纯负距离逐步添加shaping/密度，补偿状态不足。

性能提升：基础gap>20-50% → 升级<15% → 最优<5-20%。缺点从混叠/不稳转向计算开销，但通过padding/裁剪缓解。

建议：如果需要代码实现或进一步细节（如特定版本的PyTorch编辑），请提供文件路径或框架，我可以生成针对性代码块。





future work
# 1.
本方案针对之前实验设计中的Experiment 5，提供具体的实现细节：如何在合成数据集（10、20、30、50城市，随机2D坐标）上训练RL模型，然后在TSPLIB Berlin52（52城市）上进行零-shot测试或少-shot微调测试。核心挑战是城市数量（N）的变化导致网络输入维度不匹配（例如，visited_mask从大小50变为52），以及潜在的分布偏移（合成随机坐标 vs. 真实世界城市坐标）。方案设计强调学术价值，通过处理变N泛化来创新，揭示RL在TSP中的跨规模鲁棒性。这可支持论文点，如“强化学习在变规模TSP实例上的零-shot泛化”。预期结论：通过padding和嵌入技术，模型可实现<25%的最优性差距，证明状态丰富设计的可转移性。所有代码基于PyTorch，实现可重复。

问题分析和创新点

输入变化：原状态（如visited_mask [N]、current_city_onehot [N]、order_embedding [N]、distances_from_current [N]）依赖于N。训练时N固定（e.g., 50），测试时N=52导致维度 mismatch。

城市数量变化：合成N=10-50，Berlin52 N=52。直接测试会崩溃，除非模型设计为N-invariant。

创新：引入padding到固定最大N（e.g., 64），结合可变长度处理（e.g., masking in MLP or LSTM）。这是一种轻量级方法，避免重训全GNN，适合小规模TSP研究。学术价值：首次探讨RL-TSP中跨N泛化，填补从固定N到变N的空白，支持博士研究 on “Scalable RL for Variable-Sized Combinatorial Problems”。

修改的模型设计（支持变N）
为所有RL变体（QDN_visited, DQN_LSTM, Reinforce, ActorCritic, DQN_order, DQN_optimal）进行统一修改，使其支持变N：

固定输入维度：设置最大N_max=64（覆盖52+缓冲）。所有状态向量padding到[N_max]：

Visited_mask：原[N] → [N_max]，padding 0（表示虚拟城市，未使用）。

Current_city_onehot：原[N] → [N_max]，当前城市索引映射到0-(N-1)，padding 0。

Order_embedding：原[N] → [N_max]，使用预计算嵌入（e.g., 基于坐标排序），padding with 0 or mean value。

Distances_from_current：原[N] → [N_max]，计算实际N的距离，padding with large value (e.g., 1e6) 表示不可达。

网络调整：

MLP-based (QDN_visited, Reinforce, ActorCritic, DQN_order, DQN_optimal)：输入层改为固定大小 (e.g., 4*N_max if concatenating all)，添加masking层（e.g., multiply input by a mask vector to ignore padded parts）。输出层：Q值或policy logits 为[N_max]，但在动作选择时mask掉padding部分和已访问城市。

DQN_LSTM：序列输入[step, N_max]（每步padding到N_max），LSTM处理变长序列 via padding masks（PyTorch LSTM支持）。

动作空间：在测试时，动作仅限于0-(N-1)，使用mask禁止padding城市。Invalid动作（已访问或padding）设Q值=-inf。

奖励和环境：保持不变，但环境适配N：距离矩阵为[N x N]，padding到[N_max x N_max] with inf距离。

其他：探索策略（epsilon-greedy）不变；奖励塑造应用到实际N的城市。

这些修改使模型在训练时学习忽略padding，增强泛化。创新：这是一种“padded state embedding”技术，可发表为方法贡献。

训练方案（在合成数据上）

数据集：合成：为每个N=10,20,30,50生成100实例（随机2D坐标 in [0,1]x[0,1]）。总训练集：80% per N（混合所有N以促进跨N学习）。

训练过程：

混合N训练：单个模型处理所有N（10-50）。每个episode随机采样一个实例，padding到N_max=64。好处：模型内在学习变N。

分离N训练（备选，for ablation）：为每个N训练独立模型（e.g., N=50模型用于接近52的测试）。

参数：10,000 episodes；batch=32；Adam lr=0.001；gamma=0.99；5 seeds。

优化：在训练中，损失仅计算实际N的部分（mask loss on padding）。

基线训练：同样适配（GA和greedy不依赖N，直接运行）。

测试方案（在TSPLIB Berlin52上）

数据集准备：Berlin52（52城市，欧几里得距离，闭合路径）。无训练数据，仅测试。计算ground truth最优（使用Concorde或已知值：7542单位）。

零-shot测试（主要方案，评估纯泛化）：

加载合成训练的模型（混合N or N=50模型）。

输入处理：N=52，padding到64（添加12虚拟城市）。距离矩阵[N=52]扩展到[64x64]，padding距离=inf。

运行：模拟TSP episode，从起点开始，选择动作（mask已访问和padding）。收集路径长度。

重复：100次rollouts per模型（使用greedy decoding，即选max Q/policy），计算平均/最佳路径长度。

指标：最优性差距 = (avg RL长度 - 7542) / 7542 * 100%；成功率（完成无重访路径的比例）；泛化差距 = Berlin52差距 - 合成50城市差距。

少-shot微调测试（可选，for Experiment 5的fine-tune部分）：

从零-shot模型开始，在Berlin52单一实例上微调1000 episodes（小lr=0.0001，避免灾难性遗忘）。

输入：直接N=52，padding到64。

目的：量化微调改善（预期差距减少10-15%）。

处理分布偏移：合成是均匀随机，Berlin52是真实城市。监控如果差距>30%，分析原因（e.g., 坐标尺度差异：归一化所有坐标到[0,1] in 预处理）。

预期结果和分析

性能预期：混合N训练的DQN_optimal差距<20% (零-shot)，LSTM<15%（序列帮助泛化）。分离N=50模型差距<25%。

消融：比较padding vs. 无padding（预期padding改善5-10%）；混合N vs. 分离N（混合更好泛化）。

结论抽取：如果差距低，证明padded状态有效，支持“RL-TSP的跨规模转移”论文。如果高，揭示局限（如需要GNN），引导未来工作。

统计：t-test比较变体；可视化路径（e.g., matplotlib plot Berlin52路径）。

实现细节和代码框架

环境：自定义Gym-like TSP env，支持变N和padding。

伪代码示例：
def state_padding(state, N, N_max):
padded = np.pad(state, (0, N_max - N), mode='constant', constant_values=0)
return padded

在forward中：
input = concat([padded_onehot, padded_mask, padded_embedding, padded_dists])
q_values = mlp(input) # size [N_max]
masked_q = q_values + invalid_mask * -1e9 # mask padding and visited

风险缓解：如果padding引入噪声，添加regularization（e.g., loss penalty on padding Q值）。计算成本：测试快速（<1小时/模型）。

此方案确保实验严谨，具有创新性，可直接集成到整体研究中。


# 2.
#
实验四：奖励函数设计对比

目标：验证改进奖励函数的有效性
实验组：基础奖励 vs 改进奖励（各算法分别测试）
评价指标：

收敛速度提升程度

最终解质量改善

训练过程稳定性

# 3.
实验二：记忆机制作用分析

目标：分析LSTM记忆机制在TSP求解中的作用
实验组：DQN_visited（MLP）vs DQN_LSTM vs DQN_Attention
控制变量：状态输入格式、奖励函数
评价指标：

路径质量对比

避免子回路的能力

对历史决策的依赖性分析

具体实验步骤：

设计特殊测试案例，包含明显的局部最优陷阱

分析各算法在这些案例上的表现

可视化注意力权重或LSTM隐状态变化

统计算法产生无效回路的频率

预期结论：LSTM在大规模问题上表现更好，Attention机制能更有效地利用全局信息




# 4.
算法名称：Hierarchical_DQN
上层策略：选择下一个要访问的区域
下层策略：在选定区域内选择具体城市
创新点：将TSP分解为层次化决策问题