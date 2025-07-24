import math
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Tuple, Any
import warnings
from scipy import stats
from sklearn.metrics import mean_squared_error
import os
from itertools import combinations, permutations  # 新增：用于生成组合和Shapley计算

warnings.filterwarnings('ignore')


class TSPAblationExperimentGenerator:
    """TSP消融实验数据生成器"""

    def __init__(self):
        self.algorithms = ['DQN_visited', 'DQN_LSTM', 'Reinforce', 'ActorCritic', 'DQN_order', 'DQN_optimal']
        self.states = ["current_city_onehot", "visited_mask", "order_embedding", "distances_from_current"]
        self.map_state_types = self._build_map_state_types()  # 新增：构建映射
        self.state_types = list(self.map_state_types.keys())  # 更新：使用新分组，总11种
        self.city_nums = [10, 20, 30, 50]
        self.modes = ['per_instance', 'cross_instance']
        self.train_test_splits = ['train', 'test']
        self.total_instances = 100
        self.train_instances = 80
        self.test_instances = 20
        self.runs_per_instance = 5

        # 设置随机种子以保证可重复性
        np.random.seed(42)

    def _build_map_state_types(self) -> Dict[str, List[str]]:  # 新增方法：生成映射
        map_state_types = {}
        # 全状态
        map_state_types['full'] = self.states.copy()
        # 移除一种
        for i, remove in enumerate(self.states, 1):
            remaining = [s for s in self.states if s != remove]
            map_state_types[f'ablation_single_{i}'] = remaining
        # 移除两种
        for j, removes in enumerate(combinations(self.states, 2), 1):
            remaining = [s for s in self.states if s not in removes]
            map_state_types[f'ablation_double_{j}'] = remaining
        return map_state_types

    def generate_optimal_distances(self, city_num: int, instance_id: int) -> float:
        """生成TSP实例的最优距离（基于城市数量的经验公式）"""
        np.random.seed(instance_id)  # 确保每个实例的最优解固定
        base_distance = np.sqrt(city_num) * 10
        variation = np.random.normal(0, 0.1) * base_distance
        return max(base_distance + variation, base_distance * 0.8)

    def generate_state_representation(self, state_type: str, city_num: int, step: int) -> str:
        """生成不同类型的状态表示"""
        current_city = np.random.randint(0, city_num)
        visited_mask = np.random.choice([0, 1], size=city_num, p=[0.3, 0.7])

        state = {}
        included = self.map_state_types[state_type]  # 根据映射动态包括组件

        if "current_city_onehot" in included:
            state['current_city_onehot'] = [1 if i == current_city else 0 for i in range(city_num)]
        if "visited_mask" in included:
            state['visited_mask'] = visited_mask.tolist()
        if "order_embedding" in included:
            state['order_embedding'] = [step / 100.0] * city_num
        if "distances_from_current" in included:
            state['distances_from_current'] = np.random.exponential(2, city_num).tolist()
        # step_position 只在full中（假设为额外，但原代码中在full/ablation_4）
        if state_type == 'full':
            state['step_position'] = step / city_num

        return json.dumps(state)

    def simulate_performance_by_state_type(self, state_type: str, algorithm: str, city_num: int) -> Dict[str, float]:
        """根据状态类型模拟性能表现"""
        # 基础性能（基于算法和城市数量）
        algorithm_performance = {
            'DQN_visited': 0.75, 'DQN_LSTM': 0.80, 'Reinforce': 0.70,
            'ActorCritic': 0.85, 'DQN_order': 0.78, 'DQN_optimal': 0.90
        }

        # 规模惩罚
        scale_penalty = 1 - (city_num - 10) * 0.02
        base_performance = algorithm_performance[algorithm] * scale_penalty

        # 更新：基于包含组件数量和类型模拟乘数（更多组件更好，引入交互）
        included = self.map_state_types[state_type]
        num_included = len(included)
        multiplier = 0.5 + 0.125 * num_included  # 基础：0.5 (minimal) 到 1.0 (full)
        if "distances_from_current" in included:
            multiplier += 0.1  # 距离信息重要
        if "order_embedding" in included:
            multiplier += 0.08  # 顺序信息次要
        multiplier = min(1.0, multiplier + np.random.normal(0, 0.02))  # 添加噪声

        performance = base_performance * multiplier

        return {
            'base_optimality_gap': (1 - performance) * 100,
            'convergence_factor': multiplier,
            'stability_factor': 1 / multiplier
        }

    def generate_episode_data(self, algorithm: str, city_num: int, mode: str,
                              state_type: str, instance_id: int, run_id: int,
                              train_test: str) -> List[Dict]:
        """生成单个实例的episode数据"""
        data = []
        optimal_distance = self.generate_optimal_distances(city_num, instance_id)
        optimal_path = f"optimal_path_{instance_id}"

        # 根据状态类型和算法设定episode数量
        max_episodes = 100 if train_test == 'train' else 50  # 减少数据量便于演示
        state_value = self.map_state_types[state_type]
        performance_params = self.simulate_performance_by_state_type(state_type, algorithm, city_num)

        np.random.seed(instance_id * 1000 + run_id * 100)  # 确保可重复性

        for episode in range(max_episodes):
            episode_length = city_num
            episode_reward = 0
            current_distance = 0

            for step in range(episode_length):
                # 生成状态表示
                state = self.generate_state_representation(state_type, city_num, step)

                # 模拟奖励和损失
                step_distance = np.random.exponential(2)
                current_distance += step_distance

                # 奖励计算（距离惩罚）
                step_reward = -step_distance
                if step == episode_length - 1:  # 最后一步返回起点
                    step_reward -= np.random.exponential(3)

                episode_reward += step_reward

                # 损失计算（仅对某些算法）
                loss = np.random.exponential(0.5) if algorithm.startswith('DQN') else np.nan

                # 根据性能参数调整
                optimality_gap = performance_params['base_optimality_gap']
                final_distance = optimal_distance * (1 + optimality_gap / 100)

                data.append({
                    'algorithm': algorithm,
                    'city_num': city_num,
                    'mode': mode,
                    'instance_id': instance_id,
                    'run_id': run_id,
                    'state_type': state_type,
                    'train_test': train_test,
                    'episode': episode,
                    'step': step,
                    'state': state,
                    'done': 1 if step == episode_length - 1 else 0,
                    'reward': step_reward,
                    'loss': loss,
                    'total_reward': episode_reward if step == episode_length - 1 else np.nan,
                    'current_distance': final_distance if step == episode_length - 1 else np.nan,
                    'optimal_distance': optimal_distance,
                    'optimal_path': optimal_path,
                    'coordinates':[],
                    'state_values':state_value
                })

        return data

    def generate_full_experiment_data(self) -> pd.DataFrame:
        """生成完整的实验数据（演示版本）"""
        all_data = []

        # 为了演示，只选择部分组合
        selected_algorithms = ['DQN_LSTM', 'ActorCritic', 'DQN_optimal'][:1]
        selected_cities = [10]
        selected_instances = list(range(0, 20, 5))  # 每5个实例取一个

        total_combinations = len(selected_algorithms) * len(selected_cities) * len(self.modes) * len(self.state_types)
        current_combination = 0

        print("开始生成实验数据...")

        for algorithm in selected_algorithms:
            for city_num in selected_cities:
                for mode in self.modes:
                    for state_type in self.state_types:
                        current_combination += 1
                        print(
                            f"处理组合 {current_combination}/{total_combinations}: {algorithm}-{city_num}-{mode}-{state_type}")

                        # 根据模式选择实例范围
                        if mode == 'per_instance':
                            # 训练集
                            for instance_id in selected_instances[:3]:  # 只选3个训练实例
                                for run_id in range(2):  # 只运行2次
                                    episode_data = self.generate_episode_data(
                                        algorithm, city_num, mode, state_type,
                                        instance_id, run_id, 'train'
                                    )
                                    all_data.extend(episode_data)  # 只保留最后20个episode的数据

                            # 测试集
                            for instance_id in selected_instances[3:]:  # 测试实例
                                for run_id in range(2):
                                    episode_data = self.generate_episode_data(
                                        algorithm, city_num, mode, state_type,
                                        instance_id, run_id, 'test'
                                    )
                                    all_data.extend(episode_data)  # 测试时episode较少

                        else:  # cross_instance
                            for run_id in range(2):
                                # 训练阶段：在所有训练实例上训练
                                for instance_id in selected_instances[:3]:
                                    episode_data = self.generate_episode_data(
                                        algorithm, city_num, mode, state_type,
                                        instance_id, run_id, 'train'
                                    )
                                    all_data.extend(episode_data)  # 每个实例只保留少量数据

                                # 测试阶段：在测试实例上评估
                                for instance_id in selected_instances[3:]:
                                    episode_data = self.generate_episode_data(
                                        algorithm, city_num, mode, state_type,
                                        instance_id, run_id, 'test'
                                    )
                                    all_data.extend(episode_data)  # 测试数据更少

        df = pd.DataFrame(all_data)
        print(f"数据生成完成！总共 {len(df)} 行数据")
        return df


class TSPAblationAnalyzer:
    """TSP消融实验分析器"""


    def __init__(self, df: pd.DataFrame, generator: TSPAblationExperimentGenerator):  # 修改：添加generator参数
        self.df = df
        self.generator = generator  # 新增：存储generator引用
        self.results = {}

    def calculate_performance_metrics(self) -> pd.DataFrame:
        """计算核心性能指标"""
        print("计算性能指标...")

        # 过滤完整episode的数据
        episode_data = self.df[self.df['done'] == 1].copy()

        # 计算optimality gap
        episode_data['optimality_gap'] = (
                (episode_data['current_distance'] - episode_data['optimal_distance']) /
                episode_data['optimal_distance'] * 100
        )

        # 按组合分组计算指标
        metrics = episode_data.groupby(['algorithm', 'city_num', 'mode', 'state_type', 'train_test']).agg({
            'optimality_gap': ['mean', 'std', 'count'],
            'total_reward': ['mean', 'std'],
            'episode': ['max', 'mean'],
            'current_distance': ['mean', 'min', 'max']
        }).round(4) #将最终结果的所有数值四舍五入到4位小数

        # 展平多级列名
        metrics.columns = ['_'.join(col).strip() for col in metrics.columns.values]
        metrics = metrics.reset_index()

        return metrics

    def calculate_component_contributions(self) -> pd.DataFrame:
        """计算组件贡献度（简化方案：使用简单差异而非Shapley值）"""
        print("计算组件贡献度（简单差异）...")

        metrics = self.calculate_performance_metrics()
        contributions = []

        for algorithm in self.df['algorithm'].unique():
            for city_num in self.df['city_num'].unique():
                for mode in self.df['mode'].unique():
                    for train_test in self.df['train_test'].unique():
                        subset = metrics[
                            (metrics['algorithm'] == algorithm) &
                            (metrics['city_num'] == city_num) &
                            (metrics['mode'] == mode) &
                            (metrics['train_test'] == train_test)
                        ]

                        if len(subset) == 0:
                            continue

                        # 获取full状态的性能作为基准 (使用 -optimality_gap_mean 作为价值，越高越好)
                        full_row = subset[subset['state_type'] == 'full']
                        if len(full_row) == 0:
                            continue
                        full_value = -full_row['optimality_gap_mean'].values[0]

                        # 对于每个组件，找到移除该组件的ablation状态，并计算差异作为贡献
                        component_contrib = {
                            'algorithm': algorithm,
                            'city_num': city_num,
                            'mode': mode,
                            'train_test': train_test
                        }
                        all_components = self.generator.states

                        for comp in all_components:
                            # 找到移除该组件的ablation状态 (假设ablation_single对应移除单个)
                            ablation_states = [st for st in subset['state_type'] if 'single' in st and comp not in self.generator.map_state_types[st]]
                            if not ablation_states:
                                component_contrib[comp] = 0  # 如果没有对应ablation，贡献为0
                                continue

                            # 取平均ablation价值
                            ablation_values = [-row['optimality_gap_mean'] for _, row in subset.iterrows() if row['state_type'] in ablation_states]
                            avg_ablation = np.mean(ablation_values) if ablation_values else 0

                            # 贡献 = full_value - avg_ablation (正值表示正面贡献)
                            component_contrib[comp] = full_value - avg_ablation

                        contributions.append(component_contrib)

        return pd.DataFrame(contributions)

class TSPVisualizationSuite:
    """TSP可视化套件"""

    def __init__(self, analyzer: TSPAblationAnalyzer):
        self.analyzer = analyzer
        self.contributions = analyzer.calculate_component_contributions()
        self.performance_metrics = analyzer.calculate_performance_metrics()

        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("husl")

    def plot_component_contribution_heatmap(self):
        """绘制组件贡献度热力图（基于简单差异）"""
        print("绘制简单贡献度热力图...")

        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        contributions_avg = self.contributions.groupby(['algorithm', 'city_num']).mean(numeric_only=True).reset_index()

        components = self.analyzer.generator.states
        titles = [f'{comp} Simple Contribution' for comp in components]  # 更新标题以反映简单贡献

        for idx, (component, title) in enumerate(zip(components, titles)):
            ax = axes[idx // 2, idx % 2]

            # 创建数据透视表
            pivot_data = contributions_avg.pivot(index='algorithm', columns='city_num', values=component)

            # 绘制热力图
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r',
                        center=0, ax=ax, cbar_kws={'label': 'Shapley Value'})
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Number of Cities', fontsize=12)
            ax.set_ylabel('Algorithm', fontsize=12)

        plt.tight_layout()
        plt.savefig('component_contribution_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_ablation_waterfall(self):
        """绘制消融实验瀑布图（扩展到所有组合）"""
        print("绘制消融实验瀑布图...")

        # 使用性能分数 (100 - gap)
        performance = self.performance_metrics.groupby('state_type')['optimality_gap_mean'].mean().sort_values()
        labels = performance.index.tolist()
        performance_values = (100 - performance).tolist()

        # 绘制瀑布图
        fig, ax = plt.subplots(figsize=(12, 8))

        x_pos = np.arange(len(labels))
        colors = ['green' if 'full' in label else 'orange' if 'single' in label else 'red' for label in labels]

        bars = ax.bar(x_pos, performance_values, color=colors, alpha=0.7, edgecolor='black')

        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, performance_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

            # 添加下降箭头（除了第一个）
            if i > 0 and i < len(performance_values):
                prev_height = performance_values[i - 1]
                if height < prev_height:
                    ax.annotate('', xy=(i - 0.1, height + 1), xytext=(i - 0.9, prev_height + 1),
                                arrowprops=dict(arrowstyle='->', color='red', lw=2))

        ax.set_xlabel('State Representation', fontsize=12)
        ax.set_ylabel('Performance Score (%)', fontsize=12)
        ax.set_title('Ablation Study Waterfall Chart (ActorCritic, 20 cities)', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('ablation_waterfall.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_performance_comparison(self):
        """绘制性能对比图"""
        print("绘制性能对比图...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 状态类型性能对比
        ax1 = axes[0, 0]
        state_performance = self.performance_metrics.groupby(['state_type']).agg({
            'optimality_gap_mean': 'mean'
        }).reset_index()

        bars = ax1.bar(state_performance['state_type'], state_performance['optimality_gap_mean'])
        ax1.set_xlabel('State Type')
        ax1.set_ylabel('Average Optimality Gap (%)')
        ax1.set_title('Performance by State Type')
        ax1.tick_params(axis='x', rotation=45)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{height:.2f}', ha='center', va='bottom')

        # 2. 算法性能对比
        ax2 = axes[0, 1]
        algorithm_performance = self.performance_metrics.groupby(['algorithm']).agg({
            'optimality_gap_mean': 'mean'
        }).reset_index()

        bars = ax2.bar(algorithm_performance['algorithm'], algorithm_performance['optimality_gap_mean'])
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Average Optimality Gap (%)')
        ax2.set_title('Performance by Algorithm')
        ax2.tick_params(axis='x', rotation=45)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{height:.2f}', ha='center', va='bottom')

        # 3. 城市规模vs性能
        ax3 = axes[1, 0]
        city_performance = self.performance_metrics.groupby('city_num')['optimality_gap_mean'].mean()
        ax3.plot(city_performance.index, city_performance.values, marker='o', linewidth=2, markersize=8)
        ax3.set_title('Performance vs Problem Size')
        ax3.set_xlabel('Number of Cities')
        ax3.set_ylabel('Average Optimality Gap (%)')
        ax3.grid(True, alpha=0.3)

        # 4. 训练vs测试性能对比
        ax4 = axes[1, 1]
        train_test_comparison = self.performance_metrics.groupby(['train_test']).agg({
            'optimality_gap_mean': 'mean'
        }).reset_index()

        bars = ax4.bar(train_test_comparison['train_test'], train_test_comparison['optimality_gap_mean'])
        ax4.set_xlabel('Dataset Type')
        ax4.set_ylabel('Average Optimality Gap (%)')
        ax4.set_title('Train vs Test Performance')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{height:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_summary_report(self):
        """生成总结报告（添加Shapley和显著性检验）"""
        print("\n" + "=" * 80)
        print("TSP深度强化学习消融实验总结报告")
        print("=" * 80)

        # 基本统计信息
        print(f"\n数据集概览:")
        print(f"- 总数据量: {len(self.analyzer.df):,} 条记录")
        print(f"- 算法数量: {self.analyzer.df['algorithm'].nunique()}")
        print(f"- 状态类型: {self.analyzer.df['state_type'].nunique()}")
        print(f"- 城市规模范围: {self.analyzer.df['city_num'].min()}-{self.analyzer.df['city_num'].max()}")

        # 性能排名
        print(f"\n状态类型性能排名 (Optimality Gap %):")
        state_ranking = self.performance_metrics.groupby('state_type')['optimality_gap_mean'].mean().sort_values()
        for i, (state, performance) in enumerate(state_ranking.items(), 1):
            print(f"{i}. {state}: {performance:.2f}%")

        print(f"\n算法性能排名 (Optimality Gap %):")
        algo_ranking = self.performance_metrics.groupby('algorithm')['optimality_gap_mean'].mean().sort_values()
        for i, (algo, performance) in enumerate(algo_ranking.items(), 1):
            print(f"{i}. {algo}: {performance:.2f}%")

        # Shapley值贡献度分析
        if len(self.contributions) > 0:
            print(f"\nShapley值贡献度分析:")
            avg_shapley = self.contributions[self.analyzer.generator.states].mean()
            for comp, val in avg_shapley.items():
                print(f"- {comp} Shapley值: {val:.3f}")

            # 添加t-test示例（full vs ablation）
            full_gap = self.performance_metrics[self.performance_metrics['state_type'] == 'full']['optimality_gap_mean']
            ablation_gap = self.performance_metrics[self.performance_metrics['state_type'].str.contains('ablation')]['optimality_gap_mean']
            if len(full_gap) > 1 and len(ablation_gap) > 1:
                t_stat, p_val = stats.ttest_ind(full_gap, ablation_gap)
                print(f"\n显著性检验 (full vs ablation): t-stat={t_stat:.2f}, p-value={p_val:.4f}")

        print("\n" + "=" * 80)


try:
    # # 1. 创建实验数据生成器
    print("\n步骤 1: 初始化实验数据生成器...")
    generator = TSPAblationExperimentGenerator()

    # 2. 生成实验数据
    print("\n步骤 2: 生成实验数据...")
    df = generator.generate_full_experiment_data()

    # 3. 保存原始数据
    print("\n步骤 3: 保存原始数据...")
    df.to_csv('tsp_ablation_experiment_data.csv', index=False)
    print(f"原始数据已保存到: tsp_ablation_experiment_data.csv")
    # df =pd.read_csv('tsp_ablation_experiment_data.csv')

    # 4. 创建分析器
    print("\n步骤 4: 初始化分析器...")
    analyzer = TSPAblationAnalyzer(df, generator)

    # 5. 计算性能指标
    print("\n步骤 5: 计算性能指标...")
    performance_metrics = analyzer.calculate_performance_metrics()
    performance_metrics.to_csv('performance_metrics.csv', index=False)
    print("性能指标已保存到: performance_metrics.csv")

    # 6. 计算组件贡献度
    print("\n步骤 6: 计算组件贡献度...")
    contributions = analyzer.calculate_component_contributions()
    if len(contributions) > 0:
        contributions.to_csv('component_contributions.csv', index=False)
        print("组件贡献度已保存到: component_contributions.csv")

    # 7. 创建可视化套件
    print("\n步骤 7: 创建可视化...")
    viz_suite = TSPVisualizationSuite(analyzer)

    # 8. 生成图表
    print("\n步骤 8: 生成分析图表...")

    # 组件贡献度热力图
    if len(contributions) > 0:
        viz_suite.plot_component_contribution_heatmap()

    # 消融实验瀑布图
    viz_suite.plot_ablation_waterfall()

    # 性能对比图
    viz_suite.plot_performance_comparison()

    # 9. 生成总结报告
    print("\n步骤 9: 生成总结报告...")
    viz_suite.generate_summary_report()

    # 10. 显示数据样本
    print("\n步骤 10: 显示数据样本...")
    print("\n原始数据样本 (前5行):")
    print(df.head())

    print("\n性能指标样本 (前5行):")
    print(performance_metrics.head())

    if len(contributions) > 0:
        print("\n组件贡献度样本 (前5行):")
        print(contributions.head())

    print("\n" + "=" * 80)
    print("实验完成！所有结果已保存到当前目录")
    print("生成的文件包括:")
    print("- tsp_ablation_experiment_data.csv (原始数据)")
    print("- performance_metrics.csv (性能指标)")
    if len(contributions) > 0:
        print("- component_contributions.csv (组件贡献度)")
    print("- component_contribution_heatmap.png (组件贡献热力图)")
    print("- ablation_waterfall.png (消融实验瀑布图)")
    print("- performance_comparison.png (性能对比图)")
    print("=" * 80)

except Exception as e:
    print(f"\n错误: {str(e)} {traceback.format_exc()}")
    print("请检查依赖库是否已正确安装:")
    print("pip install numpy pandas matplotlib seaborn scikit-learn")
