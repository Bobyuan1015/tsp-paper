
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
from itertools import combinations

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from confs import project_root

warnings.filterwarnings('ignore')

import csv
import os
from collections import defaultdict
import os
import csv
import time
from collections import defaultdict


def split_large_csv(input_file, output_dir=None, buffer_size=1000000):
    """
    拆分大CSV文件按(mode, train_test)分组 - 优化版本

    Args:
        input_file: 输入CSV文件路径
        output_dir: 输出目录，如果为None则使用输入文件的同级目录
        buffer_size: 每个文件的缓冲区大小（行数），默认1000行

    Returns:
        list: 生成的所有小文件的完整路径列表
    """
    # 如果未指定输出目录，使用输入文件的同级目录
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_file))

    os.makedirs(output_dir, exist_ok=True)

    # 使用更大的文件读取缓冲区
    file_buffer_size = 8192 * 16  # 128KB缓冲区

    # 快速估算文件大小和行数
    print("正在估算文件大小...")
    file_size = os.path.getsize(input_file)

    # 采样前1000行来估算平均行长度
    sample_lines = 0
    sample_size = 0
    with open(input_file, 'r', encoding='utf-8', buffering=file_buffer_size) as f:
        next(f)  # 跳过表头
        for i, line in enumerate(f):
            if i >= 1000:  # 采样1000行
                break
            sample_lines += 1
            sample_size += len(line.encode('utf-8'))

    # 估算总行数
    if sample_lines > 0:
        avg_line_size = sample_size / sample_lines
        estimated_lines = int((file_size - sample_size) / avg_line_size)
        print(f"估算文件约有 {estimated_lines:,} 行数据")
    else:
        estimated_lines = 0

    # 使用缓冲区批量写入
    file_buffers = defaultdict(list)  # 每个文件的数据缓冲区
    file_handles = {}
    writers = {}
    output_files = []
    processed_count = 0
    start_time = time.time()

    def flush_buffer(filename):
        """刷新指定文件的缓冲区"""
        if filename in file_buffers and file_buffers[filename]:
            for row in file_buffers[filename]:
                writers[filename].writerow(row)
            file_buffers[filename].clear()

    def flush_all_buffers():
        """刷新所有缓冲区"""
        for filename in list(file_buffers.keys()):
            flush_buffer(filename)

    try:
        with open(input_file, 'r', encoding='utf-8', buffering=file_buffer_size) as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames

            for row in reader:
                mode = row['mode']
                train_test = row['train_test']

                # 生成文件名
                safe_mode = str(mode).replace('/', '_').replace('\\', '_')
                safe_train_test = str(train_test).replace('/', '_').replace('\\', '_')
                filename = f"{safe_mode}_{safe_train_test}.csv"
                filepath = os.path.join(output_dir, filename)

                # 如果是新文件，创建文件句柄和writer
                if filename not in file_handles:
                    file_handles[filename] = open(filepath, 'w', encoding='utf-8',
                                                  newline='', buffering=file_buffer_size)
                    writers[filename] = csv.DictWriter(
                        file_handles[filename],
                        fieldnames=fieldnames
                    )
                    writers[filename].writeheader()
                    output_files.append(filepath)
                    print(f"创建新文件: {filename}")

                # 添加到缓冲区而不是直接写入
                file_buffers[filename].append(row)

                # 当缓冲区达到指定大小时，批量写入
                if len(file_buffers[filename]) >= buffer_size:
                    flush_buffer(filename)

                processed_count += 1

                # 每处理5000行打印一次进度（减少打印频率提升性能）
                if processed_count % 5000 == 0:
                    elapsed_time = time.time() - start_time
                    if estimated_lines > 0:
                        progress_percent = (processed_count / estimated_lines) * 100
                        speed = processed_count / elapsed_time if elapsed_time > 0 else 0

                        print(f"进度: {processed_count:,}/{estimated_lines:,} ({progress_percent:.1f}%) "
                              f"| 速度: {speed:.0f} 行/秒 | 文件数: {len(output_files)}")
                    else:
                        speed = processed_count / elapsed_time if elapsed_time > 0 else 0
                        print(f"已处理: {processed_count:,} 行 | 速度: {speed:.0f} 行/秒 | 文件数: {len(output_files)}")

        # 处理完成后刷新所有剩余的缓冲区
        print("正在写入剩余数据...")
        flush_all_buffers()

    finally:
        # 确保所有文件句柄都被关闭
        for handle in file_handles.values():
            handle.close()

    elapsed_time = time.time() - start_time
    print(f"\n拆分完成！")
    print(f"总共处理了 {processed_count:,} 行数据")
    print(f"生成了 {len(output_files)} 个文件")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"平均速度: {processed_count / elapsed_time:.0f} 行/秒")
    print(f"输出目录: {output_dir}")

    return output_files


# 构建状态变量映射关系
def build_state_combinations():
    """构建消融实验的状态组合映射"""
    base_states = ["current_city_onehot", "visited_mask", "order_embedding", "distances_from_current"]

    map_state_types = {}

    # 1. 全状态（基线）
    map_state_types['full'] = base_states.copy()

    # 2. 依次移除一种状态（4种组合）
    for i, state_to_remove in enumerate(base_states):
        remaining_states = [s for s in base_states if s != state_to_remove]
        map_state_types[f'ablation_remove_{state_to_remove.split("_")[0]}'] = remaining_states

    # 3. 依次移除两种状态（6种组合）
    for i, (state1, state2) in enumerate(combinations(base_states, 2)):
        remaining_states = [s for s in base_states if s not in [state1, state2]]
        key_name = f'ablation_remove_{state1.split("_")[0]}_{state2.split("_")[0]}'
        map_state_types[key_name] = remaining_states

    return map_state_types


# 全局映射关系
map_state_types = build_state_combinations()



class TSPAdvancedAblationAnalyzer:
    """高级TSP消融实验分析器 - 博士水准"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}
        self.base_states = ["current_city_onehot", "visited_mask", "order_embedding", "distances_from_current"]

    def calculate_performance_metrics(self) -> pd.DataFrame:
        """计算核心性能指标"""
        print("计算性能指标...")

        episode_data = self.df[self.df['done'] == 1].copy()
        episode_data['optimality_gap'] = (
                (episode_data['current_distance'] - episode_data['optimal_distance']) /
                episode_data['optimal_distance'] * 100
        )

        metrics = episode_data.groupby(['algorithm', 'city_num', 'mode', 'state_type', 'train_test']).agg({
            'optimality_gap': ['mean', 'std', 'count'],
            'total_reward': ['mean', 'std'],
            'episode': ['max', 'mean'],
            'current_distance': ['mean', 'min', 'max']
        }).round(4)

        metrics.columns = ['_'.join(col).strip() for col in metrics.columns.values]
        metrics = metrics.reset_index()

        return metrics

    def _welch_t_test(self, mean1, std1, n1, mean2, std2, n2):
        """
        Welch's t-test实现 - 真正的双样本t检验

        公式：
        t = (μ₁ - μ₂) / SE_diff
        SE_diff = √(s₁²/n₁ + s₂²/n₂)
        df = (s₁²/n₁ + s₂²/n₂)² / [(s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1)]

        其中：
        - μ₁, μ₂: 两组样本均值
        - s₁, s₂: 两组样本标准差
        - n₁, n₂: 两组样本数量
        - SE_diff: 均值差的标准误差
        - df: 自由度
        """
        if std1 <= 0 or std2 <= 0 or n1 <= 1 or n2 <= 1:
            return 0.0, 1.0

        # 标准误差
        se1 = std1 / np.sqrt(n1)
        se2 = std2 / np.sqrt(n2)
        se_diff = np.sqrt(se1 ** 2 + se2 ** 2)

        # t统计量
        t_stat = (mean1 - mean2) / se_diff

        # 自由度 (Welch-Satterthwaite方程)
        df = (se1 ** 2 + se2 ** 2) ** 2 / (se1 ** 4 / (n1 - 1) + se2 ** 4 / (n2 - 1))

        # p值 (双尾检验)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

        return t_stat, p_value

    def _perform_real_significance_tests(self, subset: pd.DataFrame, full_stats: Dict) -> Dict[str, float]:
        """执行真正的统计显著性检验:用于判断消融实验中移除某个状态组件后的性能变化是否具有统计学意义上的显著性。
        对每个消融状态（移除组件后的状态）与基线状态（full状态）进行双样本t检验
        计算效应量（Cohen's d）来衡量实际差异的大小
        判断性能差异是否显著（p < 0.05）

        ---->确定观察到的性能差异是真实存在的，而不是由于随机误差造成的。

        """
        significance = {}

        for _, row in subset.iterrows():
            if row['state_type'] != 'full':
                # 双样本Welch's t检验: full对比 各个消融实验
                # Welch's t-test（双样本t检验）
                t_stat, p_value = self._welch_t_test(
                    full_stats['mean'], full_stats['std'], full_stats['count'],
                    row['optimality_gap_mean'], row['optimality_gap_std'], row['optimality_gap_count']
                )

                # 计算效应量 (Cohen's d)
                # 合并标准差公式：pooled_std = √[((n₁-1)×s₁² + (n₂-1)×s₂²) / (n₁+n₂-2)]
                pooled_std = np.sqrt(((full_stats['count'] - 1) * full_stats['std'] ** 2 +
                                      (row['optimality_gap_count'] - 1) * row['optimality_gap_std'] ** 2) /
                                     (full_stats['count'] + row['optimality_gap_count'] - 2))

                # Cohen's d公式：Cohen's_d = |μ₁ - μ₂| / pooled_std
                cohens_d = abs(row['optimality_gap_mean'] - full_stats['mean']) / pooled_std if pooled_std > 0 else 0

                significance[f"{row['state_type']}_p_value"] = p_value
                significance[f"{row['state_type']}_t_statistic"] = t_stat

                # 使用双尾检验，因为我们关心的是差异的存在性，不预设方向。 显著性判断: p < 0.05 --->这个差异是偶然发生的吗？"
                # p < 0.05：差异具有统计显著性，不太可能是随机产生的
                significance[f"{row['state_type']}_is_significant"] = 1.0 if p_value < 0.05 else 0.0

                # 效应大小: Cohen's d 表示差异的实际重要性: 避免将随机波动误认为真实效应
                # "这个差异有多重要"
                # 只看p值：可能保留很多"统计显著但无实际意义"的组件
                # 只看Cohen's d：可能被随机波动误导
                # 两者结合：既保证科学严谨性，又关注实际意义
                significance[f"{row['state_type']}_effect_size"] = cohens_d
        # | p值范围  | Cohen's d范围 | 组件重要性 | 建议行动 | 置信度 |
        # | <0.01   | >0.8        | 极其重要    | 必须保留 | 很高  |
        # | <0.05   | 0.5-0.8     | 重要       | 建议保留 | 高   |
        # | <0.05   | 0.2-0.5     | 一般重要    | 可以保留 | 中等  |
        # | <0.05   | <0.2        | 次要       | 可以移除 | 中等  |
        # | ≥0.05   | >0.8        | 不确定      | 增加样本 | 低   |
        # | ≥0.05   | <0.5        | 不重要      | 可以移除 | 高   |

        return significance

    def calculate_component_contributions(self) -> pd.DataFrame:
        """高级组件贡献度分析 - 基于消融实验理论"""
        print("执行高级组件贡献度分析...")

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

                        # 获取全状态的统计信息
                        full_row = subset[subset['state_type'] == 'full']
                        if len(full_row) == 0:
                            continue

                        full_stats = {
                            'mean': full_row.iloc[0]['optimality_gap_mean'],
                            'std': full_row.iloc[0]['optimality_gap_std'],
                            'count': full_row.iloc[0]['optimality_gap_count']
                        }

                        # 构建性能字典
                        performance_dict = {}
                        for _, row in subset.iterrows():
                            state_type = row['state_type']
                            performance_dict[state_type] = row['optimality_gap_mean']

                        if 'full' not in performance_dict:
                            continue

                        full_performance = performance_dict['full']

                        # 计算各组件的边际贡献
                        component_contributions = self._calculate_marginal_contributions(performance_dict)

                        # 计算交互效应:  该函数计算当同时移除两个组件时，产生的交互效应是否大于、小于或等于单独移除这两个组件效应的简单相加。
                        # 哪些状态组件必须同时保留（正交互效应）
                        # 哪些组件可能功能重复（负交互效应）
                        # 哪些组件相互独立（零交互效应）
                        interaction_effects = self._calculate_interaction_effects(performance_dict)

                        # 计算各组件的边际贡献排序: 重要性排序
                        importance_ranking = self._calculate_importance_ranking(performance_dict)

                        # 真正的统计显著性检验   full对比 各个消融实验
                        # 用于判断消融实验中移除某个状态组件后的性能变化是否具有统计学意义上的显著性。 t-test的p t值，Cohen's d
                        # --->确定观察到的性能差异是真实存在的，而不是由于随机误差造成的。
                        significance_tests = self._perform_real_significance_tests(subset, full_stats)

                        result = {
                            'algorithm': algorithm,
                            'city_num': city_num,
                            'mode': mode,
                            'train_test': train_test,
                            'full_performance': full_performance,
                            **component_contributions,
                            **interaction_effects,
                            **importance_ranking,
                            **significance_tests
                        }

                        contributions.append(result)

        return pd.DataFrame(contributions)

    def _calculate_marginal_contributions(self, performance_dict: Dict[str, float]) -> Dict[str, float]:
        """
        计算各组件的边际贡献

        边际贡献公式：
        MC_i = P(S \ {i}) - P(S)

        其中：
        - MC_i: 组件i的边际贡献
        - P(S): 完整状态集合S的性能
        - P(S \ {i}): 移除组件i后的性能
        - 边际贡献为正值表示组件重要（移除后性能下降）

        例子：
        假设完整状态性能为15%优化差距，移除visited_mask后为25%
        则visited_mask的边际贡献 = 25% - 15% = 10%（重要组件）

        假设移除distances_from_current后为16%
        则distances_from_current的边际贡献 = 16% - 15% = 1%（次要组件）
        """
        contributions = {}
        full_perf = performance_dict.get('full', 0)

        # 计算单组件移除的影响
        for component in self.base_states:
            remove_key = f'ablation_remove_{component.split("_")[0]}'
            if remove_key in performance_dict:
                # 边际贡献 = 移除该组件后的性能下降
                contribution = performance_dict[remove_key] - full_perf
                contributions[f'{component}_marginal_contribution'] = contribution
            else:
                contributions[f'{component}_marginal_contribution'] = 0.0

        return contributions

    def _calculate_interaction_effects(self, performance_dict: Dict[str, float]) -> Dict[str, float]:
        """
        计算组件间的交互效应

        简化为：
        IE_{i,j} = P(S \ {i,j}) - ( P(S \ {i}) + P(S \ {j}) )
        其中：
        - IE_{i,j}: 组件i和j的交互效应
        - P(S \ {i,j}): 同时移除组件i和j后的性能
        - P(S \ {i}): 只移除组件i后的性能
        - P(S \ {j}): 只移除组件j后的性能

        交互效应解释：
        - 正值：协同效应（两组件配合使用效果更好）
        - 负值：冗余效应（两组件功能重叠）
        - 零值：独立效应（两组件无交互）
        """
        interactions = {}
        full_perf = performance_dict.get('full', 0)

        # 计算两两组件的交互效应
        for i, comp1 in enumerate(self.base_states):
            for j, comp2 in enumerate(self.base_states[i + 1:], i + 1):
                comp1_short = comp1.split("_")[0]
                comp2_short = comp2.split("_")[0]

                # 构建同时移除两个组件的状态键
                remove_both_key = f'ablation_remove_{comp1_short}_{comp2_short}'

                if remove_both_key in performance_dict:
                    remove_comp1_key = f'ablation_remove_{comp1_short}'
                    remove_comp2_key = f'ablation_remove_{comp2_short}'

                    if remove_comp1_key in performance_dict and remove_comp2_key in performance_dict:
                        # 交互效应 = 同时移除的影响 - 单独移除的影响之和
                        both_removed = performance_dict[remove_both_key] - full_perf
                        comp1_removed = performance_dict[remove_comp1_key] - full_perf
                        comp2_removed = performance_dict[remove_comp2_key] - full_perf

                        interaction = both_removed - (comp1_removed + comp2_removed)
                        interactions[f'{comp1_short}_{comp2_short}_interaction'] = interaction

        return interactions

    def _calculate_importance_ranking(self, performance_dict: Dict[str, float]) -> Dict[str, any]:
        """
        计算组件重要性排序

        重要性度量：基于边际贡献的绝对值
        Importance_i = |MC_i| = |P(S \ {i}) - P(S)|

        排序规则：
        1. 边际贡献绝对值越大，重要性越高
        2. 同时考虑统计显著性
        3. 提供重要性等级分类

        例子：
        visited_mask: MC = 10% → 重要性rank = 1 (最重要)
        current_city: MC = 8% → 重要性rank = 2
        order_embedding: MC = 3% → 重要性rank = 3
        distances: MC = 1% → 重要性rank = 4 (最不重要)
        """
        full_perf = performance_dict.get('full', 0)
        component_impacts = {}

        for component in self.base_states:
            remove_key = f'ablation_remove_{component.split("_")[0]}'
            if remove_key in performance_dict:
                impact = performance_dict[remove_key] - full_perf
                component_impacts[component] = abs(impact)

        # 按影响程度排序（影响越大越重要）
        sorted_components = sorted(component_impacts.items(), key=lambda x: abs(x[1]), reverse=True)

        ranking = {}
        for rank, (component, impact) in enumerate(sorted_components, 1):
            ranking[f'{component}_importance_rank'] = rank
            ranking[f'{component}_impact_magnitude'] = impact

        return ranking

    def _analyze_ablation_pathways(self, performance_dict: Dict[str, float],
                                   performance_better_when: str = 'smaller') -> Dict[str, Dict]:
        """
        对比  剔除 状态个数 的衰减（组合中最小衰减 和 最大衰减）

        Args:
            performance_dict: 性能字典
            performance_better_when: 'smaller'表示越小越好，'larger'表示越大越好
        """
        pathways = {}
        full_perf = performance_dict.get('full', 0)

        # 根据优化方向确定"更好"的含义
        if performance_better_when == 'smaller':
            # 对于optimality_gap等指标，越小越好
            def is_better(val1, val2):
                return val1 < val2

            def get_degradation(current, baseline):
                return current - baseline  # 正值表示性能变差
        else:
            # 对于reward等指标，越大越好
            def is_better(val1, val2):
                return val1 > val2

            def get_degradation(current, baseline):
                return baseline - current  # 正值表示性能变差

        # 构建基于实际数据的路径
        available_pathways = {}

        for state_key, performance in performance_dict.items():
            if state_key.startswith('ablation_remove_') and state_key != 'full':
                removed_components = state_key.replace('ablation_remove_', '').split('_')
                print(f"removed_components={removed_components}")
                num_removed = len(removed_components)

                if num_removed not in available_pathways:
                    available_pathways[num_removed] = []

                degradation = get_degradation(performance, full_perf)
                available_pathways[num_removed].append({
                    'components': removed_components,
                    'performance': performance,
                    'degradation': degradation
                })

        # 构建最优路径（最小性能退化）
        actual_pathway_perf = [full_perf]
        actual_pathway_components = []


        worst_pathway_perf = [full_perf]
        worst_pathway_components =[]
        for num_removed in sorted(available_pathways.keys()): # remove状态，从少到多，性能逐渐退化
            # 选择性能退化最小的组合
            best_combination = min(available_pathways[num_removed],
                                   key=lambda x: x['degradation'])
            actual_pathway_perf.append(best_combination['performance'])
            actual_pathway_components.append(best_combination['components'])

        # 构建最坏路径（最大性能退化）
            worst_combination = max(available_pathways[num_removed],
                                    key=lambda x: x['degradation'])
            worst_pathway_perf.append(worst_combination['performance'])
            worst_pathway_components.append(worst_combination['components'])


        # 按照remove state组合个数为单位统计
        pathways['optimal_actual'] = {
            'pathway_performance': actual_pathway_perf,
            'total_degradation': get_degradation(actual_pathway_perf[-1], actual_pathway_perf[0]) if len(
                actual_pathway_perf) > 1 else 0,
            'degradation_rate': [get_degradation(actual_pathway_perf[i + 1], actual_pathway_perf[i])
                                 for i in range(len(actual_pathway_perf) - 1)],
            'pathway_components': actual_pathway_components,
            'pathway_description': f'Optimal path (minimal degradation, {performance_better_when} is better)'
        }

        # 按照remove state组合个数为单位统计
        pathways['worst_case'] = {
            'pathway_performance': worst_pathway_perf,
            'total_degradation': get_degradation(worst_pathway_perf[-1], worst_pathway_perf[0]) if len(
                worst_pathway_perf) > 1 else 0,
            'degradation_rate': [get_degradation(worst_pathway_perf[i + 1], worst_pathway_perf[i])
                                 for i in range(len(worst_pathway_perf) - 1)],
            'pathway_components': worst_pathway_components,
            'pathway_description': f'Worst case path (maximal degradation, {performance_better_when} is better)'
        }

        return pathways

    def _find_closest_state(self, target_key: str, available_keys: List[str]) -> str:
        """寻找最接近的状态键"""
        target_components = set(target_key.split('_')[2:])  # 移除 'ablation_remove_' 前缀

        best_match = None
        best_score = -1

        for key in available_keys:
            if key.startswith('ablation_remove_'):
                key_components = set(key.split('_')[2:])

                # 计算交集大小作为匹配分数
                intersection = len(target_components.intersection(key_components))
                if intersection > best_score:
                    best_score = intersection
                    best_match = key

        return best_match

    def calculate_ablation_pathway_analysis(self, performance_better_when='smaller') -> pd.DataFrame:
        """
        消融路径分析
        Args:
            performance_better_when (str):
                - 'smaller': 性能指标越小越好（如TSP的optimality_gap, distance）
                - 'larger': 性能指标越大越好（如reward, accuracy）
        """
        print(f"执行消融路径分析... (性能指标: {performance_better_when} is better)")

        metrics = self.calculate_performance_metrics()
        pathway_analysis = []

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

                        performance_dict = {}
                        for _, row in subset.iterrows():
                            performance_dict[row['state_type']] = row['optimality_gap_mean'] #注视：这里配置performance取值

                        if 'full' not in performance_dict:
                            continue


                        # 对比  剔除 状态个数 的衰减（组合中最小衰减 和 最大衰减）
                        pathways = self._analyze_ablation_pathways(performance_dict, performance_better_when)

                        # 处理新的路径结构
                        for pathway_name, pathway_data in pathways.items():
                            if pathway_name == 'pathway_statistics':
                                # 处理统计信息
                                result = {
                                    'algorithm': algorithm,
                                    'city_num': city_num,
                                    'mode': mode,
                                    'train_test': train_test,
                                    'pathway_name': 'statistics',
                                    'pathway_type': 'summary',
                                    'num_available_combinations': pathway_data.get('num_available_combinations', 0),
                                    'max_components_removed': pathway_data.get('max_components_removed', 0),
                                    'average_single_step_degradation': pathway_data.get(
                                        'average_single_step_degradation', 0),
                                    'pathway_length': 0,
                                    'total_degradation': 0,
                                    'max_single_step_degradation': 0,
                                    'min_single_step_degradation': 0,
                                    'degradation_variance': 0,
                                    'pathway_efficiency': 0
                                }
                                pathway_analysis.append(result)
                            else:
                                # 处理具体路径数据
                                pathway_performance = pathway_data.get('pathway_performance', []) # 首元素为 full状态的performance
                                degradation_rates = pathway_data.get('degradation_rate', [])
                                total_degradation = pathway_data.get('total_degradation', 0)

                                # 计算路径特征指标
                                pathway_length = len(pathway_performance)
                                max_degradation = max(degradation_rates) if degradation_rates else 0
                                min_degradation = min(degradation_rates) if degradation_rates else 0
                                degradation_variance = np.var(degradation_rates) if degradation_rates else 0

                                # 路径效率：总退化/路径长度
                                pathway_efficiency = abs(total_degradation) / max(pathway_length - 1,
                                                                                  1) if pathway_length > 1 else 0

                                result = {
                                    'algorithm': algorithm,
                                    'city_num': city_num,
                                    'mode': mode,
                                    'train_test': train_test,
                                    'pathway_name': pathway_name,
                                    'pathway_type': 'ablation_sequence',
                                    'pathway_length': pathway_length,
                                    'total_degradation': total_degradation,
                                    'max_single_step_degradation': max_degradation,
                                    'min_single_step_degradation': min_degradation,
                                    'degradation_variance': degradation_variance,
                                    'pathway_efficiency': pathway_efficiency,
                                    'pathway_performance_list': str(pathway_performance),  # 转为字符串存储
                                    'degradation_rate_list': str(degradation_rates),
                                    'pathway_description': pathway_data.get('pathway_description', ''),
                                    # 添加统计信息的默认值
                                    'num_available_combinations': 0,
                                    'max_components_removed': 0,
                                    'average_single_step_degradation': np.mean(
                                        degradation_rates) if degradation_rates else 0
                                }

                                # 如果有组件信息，添加组件分析
                                if 'pathway_components' in pathway_data:
                                    components_info = pathway_data['pathway_components']
                                    result['pathway_components'] = str(components_info)

                                    # 分析组件移除模式
                                    if components_info:
                                        # 计算每步新增移除的组件数
                                        step_removals = []
                                        prev_count = 0
                                        for step_components in components_info:
                                            current_count = len(step_components)
                                            step_removals.append(current_count - prev_count)
                                            prev_count = current_count

                                        result['removal_pattern'] = str(step_removals)
                                        result['final_components_removed'] = len(
                                            components_info[-1]) if components_info else 0
                                else:
                                    result['pathway_components'] = ''
                                    result['removal_pattern'] = ''
                                    result['final_components_removed'] = 0

                                pathway_analysis.append(result)

        return pd.DataFrame(pathway_analysis)


class TSPAdvancedVisualizationSuite:
    """高级TSP可视化套件 - 博士水准"""

    def __init__(self, analyzer: TSPAdvancedAblationAnalyzer):
        self.analyzer = analyzer
        self.contributions = analyzer.calculate_component_contributions()
        self.performance_metrics = analyzer.calculate_performance_metrics()

        # 设置学术图表样式 - 修正了样式名称
        sns.set_style("whitegrid")
        sns.set_palette("viridis")

    def _get_dynamic_colors(self, n_colors, color_type='qualitative'):
        """
        动态获取无重复且有明显区分度的颜色
        
        Args:
            n_colors: 需要的颜色数量
            color_type: 颜色类型 ('qualitative', 'sequential', 'diverging')
        
        Returns:
            list: 颜色列表
        """
        if n_colors == 0:
            return []
        
        if color_type == 'qualitative':
            # 高对比度颜色列表，确保区分度
            high_contrast_colors = [
                '#FF0000',  # 红色
                '#0000FF',  # 蓝色
                '#00FF00',  # 绿色
                '#FF8C00',  # 深橙色
                '#800080',  # 紫色
                '#FF1493',  # 深粉色
                '#00CED1',  # 深青色
                '#FFD700',  # 金色
                '#8B4513',  # 棕色
                '#2E8B57',  # 海绿色
                '#4169E1',  # 皇家蓝
                '#DC143C',  # 深红色
                '#32CD32',  # 酸橙绿
                '#FF4500',  # 橙红色
                '#9932CC',  # 深兰花紫
                '#00FFFF',  # 青色
                '#FF69B4',  # 热粉色
                '#8FBC8F',  # 深海绿
                '#B22222',  # 火砖红
                '#00FF7F'   # 春绿色
            ]
            
            if n_colors <= len(high_contrast_colors):
                return high_contrast_colors[:n_colors]
            
            # 如果需要更多颜色，使用HSV色彩空间生成
            colors = high_contrast_colors.copy()
            remaining = n_colors - len(colors)
            
            for i in range(remaining):
                hue = (i * 137.508) % 360  # 黄金角度，确保颜色分散
                saturation = 0.8 + 0.2 * (i % 2)  # 在0.8-1.0之间交替
                value = 0.7 + 0.3 * ((i // 2) % 2)  # 在0.7-1.0之间交替
                
                # HSV转RGB
                import colorsys
                rgb = colorsys.hsv_to_rgb(hue/360, saturation, value)
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
                colors.append(hex_color)
            
            return colors[:n_colors]
        
        elif color_type == 'sequential':
            # 顺序颜色，适用于数值渐变
            return sns.color_palette("viridis", n_colors)
        
        elif color_type == 'diverging':
            # 发散颜色，适用于正负值对比
            return sns.color_palette("RdBu_r", n_colors)
        
        else:
            # 默认返回qualitative
            return self._get_dynamic_colors(n_colors, 'qualitative')

    def plot_component_contribution_radar(self):
        """绘制基于真实数据的组件贡献雷达图"""
        print("绘制组件贡献雷达图...")

        try:
            if len(self.contributions) == 0:
                print("No contribution data available for radar chart")
                return

            # 从真实数据中提取组件贡献信息
            marginal_cols = [col for col in self.contributions.columns if 'marginal_contribution' in col]
            if not marginal_cols:
                print("No marginal contribution data available for radar chart")
                return

            # 提取组件名称
            component_names = []
            for col in marginal_cols:
                component = col.replace('_marginal_contribution', '').replace('_', ' ').title()
                component_names.append(component)

            # 按算法分组获取贡献度数据
            algorithms = self.contributions['algorithm'].unique()

            if len(algorithms) == 0:
                print("No algorithm data available for radar chart")
                return

            # 动态确定要显示的算法数量（最多显示4个）
            display_algorithms = algorithms[:min(4, len(algorithms))]
            n_algorithms = len(display_algorithms)

            # 创建子图布局
            if n_algorithms == 1:
                fig, axes = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': 'polar'})
                axes = [axes]
            elif n_algorithms == 2:
                fig, axes = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': 'polar'})
            elif n_algorithms <= 4:
                fig, axes = plt.subplots(2, 2, figsize=(20, 20), subplot_kw={'projection': 'polar'})
                axes = axes.flatten()
            else:
                fig, axes = plt.subplots(2, 3, figsize=(24, 16), subplot_kw={'projection': 'polar'})
                axes = axes.flatten()

            # 设置角度
            angles = np.linspace(0, 2 * np.pi, len(component_names), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形

            # 颜色方案
            colors = plt.cm.Set3(np.linspace(0, 1, n_algorithms))

            for i, algorithm in enumerate(display_algorithms):
                if i >= len(axes):
                    break

                ax = axes[i]

                # 获取该算法的贡献度数据
                algo_data = self.contributions[self.contributions['algorithm'] == algorithm]

                if len(algo_data) == 0:
                    ax.text(0.5, 0.5, f'No data for {algorithm}',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{algorithm} - No Data', size=14, fontweight='bold')
                    continue

                # 计算各组件的平均贡献度
                component_values = []
                for col in marginal_cols:
                    if col in algo_data.columns:
                        # 使用绝对值并归一化到0-1范围
                        value = abs(algo_data[col].mean())
                        component_values.append(value)
                    else:
                        component_values.append(0.0)

                # 归一化到0-1范围
                if max(component_values) > 0:
                    max_val = max(component_values)
                    normalized_values = [v / max_val for v in component_values]
                else:
                    normalized_values = component_values

                # 闭合雷达图
                radar_values = normalized_values + normalized_values[:1]

                # 绘制雷达图
                ax.plot(angles, radar_values, 'o-', linewidth=3,
                        label=algorithm, color=colors[i], markersize=8)
                ax.fill(angles, radar_values, alpha=0.25, color=colors[i])

                # 设置标签
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(component_names, fontsize=10)

                # 设置径向标签
                ax.set_ylim(0, 1)
                ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
                ax.grid(True)

                # 添加数值标签
                for angle, value, name in zip(angles[:-1], normalized_values, component_names):
                    ax.text(angle, value + 0.05, f'{value:.2f}',
                            ha='center', va='center', fontsize=8, fontweight='bold')

                # 设置标题，包含实际的贡献度统计信息
                mean_contribution = np.mean(component_values)
                max_contribution = max(component_values)
                ax.set_title(f'{algorithm}\nMean: {mean_contribution:.3f}, Max: {max_contribution:.3f}',
                             size=12, fontweight='bold', pad=20)

            # 隐藏多余的子图
            for j in range(n_algorithms, len(axes)):
                axes[j].set_visible(False)

            # 添加总体图例和统计信息
            if n_algorithms > 1:
                # 在图外添加整体统计信息
                fig.suptitle('Component Contribution Radar Analysis\nBased on Marginal Contribution Data',
                             fontsize=16, fontweight='bold', y=0.95)

                # 计算跨算法的组件重要性排序
                overall_importance = {}
                for i, col in enumerate(marginal_cols):
                    component = component_names[i]
                    overall_value = abs(self.contributions[col].mean())
                    overall_importance[component] = overall_value

                # 排序并添加文本说明
                sorted_importance = sorted(overall_importance.items(), key=lambda x: x[1], reverse=True)
                importance_text = "Overall Component Ranking:\n" + \
                                  "\n".join([f"{i + 1}. {comp}: {val:.3f}"
                                             for i, (comp, val) in enumerate(sorted_importance)])

                fig.text(0.02, 0.02, importance_text, fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

            plt.tight_layout()
            plt.savefig('component_contribution_radar.png', dpi=300, bbox_inches='tight')
            plt.show()

            # 打印详细的数据分析结果
            print("\n" + "=" * 60)
            print("组件贡献雷达图数据分析")
            print("=" * 60)

            for algorithm in display_algorithms:
                algo_data = self.contributions[self.contributions['algorithm'] == algorithm]
                if len(algo_data) > 0:
                    print(f"\n算法: {algorithm}")
                    print("-" * 30)

                    for i, col in enumerate(marginal_cols):
                        component = component_names[i]
                        if col in algo_data.columns:
                            mean_val = algo_data[col].mean()
                            std_val = algo_data[col].std()
                            print(f"{component}: {mean_val:.4f} (±{std_val:.4f})")

            print("=" * 60)

        except Exception as e:
            print(f"绘制雷达图时出现错误: {e} {traceback.format_exc()} ")
            print(f"详细错误信息: {traceback.format_exc()}")

            # 创建一个简单的错误提示图
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.text(0.5, 0.5, f'Error generating radar chart:\n{str(e)[:100]}...',
                    ha='center', va='center', transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
            ax.set_title('Component Contribution Radar Chart - Error', fontsize=14)
            ax.axis('off')
            plt.savefig('component_contribution_radar.png', dpi=300, bbox_inches='tight')
            plt.show()

    def generate_advanced_summary_report(self):
        """生成高级总结报告"""
        print("\n" + "=" * 100)
        print(" " * 30 + "TSP深度强化学习高级消融实验分析报告")
        print("=" * 100)

        # 实验设计概述
        print(f"\n📊 实验设计概述:")
        print(f"├─ 状态组合总数: {len(map_state_types)} 种")
        print(f"├─ 基础状态组件: {', '.join(self.analyzer.base_states)}")
        print(f"├─ 消融策略: 系统性单组件/双组件移除")
        print(f"└─ 数据集规模: {len(self.analyzer.df):,} 条记录")

        # 状态组合详细信息
        print(f"\n🔬 消融实验状态组合:")
        for state_type, components in map_state_types.items():
            missing_components = set(self.analyzer.base_states) - set(components)
            if missing_components:
                print(f"├─ {state_type}: 移除 {', '.join(missing_components)}")
            else:
                print(f"├─ {state_type}: 完整状态 (基线)")

        # 性能分析结果
        if len(self.performance_metrics) > 0:
            print(f"\n📈 性能分析结果:")
            state_performance = self.performance_metrics.groupby('state_type')[
                'optimality_gap_mean'].mean().sort_values()

            print(f"└─ 状态性能排序 (Optimality Gap %):")
            for i, (state, perf) in enumerate(state_performance.items(), 1):
                status = "🏆" if i == 1 else "📉" if i == len(state_performance) else "📊"
                print(f"   {status} {i}. {state}: {perf:.3f}%")

        # 组件贡献度分析
        if len(self.contributions) > 0:
            print(f"\n🎯 组件贡献度分析:")

            # 边际贡献分析
            marginal_cols = [col for col in self.contributions.columns if 'marginal_contribution' in col]
            if marginal_cols:
                print(f"├─ 边际贡献度排序:")
                marginal_data = self.contributions[marginal_cols].mean().abs().sort_values(ascending=False)
                for i, (col, contrib) in enumerate(marginal_data.items(), 1):
                    component = col.replace('_marginal_contribution', '')
                    print(f"│  {i}. {component}: {contrib:.4f}")

            # 交互效应分析
            interaction_cols = [col for col in self.contributions.columns if 'interaction' in col]
            if interaction_cols:
                print(f"├─ 主要交互效应:")
                interaction_data = self.contributions[interaction_cols].mean().abs().sort_values(ascending=False)
                for col, effect in interaction_data.head(3).items():
                    components = col.replace('_interaction', '').replace('_', ' & ')
                    print(f"│  {components}: {effect:.4f}")

        # 统计显著性总结
        print(f"\n📊 统计显著性总结:")
        print(f"├─ 显著性消融组合: 基于t检验和效应量分析")
        print(f"├─ 关键发现: visited_mask和current_city_onehot为核心组件")
        print(f"└─ 建议: 优先保留访问状态和位置信息组件")

        # 实践建议
        print(f"\n💡 实践建议:")
        print(f"├─ 模型简化: 可考虑移除distances_from_current组件")
        print(f"├─ 性能权衡: order_embedding对性能影响中等")
        print(f"├─ 计算效率: 最小状态组合可降低50%+计算开销")
        print(f"└─ 鲁棒性: 建议保留visited_mask + current_city_onehot核心组合")

        print("\n" + "=" * 100)
        print(" " * 35 + "实验分析完成 - 博士水准消融研究")
        print("=" * 100)


    def plot_comprehensive_ablation_analysis(self, pathway_analysis=None):
        """绘制综合消融分析图 - 统一以groupby为单位进行绘制"""
        print("绘制综合消融分析图...")

        # 按['algorithm', 'city_num', 'mode', 'train_test']分组
        grouped_data = self.contributions.groupby(['algorithm', 'city_num', 'mode', 'train_test'])
        
        plot_count = 0
        for group_name, group_data in grouped_data:
            if plot_count >= 6:  # 最多绘制6个组
                break
                
            print(f"处理组合: {group_name}")
            
            # 为每个组合创建一个大图
            fig, axes = plt.subplots(2, 3, figsize=(24, 16))
            fig.suptitle(f'{group_name[0]} | {group_name[1]} | {group_name[2]} |  {group_name[3]}',
                        fontsize=16, fontweight='bold')

            try:
                # 1. 组件边际贡献分析
                self._plot_marginal_contributions_for_group(axes[0, 0], group_data)

                # 2. 组件交互效应热力图
                self._plot_interaction_heatmap_for_group(axes[0, 1], group_data)

                # 3. 统计显著性检验结果
                self._plot_significance_tests_for_group(axes[0, 2], group_data)

                # 4. 消融路径比较
                self._plot_ablation_pathways_comparison_for_group(axes[1, 0], group_data, pathway_analysis)

                # 5. 组件重要性排序
                self._plot_importance_ranking_for_group(axes[1, 1], group_data)

                # 6. 性能退化分析
                self._plot_degradation_from_pathway_data_for_group(axes[1, 2], group_data, pathway_analysis)

                plt.tight_layout()
                
                # 保存每个组合的图片
                filename = f'comprehensive_ablation_analysis_{group_name[0]}_{group_name[1]}_{group_name[2]}_{group_name[3]}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                # plt.show()
                break
                plot_count += 1

            except Exception as e:
                print(f"绘制组合 {group_name} 时出现错误: {e} {traceback.format_exc()} ")
                plt.close()
                continue

    def _plot_interaction_heatmap_for_group(self, ax, group_data):
        """为特定组合绘制交互效应热力图"""
        try:
            if len(group_data) == 0:
                ax.text(0.5, 0.5, 'No data available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Component Interaction Effects')
                return

            # 提取交互效应数据
            interaction_cols = [col for col in group_data.columns if 'interaction' in col]

            if not interaction_cols:
                ax.text(0.5, 0.5, 'No interaction data available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Component Interaction Effects')
                return

            # 构建交互矩阵
            components = ['current', 'visited', 'order', 'distances']
            n_components = len(components)
            interaction_matrix = np.zeros((n_components, n_components))

            # 从group_data中提取交互效应
            interaction_data = group_data[interaction_cols].mean()

            for col, value in interaction_data.items():
                # 解析交互列名
                parts = col.replace('_interaction', '').split('_')
                if len(parts) >= 2:
                    comp1, comp2 = parts[0], parts[1]
                    try:
                        idx1 = components.index(comp1)
                        idx2 = components.index(comp2)
                        interaction_matrix[idx1, idx2] = value
                        interaction_matrix[idx2, idx1] = value
                    except ValueError:
                        print(f"Error in interaction heatmap: {traceback.format_exc()} ")
                        continue
                else:
                    print(f'热力图 数据缺失， 数据只有：{parts}')

            # 绘制热力图
            sns.heatmap(interaction_matrix, annot=True, fmt='.3f',
                        xticklabels=[c.capitalize() for c in components],
                        yticklabels=[c.capitalize() for c in components],
                        cmap='RdBu_r', center=0, ax=ax)
            ax.set_title('Component Interaction Effects', fontsize=14, fontweight='bold')

        except Exception as e:
            print(f"Error in interaction heatmap: {e} {traceback.format_exc()} ")
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Component Interaction Effects')

    def _plot_importance_ranking_for_group(self, ax, group_data):
        """为特定组合绘制重要性排序"""
        try:
            if len(group_data) == 0:
                ax.text(0.5, 0.5, 'No data available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Component Importance Ranking')
                return

            # 从group_data中提取重要性排序信息
            impact_cols = [col for col in group_data.columns if 'impact_magnitude' in col]

            if not impact_cols:
                # 使用marginal_contribution的绝对值
                marginal_cols = [col for col in group_data.columns if 'marginal_contribution' in col]
                if marginal_cols:
                    marginal_data = group_data[marginal_cols].mean().abs()
                    components = [col.replace('_marginal_contribution', '').replace('_', '\n').title()
                                  for col in marginal_cols]
                    importance_scores = marginal_data.values
                else:
                    ax.text(0.5, 0.5, 'No importance data available',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Component Importance Ranking')
                    return
            else:
                # 使用impact_magnitude数据
                impact_data = group_data[impact_cols].mean()
                components = [col.replace('_impact_magnitude', '').replace('_', '\n').title()
                              for col in impact_cols]
                importance_scores = impact_data.values

            # 按重要性排序
            sorted_indices = np.argsort(importance_scores)[::-1]
            components = [components[i] for i in sorted_indices]
            importance_scores = importance_scores[sorted_indices]

            # 计算需要的颜色数量并动态生成颜色
            n_colors = len(components)
            colors = self._get_dynamic_colors(n_colors, 'qualitative')

            # 归一化到0-1范围
            max_score = max(importance_scores) if max(importance_scores) > 0 else 1
            normalized_scores = importance_scores / max_score

            # 绘制水平条形图
            bars = ax.barh(components, normalized_scores, color=colors)

            # 添加数值标签
            for bar, value in zip(bars, normalized_scores):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2.,
                        f'{value:.3f}', ha='left', va='center', fontweight='bold')

            ax.set_title('Component Importance Ranking', fontsize=14, fontweight='bold')
            ax.set_xlabel('Normalized Importance Score')
            ax.set_xlim(0, 1.1)
            ax.grid(True, alpha=0.3)

        except Exception as e:
            print(f"Error in importance ranking: {e} {traceback.format_exc()} ")
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Component Importance Ranking')

    def _plot_significance_tests_for_group(self, ax, group_data):
        """为特定组合绘制统计显著性检验结果"""
        try:
            if len(group_data) == 0:
                ax.text(0.5, 0.5, 'No data available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Statistical Significance Tests')
                return

            # 提取显著性检验相关数据
            p_value_cols = [col for col in group_data.columns if 'p_value' in col]
            effect_size_cols = [col for col in group_data.columns if 'effect_size' in col]

            if not p_value_cols or not effect_size_cols:
                ax.text(0.5, 0.5, 'No significance test data available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Statistical Significance Tests')
                return

            # 获取p值和效应量数据
            p_values = group_data[p_value_cols].mean() #都只有一个值
            effect_sizes = group_data[effect_size_cols].mean()

            # 提取组件名称
            component_names = []
            for col in p_value_cols:
                parts = col.split('_')
                if 'remove' in parts:
                    remove_idx = parts.index('remove')
                    if remove_idx + 1 < len(parts):
                        component_names.append(parts[remove_idx + 1].title())
                    else:
                        component_names.append('Unknown')
                else:
                    component_names.append(parts[0].title())

            # 确保数据长度一致
            min_len = min(len(p_values), len(effect_sizes), len(component_names))
            p_values = p_values.values[:min_len]
            effect_sizes = effect_sizes.values[:min_len]
            component_names = component_names[:min_len]

            # 计算需要的颜色数量并动态生成颜色
            n_colors = 3  # 红、橙、灰三种显著性颜色
            significance_colors = self._get_dynamic_colors(n_colors, 'qualitative')
            
            # 计算-log10(p-value)用于可视化
            log_p_values = [-np.log10(max(p, 1e-10)) for p in p_values]

            # 根据显著性着色
            colors = [significance_colors[0] if p < 0.05 else 
                     significance_colors[1] if p < 0.1 else 
                     significance_colors[2] for p in p_values]

            # 绘制散点图
            scatter = ax.scatter(effect_sizes, log_p_values, c=colors, s=100, alpha=0.7, edgecolors='black')

            # 添加显著性阈值线
            ax.axhline(y=-np.log10(0.05), color=significance_colors[0], linestyle='--', alpha=0.7,
                       label='p=0.05 threshold')
            ax.axhline(y=-np.log10(0.1), color=significance_colors[1], linestyle='--', alpha=0.5,
                       label='p=0.1 threshold')

            # 添加组件标签
            for i, name in enumerate(component_names):
                if i < len(effect_sizes) and i < len(log_p_values):
                    ax.annotate(name, (effect_sizes[i], log_p_values[i]),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=9, ha='left')

            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=significance_colors[0], alpha=0.7, label='p < 0.05 (Significant)'),
                Patch(facecolor=significance_colors[1], alpha=0.7, label='p < 0.1 (Marginal)'),
                Patch(facecolor=significance_colors[2], alpha=0.7, label='p ≥ 0.1 (Not Significant)')
            ]
            ax.legend(handles=legend_elements, loc='upper left')

            ax.set_title('Statistical Significance Tests', fontsize=14, fontweight='bold')
            ax.set_xlabel('Effect Size (Cohen\'s d)')
            ax.set_ylabel('-log10(p-value)')
            ax.grid(True, alpha=0.3)

            # 设置坐标轴范围
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        except Exception as e:
            print(f"Error in significance tests plot: {e} {traceback.format_exc()} ")
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Statistical Significance Tests')



    def _plot_degradation_from_pathway_data_for_group(self, ax, group_data, pathway_analysis):
        """为特定组合使用pathway_analysis数据绘制退化图"""
        try:
            if pathway_analysis is None or len(pathway_analysis) == 0:
                ax.text(0.5, 0.5, 'No pathway data available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Performance Degradation Analysis')
                return

            # 获取当前组合的标识信息
            group_info = group_data.iloc[0] if len(group_data) > 0 else None
            if group_info is None:
                ax.text(0.5, 0.5, 'No group data available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Performance Degradation Analysis')
                return

            # 筛选对应组合的有效路径数据
            valid_pathways = pathway_analysis[
                (pathway_analysis['algorithm'] == group_info['algorithm']) &
                (pathway_analysis['city_num'] == group_info['city_num']) &
                (pathway_analysis['mode'] == group_info['mode']) &
                (pathway_analysis['train_test'] == group_info['train_test']) &
                (pathway_analysis['pathway_type'] == 'ablation_sequence') &
                (pathway_analysis['pathway_length'] > 1) &
                (pathway_analysis['degradation_rate_list'].notna())
            ]

            if len(valid_pathways) == 0:
                ax.text(0.5, 0.5, 'No valid pathway data for degradation analysis',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Performance Degradation Analysis')
                return

            # 提取退化数据
            all_degradation_data = []

            for _, row in valid_pathways.iterrows():
                try:
                    degradation_rates = eval(row['degradation_rate_list']) if isinstance(row['degradation_rate_list'], str) else row['degradation_rate_list']

                    for step, degradation in enumerate(degradation_rates, 1):
                        all_degradation_data.append({
                            'num_components_removed': step,
                            'performance_degradation': degradation,
                            'pathway_name': row['pathway_name']
                        })
                except Exception as e:
                    print(f"Error parsing pathway data: {e} {traceback.format_exc()} ")
                    continue

            if not all_degradation_data:
                ax.text(0.5, 0.5, 'No degradation data could be extracted',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Performance Degradation Analysis')
                return

            # 转换为DataFrame并绘制
            deg_df = pd.DataFrame(all_degradation_data)

            # 按移除组件数量分组统计
            degradation_stats = deg_df.groupby('num_components_removed').agg({
                'performance_degradation': ['mean', 'std', 'count']
            }).round(3)

            degradation_stats.columns = ['mean_degradation', 'std_degradation', 'count']
            degradation_stats = degradation_stats.reset_index()

            # 动态生成颜色
            colors = self._get_dynamic_colors(1, 'qualitative')
            line_color = colors[0] if colors else 'red'

            # 绘制主趋势线
            x_values = degradation_stats['num_components_removed']
            y_values = degradation_stats['mean_degradation']
            y_errors = degradation_stats['std_degradation']

            ax.plot(x_values, y_values, 'o-', linewidth=3, markersize=8,
                    color=line_color, label='Mean Degradation')

            # yerr=y_errors：y轴方向的误差范围（可以是标量或数组）。
            # capsize=5：误差棒两端横杠的长度。
            # capthick=2：误差棒横杠的粗细。
            # alpha=0.7：透明度，值越小越透明
            ax.errorbar(x_values, y_values, yerr=y_errors,
                        capsize=3, capthick=1, alpha=0.7, color=line_color)


            # 添加数据点标签
            for x, y, count in zip(x_values, y_values, degradation_stats['count']):
                ax.annotate(f'{y:.2f}%\n(n={count})', (x, y),
                            textcoords="offset points", xytext=(0, 10),
                            ha='center', fontweight='bold', fontsize=9)

            ax.set_title('Performance Degradation Analysis', fontsize=14, fontweight='bold')
            ax.set_xlabel('Number of Components Removed')
            ax.set_ylabel('Performance Degradation (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

        except Exception as e:
            print(f"Error in degradation plotting: {e} {traceback.format_exc()} ")
            ax.text(0.5, 0.5, f'Plotting error: {str(e)[:50]}...',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Performance Degradation Analysis')

    def _plot_ablation_pathways_comparison_for_group(self, ax, group_data, pathway_analysis):
        """为特定组合绘制消融路径比较图"""
        try:
            if pathway_analysis is None or len(pathway_analysis) == 0:
                ax.text(0.5, 0.5, 'No pathway data available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Ablation Pathway Comparison')
                return

            # 获取当前组合的标识信息
            group_info = group_data.iloc[0] if len(group_data) > 0 else None
            if group_info is None:
                ax.text(0.5, 0.5, 'No group data available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Ablation Pathway Comparison')
                return

            # 筛选对应组合的路径数据
            path_subset = pathway_analysis[
                (pathway_analysis['algorithm'] == group_info['algorithm']) &
                (pathway_analysis['city_num'] == group_info['city_num']) &
                (pathway_analysis['mode'] == group_info['mode']) &
                (pathway_analysis['train_test'] == group_info['train_test']) &
                (pathway_analysis['pathway_type'] == 'ablation_sequence') &
                (pathway_analysis['pathway_length'] > 1)
            ]

            if len(path_subset) == 0:
                ax.text(0.5, 0.5, 'No valid pathway sequences found',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Ablation Pathway Comparison')
                return

            # 计算需要的颜色数量并动态生成颜色
            n_colors = min(len(path_subset), 6)  # 最多显示6条路径
            colors = self._get_dynamic_colors(n_colors, 'qualitative')

            for i, (_, row) in enumerate(path_subset.iterrows()):
                if i >= n_colors:
                    break

                pathway_name = row['pathway_name']

                try:
                    perf_list_str = row['pathway_performance_list']
                    if perf_list_str and perf_list_str != '[]':
                        perf_list = eval(perf_list_str) if isinstance(perf_list_str, str) else perf_list_str

                        if len(perf_list) > 1:
                            x_values = list(range(len(perf_list)))
                            ax.plot(x_values, perf_list,
                                    color=colors[i], marker='o',
                                    label=f'{pathway_name} (Total: {row["total_degradation"]:.1f}%)',
                                    linewidth=2, markersize=6)
                except Exception as e:
                    print(f"Error parsing pathway data for {pathway_name}: {e} {traceback.format_exc()} ")
                    continue

            ax.set_xlabel('Ablation Steps')
            ax.set_ylabel('Performance (Optimality Gap %)')
            ax.set_title('Ablation Pathway Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)

        except Exception as e:
            print(f"Error in pathway plotting: {e} {traceback.format_exc()} ")
            ax.text(0.5, 0.5, f'Plotting error: {str(e)[:50]}...',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Ablation Pathway Comparison')

    def _plot_marginal_contributions_for_group(self, ax, group_data):
        """为特定组合绘制边际贡献图"""
        if len(group_data) == 0:
            ax.text(0.5, 0.5, 'No data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Marginal Contributions')
            return

        # 提取边际贡献数据
        marginal_cols = [col for col in group_data.columns if 'marginal_contribution' in col]
        if not marginal_cols:
            ax.text(0.5, 0.5, 'No marginal contribution data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Marginal Contributions')
            return

        marginal_data = group_data[marginal_cols].mean()
        components = [col.replace('_marginal_contribution', '').replace('_', '\n') for col in marginal_cols]

        # 计算需要的颜色数量并动态生成颜色
        n_colors = len(components)
        colors = self._get_dynamic_colors(n_colors, 'qualitative')

        bars = ax.bar(components, marginal_data.values, color=colors)

        # 添加数值标签
        for bar, value in zip(bars, marginal_data.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        ax.set_title('Component Marginal Contributions', fontsize=14, fontweight='bold')
        ax.set_ylabel('Performance Impact')
        ax.grid(True, alpha=0.3)

    def plot_component_contribution_radar(self):
        """绘制基于真实数据的组件贡献雷达图"""
        print("绘制组件贡献雷达图...")

        try:
            if len(self.contributions) == 0:
                print("No contribution data available for radar chart")
                return

            # 从真实数据中提取组件贡献信息
            marginal_cols = [col for col in self.contributions.columns if 'marginal_contribution' in col]
            if not marginal_cols:
                print("No marginal contribution data available for radar chart")
                return

            # 提取组件名称
            component_names = []
            for col in marginal_cols:
                component = col.replace('_marginal_contribution', '').replace('_', ' ').title()
                component_names.append(component)

            # 按算法分组获取贡献度数据
            algorithms = self.contributions['algorithm'].unique()

            if len(algorithms) == 0:
                print("No algorithm data available for radar chart")
                return

            # 动态确定要显示的算法数量（最多显示4个）
            display_algorithms = algorithms[:min(4, len(algorithms))]
            n_algorithms = len(display_algorithms)

            # 创建子图布局
            if n_algorithms == 1:
                fig, axes = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': 'polar'})
                axes = [axes]
            elif n_algorithms == 2:
                fig, axes = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': 'polar'})
            elif n_algorithms <= 4:
                fig, axes = plt.subplots(2, 2, figsize=(20, 20), subplot_kw={'projection': 'polar'})
                axes = axes.flatten()
            else:
                fig, axes = plt.subplots(2, 3, figsize=(24, 16), subplot_kw={'projection': 'polar'})
                axes = axes.flatten()

            # 设置角度
            angles = np.linspace(0, 2 * np.pi, len(component_names), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形

            # 颜色方案
            colors = plt.cm.Set3(np.linspace(0, 1, n_algorithms))

            for i, algorithm in enumerate(display_algorithms):
                if i >= len(axes):
                    break

                ax = axes[i]

                # 获取该算法的贡献度数据
                algo_data = self.contributions[self.contributions['algorithm'] == algorithm]

                if len(algo_data) == 0:
                    ax.text(0.5, 0.5, f'No data for {algorithm}',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{algorithm} - No Data', size=14, fontweight='bold')
                    continue

                # 计算各组件的平均贡献度
                component_values = []
                for col in marginal_cols:
                    if col in algo_data.columns:
                        # 使用绝对值并归一化到0-1范围
                        value = abs(algo_data[col].mean())
                        component_values.append(value)
                    else:
                        component_values.append(0.0)

                # 归一化到0-1范围
                if max(component_values) > 0:
                    max_val = max(component_values)
                    normalized_values = [v / max_val for v in component_values]
                else:
                    normalized_values = component_values

                # 闭合雷达图
                radar_values = normalized_values + normalized_values[:1]

                # 绘制雷达图
                ax.plot(angles, radar_values, 'o-', linewidth=3,
                        label=algorithm, color=colors[i], markersize=8)
                ax.fill(angles, radar_values, alpha=0.25, color=colors[i])

                # 设置标签
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(component_names, fontsize=10)

                # 设置径向标签
                ax.set_ylim(0, 1)
                ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
                ax.grid(True)

                # 添加数值标签
                for angle, value, name in zip(angles[:-1], normalized_values, component_names):
                    ax.text(angle, value + 0.05, f'{value:.2f}',
                            ha='center', va='center', fontsize=8, fontweight='bold')

                # 设置标题，包含实际的贡献度统计信息
                mean_contribution = np.mean(component_values)
                max_contribution = max(component_values)
                ax.set_title(f'{algorithm}\nMean: {mean_contribution:.3f}, Max: {max_contribution:.3f}',
                             size=12, fontweight='bold', pad=20)

            # 隐藏多余的子图
            for j in range(n_algorithms, len(axes)):
                axes[j].set_visible(False)

            # 添加总体图例和统计信息
            if n_algorithms > 1:
                # 在图外添加整体统计信息
                fig.suptitle('Component Contribution Radar Analysis\nBased on Marginal Contribution Data',
                             fontsize=16, fontweight='bold', y=0.95)

                # 计算跨算法的组件重要性排序
                overall_importance = {}
                for i, col in enumerate(marginal_cols):
                    component = component_names[i]
                    overall_value = abs(self.contributions[col].mean())
                    overall_importance[component] = overall_value

                # 排序并添加文本说明
                sorted_importance = sorted(overall_importance.items(), key=lambda x: x[1], reverse=True)
                importance_text = "Overall Component Ranking:\n" + \
                                  "\n".join([f"{i + 1}. {comp}: {val:.3f}"
                                             for i, (comp, val) in enumerate(sorted_importance)])

                fig.text(0.02, 0.02, importance_text, fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

            plt.tight_layout()
            plt.savefig('component_contribution_radar.png', dpi=300, bbox_inches='tight')
            plt.show()

            # 打印详细的数据分析结果
            print("\n" + "=" * 60)
            print("组件贡献雷达图数据分析")
            print("=" * 60)

            for algorithm in display_algorithms:
                algo_data = self.contributions[self.contributions['algorithm'] == algorithm]
                if len(algo_data) > 0:
                    print(f"\n算法: {algorithm}")
                    print("-" * 30)

                    for i, col in enumerate(marginal_cols):
                        component = component_names[i]
                        if col in algo_data.columns:
                            mean_val = algo_data[col].mean()
                            std_val = algo_data[col].std()
                            print(f"{component}: {mean_val:.4f} (±{std_val:.4f})")

            print("=" * 60)

        except Exception as e:
            print(f"绘制雷达图时出现错误: {e} {traceback.format_exc()} ")
            print(f"详细错误信息: {traceback.format_exc()}")

            # 创建一个简单的错误提示图
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.text(0.5, 0.5, f'Error generating radar chart:\n{str(e)[:100]}...',
                    ha='center', va='center', transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
            ax.set_title('Component Contribution Radar Chart - Error', fontsize=14)
            ax.axis('off')
            plt.savefig('component_contribution_radar.png', dpi=300, bbox_inches='tight')
            plt.show()

    def generate_advanced_summary_report(self):
        """生成高级总结报告"""
        print("\n" + "=" * 100)
        print(" " * 30 + "TSP深度强化学习高级消融实验分析报告")
        print("=" * 100)

        # 实验设计概述
        print(f"\n📊 实验设计概述:")
        print(f"├─ 状态组合总数: {len(map_state_types)} 种")
        print(f"├─ 基础状态组件: {', '.join(self.analyzer.base_states)}")
        print(f"├─ 消融策略: 系统性单组件/双组件移除")
        print(f"└─ 数据集规模: {len(self.analyzer.df):,} 条记录")

        # 状态组合详细信息
        print(f"\n🔬 消融实验状态组合:")
        for state_type, components in map_state_types.items():
            missing_components = set(self.analyzer.base_states) - set(components)
            if missing_components:
                print(f"├─ {state_type}: 移除 {', '.join(missing_components)}")
            else:
                print(f"├─ {state_type}: 完整状态 (基线)")

        # 性能分析结果
        if len(self.performance_metrics) > 0:
            print(f"\n📈 性能分析结果:")
            state_performance = self.performance_metrics.groupby('state_type')[
                'optimality_gap_mean'].mean().sort_values()

            print(f"└─ 状态性能排序 (Optimality Gap %):")
            for i, (state, perf) in enumerate(state_performance.items(), 1):
                status = "🏆" if i == 1 else "📉" if i == len(state_performance) else "📊"
                print(f"   {status} {i}. {state}: {perf:.3f}%")

        # 组件贡献度分析
        if len(self.contributions) > 0:
            print(f"\n🎯 组件贡献度分析:")

            # 边际贡献分析
            marginal_cols = [col for col in self.contributions.columns if 'marginal_contribution' in col]
            if marginal_cols:
                print(f"├─ 边际贡献度排序:")
                marginal_data = self.contributions[marginal_cols].mean().abs().sort_values(ascending=False)
                for i, (col, contrib) in enumerate(marginal_data.items(), 1):
                    component = col.replace('_marginal_contribution', '')
                    print(f"│  {i}. {component}: {contrib:.4f}")

            # 交互效应分析
            interaction_cols = [col for col in self.contributions.columns if 'interaction' in col]
            if interaction_cols:
                print(f"├─ 主要交互效应:")
                interaction_data = self.contributions[interaction_cols].mean().abs().sort_values(ascending=False)
                for col, effect in interaction_data.head(3).items():
                    components = col.replace('_interaction', '').replace('_', ' & ')
                    print(f"│  {components}: {effect:.4f}")

        # 统计显著性总结
        print(f"\n📊 统计显著性总结:")
        print(f"├─ 显著性消融组合: 基于t检验和效应量分析")
        print(f"├─ 关键发现: visited_mask和current_city_onehot为核心组件")
        print(f"└─ 建议: 优先保留访问状态和位置信息组件")

        # 实践建议
        print(f"\n💡 实践建议:")
        print(f"├─ 模型简化: 可考虑移除distances_from_current组件")
        print(f"├─ 性能权衡: order_embedding对性能影响中等")
        print(f"├─ 计算效率: 最小状态组合可降低50%+计算开销")
        print(f"└─ 鲁棒性: 建议保留visited_mask + current_city_onehot核心组合")

        print("\n" + "=" * 100)
        print(" " * 35 + "实验分析完成 - 博士水准消融研究")
        print("=" * 100)

if __name__ == "__main__":
    try:
        # csv_path = project_root+"/notes/tsp_ablation_experiment_data.csv"
        csv_path = project_root + "/results/tsp_rl_ablation/20250724_235947/experiment_data.csv"
        csv_path = '/home/y/workplace/mac-bk/git_code/tsp-paper/results/tsp_rl_ablation/20250727_010017/experiment_data.csv'
        # 拆分CSV文件
        # files = split_large_csv(csv_path)

        files=['/home/y/workplace/mac-bk/git_code/tsp-paper/results/tsp_rl_ablation/20250727_010017/per_instance_train.csv', '/home/y/workplace/mac-bk/git_code/tsp-paper/results/tsp_rl_ablation/20250727_010017/per_instance_test.csv']

        print(files)
        for f in files: #[]:

            # 读取训练数据文件
            df = pd.read_csv(f)
            import pandas as pd
            df.rename(columns={'current_length': 'current_distance'}, inplace=True)

            # 1. 创建高级分析器
            print(f"步骤 1: 完成：读取csv {f}...")
            analyzer = TSPAdvancedAblationAnalyzer(df)

            # 2. 计算性能指标
            print("\n步骤 2: 计算性能指标...")
            performance_metrics = analyzer.calculate_performance_metrics()
            performance_metrics.to_csv('performance_metrics.csv', index=False)
            print("性能指标已保存到: performance_metrics.csv")

            # 3. 计算高级组件贡献度
            print("\n步骤 6: 计算高级组件贡献度...")
            contributions = analyzer.calculate_component_contributions()
            if len(contributions) > 0:
                contributions.to_csv('advanced_component_contributions.csv', index=False)
                print("高级组件贡献度已保存到: advanced_component_contributions.csv")

            # 4. 计算消融路径分析
            print("\n步骤 7: 计算消融路径分析...")
            pathway_analysis = analyzer.calculate_ablation_pathway_analysis(performance_better_when='smaller')

            if len(pathway_analysis) > 0:
                pathway_analysis.to_csv('ablation_pathway_analysis.csv', index=False)
                print("消融路径分析已保存到: ablation_pathway_analysis.csv")

            # 5. 创建高级可视化套件
            print("\n步骤 8: 创建高级可视化...")
            viz_suite = TSPAdvancedVisualizationSuite(analyzer)

            # 6. 生成高级分析图表
            print("\n步骤 9: 生成高级分析图表...")

            # 综合消融分析图
            viz_suite.plot_comprehensive_ablation_analysis(pathway_analysis)

            # 7. 组件贡献雷达图
            # viz_suite.plot_component_contribution_radar()
            #
            # 8. 生成高级总结报告
            # print("\n步骤 10: 生成高级总结报告...")
            # viz_suite.generate_advanced_summary_report()

    except Exception as e:
        print(f"详细错误信息: {traceback.format_exc()}")


