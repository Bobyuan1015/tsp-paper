下面train_agent函数中evaluate_agent函数是重新random生成坐标点，但是训练是在固定的n_city个坐标点，评估的时候新坐标是不是代表环境发生变化了。
要求：
1.select_action中在evaluate的时候使用贪心，即关闭随机选择动作
2.evaluate的时候，需要训练集上做测试，也需要评估在新坐标上随机生成n_cities个坐标点测试。
3.对比实验画图的时候，要求把训练集测试结果（对比最优路径和模型的路径，同时要对比最优距离和模型当前距离；如果是训练集，跑一次就行，因为是贪心策略，跑多次结果也是一样）。
4.对比实验的时候，要求测试集（新坐标）上的效果也要对比


def run_comparison_experiment(n_cities: int = 10,
                              algorithms: List[str] = None,
                              n_runs: int = 5,
                              base_seed: int = 42,
                              is_tsplib=False,
                              libname=""
                              ) -> Dict[str, Any]:
    """
    Run comparison experiment across multiple algorithms.

    Args:
        n_cities: Number of cities
        algorithms: List of algorithms to compare
        n_runs: Number of independent runs per algorithm
        base_seed: Base random seed

    Returns:
        Comparison results
    """
    if algorithms is None:
        algorithms = ["DQN_Basic", "REINFORCE", "DQN_LSTM",
                      "DQN_OrderEmbedding", "ActorCritic", "DQN_Optimal"]

    logger = setup_logging(level="INFO")
    logger.info(f"Starting comparison experiment: {n_cities} cities, {len(algorithms)} algorithms, {n_runs} runs each")

    results = {}
    training_data = {}

    for algorithm in algorithms:
        logger.info(f"Training {algorithm}...")

        algorithm_results = {
            'distances': [],
            'rewards': [],
            'gaps': [],
            'convergence_episodes': [],
            'final_losses': []
        }

        algorithm_training_data = []

        for run in range(n_runs):
            seed = base_seed + run * 1000
            logger.info(f"  Run {run + 1}/{n_runs} (seed={seed})")

            # Create config for this run
            config = create_experiment_config(
                agent_type=algorithm,
                n_cities=n_cities,
                custom_params={'seed': seed, 'experiment_name': f"{algorithm}_{n_cities}cities_run{run}",
                               'use_tsplib': is_tsplib, 'tsplib_name': libname}
            )

            # Train agent
            agent, run_results = train_agent(config)

            # Store results
            algorithm_results['distances'].append(run_results['evaluation']['avg_distance'])
            algorithm_results['rewards'].append(run_results['evaluation']['avg_reward'])
            algorithm_results['gaps'].append(run_results['gap_percentage'])
            algorithm_results['final_losses'].append(run_results['episode_losses'][-1])

            # Store training data for plotting
            episode_data = {
                'episode': list(range(len(run_results['episode_distances']))),
                'total_distance': run_results['episode_distances'],
                'total_reward': run_results['episode_rewards'],
                'loss': run_results['episode_losses'],
                'algorithm': algorithm,
                'run': run
            }
            algorithm_training_data.append(pd.DataFrame(episode_data))

        # Aggregate results for this algorithm
        results[algorithm] = {
            'avg_distance': np.mean(algorithm_results['distances']),
            'std_distance': np.std(algorithm_results['distances']),
            'avg_reward': np.mean(algorithm_results['rewards']),
            'std_reward': np.std(algorithm_results['rewards']),
            'avg_gap': np.mean(algorithm_results['gaps']),
            'std_gap': np.std(algorithm_results['gaps']),
            'best_distance': np.min(algorithm_results['distances']),
            'worst_distance': np.max(algorithm_results['distances']),
            'raw_results': algorithm_results
        }

        # Combine training data
        training_data[algorithm] = pd.concat(algorithm_training_data, ignore_index=True)

        logger.info(
            f"  {algorithm} completed. Avg distance: {results[algorithm]['avg_distance']:.4f} ± {results[algorithm]['std_distance']:.4f}")

    return {
        'results': results,
        'training_data': training_data,
        'experiment_config': {
            'n_cities': n_cities,
            'algorithms': algorithms,
            'n_runs': n_runs,
            'base_seed': base_seed
        }
    }


def create_comparison_plots(comparison_results: Dict[str, Any], output_dir: str = "plots"):
    """Create comparison plots from experiment results."""

    plotter = create_plotter(output_dir)
    results = comparison_results['results']
    training_data = comparison_results['training_data']
    config = comparison_results['experiment_config']

    # 1. Learning curves comparison
    plotter.plot_learning_curves(
        training_data=training_data,
        metric='total_distance',  # episode_distance
        window=100,
        title=f"Learning Curves Comparison - {config['n_cities']} Cities",
        save_name=f"learning_curves_{config['n_cities']}cities"
    )

    # 2. Performance comparison
    plotter.plot_performance_comparison(
        results=results,
        metrics=['avg_distance', 'avg_gap'],
        title=f"Performance Comparison - {config['n_cities']} Cities",
        save_name=f"performance_comparison_{config['n_cities']}cities"
    )

    # 3. Convergence analysis
    plotter.plot_convergence_analysis(
        training_data=training_data,
        title=f"Convergence Analysis - {config['n_cities']} Cities",
        save_name=f"convergence_analysis_{config['n_cities']}cities"
    )

    # 4. Paper-quality comprehensive figure
    # Calculate optimal distances (dummy values for now)
    optimal_distances = {config['n_cities']: 3.0}  # Would be calculated from actual optimal solutions

    plotter.create_paper_figure(
        training_data=training_data,
        evaluation_results=results,
        optimal_distances=optimal_distances,
        title=f"TSP Reinforcement Learning Results - {config['n_cities']} Cities",
        save_name=f"paper_figure_{config['n_cities']}cities"
    )


def save_comparison_results(results: Dict[str, Any], output_dir: str = "results"):
    """Save comparison results to files."""

    os.makedirs(output_dir, exist_ok=True)

    config = results['experiment_config']
    filename_base = f"comparison_{config['n_cities']}cities_{len(config['algorithms'])}algorithms"

    # Save summary results
    summary_data = []
    for algorithm, result in results['results'].items():
        summary_data.append({
            'Algorithm': algorithm,
            'Avg Distance': f"{result['avg_distance']:.4f} ± {result['std_distance']:.4f}",
            'Avg Gap (%)': f"{result['avg_gap']:.2f} ± {result['std_gap']:.2f}",
            'Best Distance': f"{result['best_distance']:.4f}",
            'Worst Distance': f"{result['worst_distance']:.4f}"
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, f"{filename_base}_summary.csv"), index=False)

    # Save detailed results
    with open(os.path.join(output_dir, f"{filename_base}_detailed.json"), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for alg, res in results['results'].items():
            json_results[alg] = {k: v.tolist() if isinstance(v, np.ndarray) else v
                                 for k, v in res.items() if k != 'raw_results'}

        json.dump({
            'results': json_results,
            'config': config
        }, f, indent=2)

    # Save training data
    for algorithm, data in results['training_data'].items():
        data.to_csv(os.path.join(output_dir, f"{filename_base}_{algorithm}_training.csv"), index=False)

    print(f"Results saved to {output_dir}/")
    print(f"Summary table saved as: {filename_base}_summary.csv")


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description='Compare TSP RL algorithms')
    parser.add_argument('--cities', type=int, default=10,
                        help='Number of cities')
    parser.add_argument('--algorithms', nargs='+',
                        default=['DQN_Basic', 'REINFORCE', 'DQN_LSTM', 'ActorCritic', 'DQN_Optimal'],
                        help='Algorithms to compare')
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of independent runs per algorithm')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                        help='Output directory for results')
    parser.add_argument('--plot', action='store_true',
                        help='Generate comparison plots')
    parser.add_argument('--tsplib', action='store_true',
                        help='Use TSPLIB dataset (default: False)')
    parser.add_argument('--libname', type=str, default='False',
                        help='berlin52')

    args = parser.parse_args()

    print(f"Starting comparison experiment:")
    print(f"  Cities: {args.cities}")
    print(f"  Algorithms: {args.algorithms}")
    print(f"  Runs per algorithm: {args.runs}")
    print(f"  Base seed: {args.seed}")

    # Run comparison
    results = run_comparison_experiment(
        n_cities=args.cities,
        algorithms=args.algorithms,
        n_runs=args.runs,
        base_seed=args.seed,
        is_tsplib=args.tsplib,
        libname=args.libname
    )

    # Save results
    save_comparison_results(results, args.output_dir)

    # Create plots if requested
    if args.plot:
        create_comparison_plots(results, os.path.join(args.output_dir, "plots"))

    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS SUMMARY")
    print("=" * 80)

    for algorithm, result in results['results'].items():
        print(f"{algorithm:20s}: {result['avg_distance']:.4f} ± {result['std_distance']:.4f} "
              f"(gap: {result['avg_gap']:.2f}% ± {result['std_gap']:.2f}%)")

    print("=" * 80)
    print(f"Results saved to: {args.output_dir}/")



def create_agent(config: TSPConfig, device: str):
    """Create agent based on configuration."""

    agent_params = {
        'n_cities': config.n_cities,
        'lr': config.lr,
        'gamma': config.gamma,
        'device': device,
        'seed': config.seed,
        'save_model_diagram': True  # Disable auto diagram generation, will be called manually
    }

    if config.agent_type == "DQN_Basic":
        agent_params.update({
            'epsilon': config.epsilon,
            'epsilon_decay': config.epsilon_decay,
            'epsilon_min': config.epsilon_min,
            'buffer_size': config.buffer_size,
            'batch_size': config.batch_size,
            'target_update_freq': config.target_update_freq,
            'hidden_sizes': config.hidden_sizes
        })
        return DQNBasic(**agent_params)

    elif config.agent_type == "REINFORCE":
        agent_params.update({
            'hidden_sizes': config.hidden_sizes
        })
        return REINFORCE(**agent_params)

    elif config.agent_type == "DQN_LSTM":
        agent_params.update({
            'epsilon': config.epsilon,
            'epsilon_decay': config.epsilon_decay,
            'epsilon_min': config.epsilon_min,
            'buffer_size': config.buffer_size,
            'batch_size': config.batch_size,
            'target_update_freq': config.target_update_freq,
            'max_sequence_length': config.n_cities
        })
        return DQNLSTM(**agent_params)

    elif config.agent_type == "DQN_OrderEmbedding":
        agent_params.update({
            'epsilon': config.epsilon,
            'epsilon_decay': config.epsilon_decay,
            'epsilon_min': config.epsilon_min,
            'buffer_size': config.buffer_size,
            'batch_size': config.batch_size,
            'target_update_freq': config.target_update_freq,
            'hidden_sizes': config.hidden_sizes
        })
        return DQNOrderEmbedding(**agent_params)

    elif config.agent_type == "ActorCritic":
        agent_params.update({
            'hidden_sizes': config.hidden_sizes
        })
        return ActorCritic(**agent_params)

    elif config.agent_type == "DQN_Optimal":
        agent_params.update({
            'epsilon': config.epsilon,
            'epsilon_decay': config.epsilon_decay,
            'epsilon_min': config.epsilon_min,
            'buffer_size': config.buffer_size,
            'batch_size': config.batch_size,
            'target_update_freq': config.target_update_freq,
            'hidden_sizes': config.hidden_sizes
        })
        return DQNOptimal(**agent_params)

    else:
        raise ValueError(f"Unknown agent type: {config.agent_type}")


def create_environment(config: TSPConfig):

    return TSPEnvironment(n_cities=config.n_cities, seed=config.seed, use_tsplib=config.use_tsplib,
                          tsplib_name=config.tsplib_name)


def train_agent(config: TSPConfig) -> Tuple[Any, Dict[str, Any]]:

    set_global_seed(config.seed)

    # Create unified timestamp for this training run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup logging, data recording, and model management with unified timestamp
    logger = setup_logging(level="INFO")
    data_recorder = create_data_recorder(config.experiment_name, run_timestamp=run_timestamp)
    data_recorder.set_config(config.to_dict())
    model_manager = create_model_manager(run_timestamp=run_timestamp)
    model_dir = model_manager.get_model_dir()

    # Create environment and agent
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    env = create_environment(config)
    config.n_cities = env.n_cities  # for tsplib
    agent = create_agent(config, device)

    # Generate model architecture diagrams with unified timestamp
    if hasattr(agent, 'save_architecture_diagram'):
        agent.save_architecture_diagram(config.to_dict(), out_put_dir=model_dir)

    logger.info(f"Starting training: {config.agent_type} on {config.n_cities} cities")
    logger.log_experiment_start(config.to_dict())

    # Training loop
    episode_rewards = []
    episode_distances = []
    episode_losses = []

    for episode in range(config.n_episodes):
        state = env.reset()
        total_reward = 0
        total_loss = 0
        step_count = 0

        # For episodic algorithms (REINFORCE, Actor-Critic)
        if hasattr(agent, 'episode_rewards'):
            agent.episode_rewards = []
        if hasattr(agent, 'episode_log_probs'):
            agent.episode_log_probs = []
        if hasattr(agent, 'episode_states'):
            agent.episode_states = []

        while not env.done and step_count < config.max_steps_per_episode:
            # Select action
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)

            # Take action
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            # Store reward for episodic algorithms
            if hasattr(agent, 'store_reward'):
                agent.store_reward(reward)

            # Update agent (for online algorithms)
            if hasattr(agent, 'update') and config.agent_type not in ["REINFORCE", "ActorCritic"]:
                loss = agent.update(state, action, reward, next_state, done)
                total_loss += loss

            state = next_state
            step_count += 1

        # Update episodic algorithms at end of episode
        if config.agent_type in ["REINFORCE", "ActorCritic"]:
            # policy gradient needs to wait for the whole episode, in order to calculate G
            if hasattr(agent, 'update'):
                total_loss = agent.update()

        # Record episode data
        episode_rewards.append(total_reward)
        episode_distances.append(info['total_distance'])
        episode_losses.append(total_loss)

        data_recorder.record_episode(
            episode=episode,
            total_reward=total_reward,
            total_distance=info['total_distance'],
            episode_length=step_count,
            path=info['path'],
            final_loss=total_loss,
            agent_name=config.agent_type
        )

        # Log progress
        if episode % config.log_frequency == 0:
            avg_reward = np.mean(episode_rewards[-config.log_frequency:])
            avg_distance = np.mean(episode_distances[-config.log_frequency:])
            avg_loss = np.mean(episode_losses[-config.log_frequency:])

            metrics = {
                'type': config.agent_type,
                'avg_reward': avg_reward,
                'avg_distance': avg_distance,
                'avg_loss': avg_loss,
                'epsilon': getattr(agent, 'epsilon', None)
            }

            logger.log_episode(episode, metrics)

        # Save model using intelligent model manager
        if episode % config.save_frequency == 0 and episode > 0:
            model_manager.save_model(
                agent=agent,
                model_path="",  # Model manager will determine the path
                episode=episode,
                performance_metric=info['total_distance'],
                experiment_name=config.experiment_name,
                is_final=False
            )

    # Evaluation
    logger.info("Starting final evaluation...")
    eval_results = evaluate_agent(agent, env, config.eval_episodes, config.seed)

    # Final model save using model manager
    final_model_path = model_manager.save_model(
        agent=agent,
        model_path="",  # Model manager will determine the path
        episode=config.n_episodes,
        performance_metric=eval_results['avg_distance'],
        experiment_name=config.experiment_name,
        is_final=True
    )

    # Record evaluation results
    # optimal_path, optimal_distance = env.get_optimal_solution()
    optimal_path = env.optimal_path
    optimal_distance = env.optimal_distance
    # gap_percentage = ((eval_results['avg_distance'] - optimal_distance) / optimal_distance) * 100

    gap_percentage = eval_results['avg_gap_percentage']

    data_recorder.record_evaluation(
        agent_name=config.agent_type,
        test_episodes=config.eval_episodes,
        avg_distance=eval_results['avg_distance'],
        std_distance=eval_results['std_distance'],
        avg_reward=eval_results['avg_reward'],
        std_reward=eval_results['std_reward'],
        optimal_distance=optimal_distance,
        gap_percentage=gap_percentage
    )

    # Save all data
    data_recorder.save_all_data()

    # Prepare results
    results = {
        'episode_rewards': episode_rewards,
        'episode_distances': episode_distances,
        'episode_losses': episode_losses,
        'evaluation': eval_results,
        'optimal_distance': optimal_distance,
        'gap_percentage': gap_percentage,
        'config': config.to_dict()
    }

    logger.log_experiment_end(results)
    logger.info(f"Training completed. Gap from optimal: {gap_percentage:.2f}%")

    # Log model management summary
    storage_summary = model_manager.get_storage_summary()
    logger.info(f"Model storage: {storage_summary['total_files']} files, "
                f"{storage_summary['total_size_mb']:.1f} MB")

    return agent, results


def evaluate_agent(agent, env, n_episodes: int, seed: int) -> Dict[str, Any]:
    """
    Evaluate trained agent.

    Args:
        agent: Trained agent
        env: Environment
        n_episodes: Number of evaluation episodes
        seed: Random seed

    Returns:
        Evaluation results including gap percentages for each episode
    """
    agent.set_eval_mode()

    episode_rewards = []
    episode_distances = []
    gap_percentages = []

    for episode in range(n_episodes):
        eval_seed = seed + episode + 10000  # Different seed for evaluation
        state = env.reset(eval_seed)
        total_reward = 0

        while not env.done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state

        # Calculate optimal distance for this episode's coordinates
        _, optimal_distance = env.get_optimal_solution()
        episode_distance = info['total_distance']
        # Calculate gap percentage for this episode
        gap_percentage = ((episode_distance - optimal_distance) / optimal_distance) * 100 if optimal_distance > 0 else 0

        episode_rewards.append(total_reward)
        episode_distances.append(episode_distance)
        gap_percentages.append(gap_percentage)

    agent.set_train_mode()

    return {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_distance': np.mean(episode_distances),
        'std_distance': np.std(episode_distances),
        'best_distance': np.min(episode_distances),
        'worst_distance': np.max(episode_distances),
        'gap_percentages': gap_percentages,  # Array of gap percentages
        'avg_gap_percentage': np.mean(gap_percentages),
        'std_gap_percentage': np.std(gap_percentages)
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train TSP RL agents')
    parser.add_argument('--agent', type=str, default='DQN_Basic',
                        choices=['DQN_Basic', 'REINFORCE', 'DQN_LSTM',
                                 'DQN_OrderEmbedding', 'ActorCritic', 'DQN_Optimal'],
                        help='Agent type to train')
    parser.add_argument('--cities', type=int, default=10,
                        help='Number of cities')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config JSON file')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--tsplib', action='store_true',
                        help='Use TSPLIB dataset (default: False)')
    parser.add_argument('--libname', type=str, default='False',
                        help='berlin52')

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Create configuration
    if args.config:
        config = TSPConfig.from_json(args.config)
    else:
        custom_params = {'seed': args.seed, 'device': device,
                         'use_tsplib': args.tsplib, 'tsplib_name': args.libname}
        if args.episodes:
            custom_params['n_episodes'] = args.episodes

        config = create_experiment_config(
            agent_type=args.agent,
            n_cities=args.cities,
            custom_params=custom_params,
        )

    print(f"Training {config.agent_type} on {config.n_cities} cities")
    print(f"Device: {device}")
    print(f"Episodes: {config.n_episodes}")

    # Train agent
    agent, results = train_agent(config)

    print("\nTraining completed!")
    print(f"Final average distance: {results['evaluation']['avg_distance']:.4f}")
    print(f"Gap from optimal: {results['gap_percentage']:.2f}%")


class DQNBasic(BaseAgent):
    """
    Basic DQN agent for TSP (version 1.1).
    Uses only current city one-hot encoding as input.
    """

    def __init__(self,
                 n_cities: int,
                 lr: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.1,
                 buffer_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 100,
                 hidden_sizes: List[int] = [256, 256, 256],
                 device: str = 'cpu',
                 seed: Optional[int] = None,
                 save_model_diagram: bool = True):
        """
        Initialize DQN Basic agent.

        Args:
            n_cities: Number of cities
            lr: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
            buffer_size: Replay buffer size
            batch_size: Batch size for training
            target_update_freq: Frequency to update target network
            hidden_sizes: Hidden layer sizes for MLP
            device: Device to use
            seed: Random seed
        """
        super().__init__(n_cities, lr, device, seed, save_model_diagram)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Input size is n_cities (one-hot encoding)
        input_size = n_cities

        # Create Q-networks
        self.q_network = MLP(input_size, hidden_sizes, n_cities).to(device)
        self.target_network = MLP(input_size, hidden_sizes, n_cities).to(device)

        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, seed)

        # Training step counter
        self.step_count = 0

        self.logger.info(f"Initialized DQN Basic with {sum(p.numel() for p in self.q_network.parameters())} parameters")

        # Save architecture diagram
        config_dict = {
            'n_cities': n_cities,
            'lr': lr,
            'gamma': gamma,
            'epsilon': epsilon,
            'epsilon_decay': epsilon_decay,
            'epsilon_min': epsilon_min,
            'buffer_size': buffer_size,
            'batch_size': batch_size,
            'target_update_freq': target_update_freq,
            'hidden_sizes': hidden_sizes,
            'device': device,
            'seed': seed
        }
        # self.save_architecture_diagram(config_dict)

    def select_action(self, state: Dict[str, Any], valid_actions: List[int]) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state containing 'current_city_onehot'
            valid_actions: List of valid actions

        Returns:
            Selected action
        """
        if random.random() < self.epsilon:
            # Random action from valid actions
            return random.choice(valid_actions)
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state['visited_mask']).unsqueeze(0).to(self.device)
                # state_tensor = torch.FloatTensor(state['current_city_onehot']).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor).cpu().numpy()[0]

                # Mask invalid actions
                masked_q_values = q_values.copy()
                for i in range(self.n_cities):
                    if i not in valid_actions:
                        masked_q_values[i] = -float('inf')

                return np.argmax(masked_q_values)

    def update(self, state: Dict[str, Any], action: int, reward: float,
               next_state: Dict[str, Any], done: bool) -> float:
        """
        Update Q-network using experience replay.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done

        Returns:
            Loss value
        """
        # Store transition in replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)

        # Don't train until we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)

        # Separate batch elements
        states = [b[0] for b in batch]
        actions = [b[1] for b in batch]
        rewards = [b[2] for b in batch]
        next_states = [b[3] for b in batch]
        dones = [b[4] for b in batch]

        # Convert to tensors
        state_batch = torch.FloatTensor([s['current_city_onehot'] for s in states]).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor([s['current_city_onehot'] for s in next_states]).to(self.device)
        done_batch = torch.BoolTensor(dones).to(self.device)

        # Current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))

        # Next Q values from target network
        next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save_model(self, filepath: str) -> None:
        """Save model parameters."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load model parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.logger.info(f"Model loaded from {filepath}")

    def set_eval_mode(self) -> None:
        """Set agent to evaluation mode (no exploration)."""
        self.epsilon = 0.0
        self.q_network.eval()

    def set_train_mode(self) -> None:
        """Set agent to training mode."""
        self.q_network.train()

    def get_q_values(self, state: Dict[str, Any]) -> np.ndarray:
        """Get Q-values for current state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state['current_city_onehot']).unsqueeze(0).to(self.device)
            return self.q_network(state_tensor).cpu().numpy()[0]


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

    # def _prepare_tsplib_data(self):
    #     """Download and prepare TSPLIB data if not exists."""
    #     if not os.path.exists(self.tsplib_path):
    #         os.makedirs(self.tsplib_path)
    #
    #     # List of common EUC_2D problems to download
    #     euc_2d_problems = [
    #         'berlin52', 'eil51', 'eil76', 'eil101', 'ch130', 'ch150',
    #         'a280', 'pr76', 'rat195', 'kroA100', 'kroB100', 'kroC100',
    #         'kroD100', 'kroE100', 'rd100', 'st70'
    #     ]
    #
    #     base_url = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/"
    #
    #     for problem in euc_2d_problems:
    #         tsp_file = f"{problem}.tsp"
    #         opt_file = f"{problem}.opt.tour"
    #
    #         tsp_path = os.path.join(self.tsplib_path, tsp_file)
    #         opt_path = os.path.join(self.tsplib_path, opt_file)
    #
    #         # Download .tsp file
    #         if not os.path.exists(tsp_path):
    #             try:
    #                 u = f"{base_url}{tsp_file}"
    #                 urllib.request.urlretrieve(u, tsp_path)
    #                 print(f"Downloaded {tsp_file}")
    #             except Exception as e:
    #                 print(f"Failed to download {u} {tsp_file}: {e}")
    #
    #         # Download .opt.tour file
    #         if not os.path.exists(opt_path):
    #             try:
    #                 uu = f"{base_url}{opt_file}"
    #                 urllib.request.urlretrieve(uu, opt_path)
    #                 print(f"Downloaded {opt_file}")
    #             except Exception as e:
    #                 print(f"Failed to download {uu}   {opt_file}: {e}")

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
            'current_city_onehot': current_city_onehot,  # basic
            'sequence_onehot': sequence_onehot,  # lstm
            'sequence_length': sequence_length,  # lstm
            'visited_mask': visited_mask,  # basic
            'order_embedding': order_embedding,  # order_embedding
            'distances_from_current': distances_from_current,
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




--------model_visualizer.py
import traceback

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import os
from typing import Dict, Any, List, Tuple, Optional
import json
import graphviz
from graphviz import Digraph

from datetime import datetime


class ModelVisualizer:
    """
    Model structure visualization for TSP RL agents.
    Creates network architecture diagrams with objective and loss functions.
    """

    def __init__(self, base_output_dir: str = "models"):
        """
        Initialize model visualizer.

        Args:
            base_output_dir: Base directory to save diagrams

        """
        self.output_dir = os.path.join(base_output_dir, "diagrams/")
        os.makedirs(self.output_dir, exist_ok=True)

        # Color scheme for different components
        self.colors = {
            'input': '#E8F4FD',
            'hidden': '#B3D9FF',
            'output': '#4A90E2',
            'lstm': '#FF9F43',
            'embedding': '#26C281',
            'critic': '#8E44AD',
            'actor': '#E74C3C',
            'loss': '#F39C12',
            'objective': '#2ECC71'
        }

    def save_model_architecture(self,
                                agent,
                                agent_name: str,
                                config: Dict[str, Any],
                                save_torchviz: bool = True) -> str:
        """
        Save complete model architecture diagram.

        Args:
            agent: The RL agent
            agent_name: Name of the agent
            config: Configuration dictionary
            save_torchviz: Whether to save torchviz diagram (requires torchviz)

        Returns:
            Path to saved network diagram
        """
        base_filename = f"{agent_name}"

        # Create network architecture diagram (without formulas)
        network_path = self._create_network_diagram(agent, agent_name, config, base_filename)

        # Create separate formulas diagram
        formula_path = self._create_formula_diagram(agent, agent_name, config, base_filename)

        # Save model summary
        self._save_model_summary(agent, agent_name, config, base_filename)

        # Try to create torchviz diagram if available
        if save_torchviz:
            try:
                self._create_torchviz_diagram(agent, agent_name, base_filename)
            except ImportError:
                print("Warning: torchviz not available. Install with: pip install torchviz")
            except Exception as e:
                print(f"Warning: Could not create torchviz diagram: {e} {traceback.format_exc()}")

        return network_path

    def _create_network_diagram(self,
                                agent,
                                agent_name: str,
                                config: Dict[str, Any],
                                base_filename: str) -> str:
        """Create network architecture diagram (without formulas)."""

        # Determine agent type and create appropriate diagram
        if hasattr(agent, 'q_network'):
            if hasattr(agent.q_network, 'lstm'):
                # LSTM-based DQN
                return self._draw_lstm_dqn_network(agent, agent_name, config, base_filename)
            else:
                # Regular DQN
                return self._draw_dqn_network(agent, agent_name, config, base_filename)
        elif hasattr(agent, 'actor') and hasattr(agent, 'critic'):
            # Actor-Critic
            return self._draw_actor_critic_network(agent, agent_name, config, base_filename)
        elif hasattr(agent, 'policy_network'):
            # REINFORCE
            return self._draw_policy_network(agent, agent_name, config, base_filename)
        else:
            # Generic diagram
            return self._draw_generic_network(agent, agent_name, config, base_filename)

    def _create_formula_diagram(self,
                                agent,
                                agent_name: str,
                                config: Dict[str, Any],
                                base_filename: str) -> str:
        """Create separate formula diagram as markdown LaTeX."""

        # 构建markdown内容
        markdown_content = []

        # 添加标题
        markdown_content.append(f"# {agent_name} - Objective and Loss Functions\n")

        # 添加配置信息
        config_text = f"**Configuration:** Cities: {config['n_cities']} | LR: {config.get('lr', 'N/A')} | Seed: {config.get('seed', 'N/A')}\n"
        markdown_content.append(config_text)

        # 根据agent类型添加相应的公式
        if hasattr(agent, 'q_network'):
            markdown_content.extend(self._generate_dqn_formulas(agent_name, config))
        elif hasattr(agent, 'actor') and hasattr(agent, 'critic'):
            markdown_content.extend(self._generate_actor_critic_formulas(agent_name, config))
        elif hasattr(agent, 'policy_network'):
            markdown_content.extend(self._generate_reinforce_formulas(agent_name, config))
        else:
            markdown_content.extend(self._generate_generic_formulas(agent_name, config))

        # 保存到markdown文件
        formula_path = os.path.join(self.output_dir, f"{base_filename}_formulas.md")
        with open(formula_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(markdown_content))

        return formula_path

    def _generate_dqn_formulas(self, agent_name: str, config: Dict[str, Any]) -> List[str]:
        """Generate DQN objective and loss formulas in LaTeX format."""

        formulas = []

        formulas.append("## Objective Function")
        formulas.append("$$\\text{Objective: } \\max Q(s,a)$$")
        formulas.append("*Learn optimal action-value function*\n")

        formulas.append("## Loss Function")
        formulas.append("$$\\text{Loss: } \\mathcal{L} = \\mathbb{E}[(Q(s,a) - \\text{target})^2]$$")
        formulas.append("*Mean Squared Error*\n")

        formulas.append("## Target Calculation")
        formulas.append("$$\\text{target} = r + \\gamma \\max_{a'} Q(s', a')$$")
        formulas.append("*Bellman equation*\n")

        return formulas

    def _generate_reinforce_formulas(self, agent_name: str, config: Dict[str, Any]) -> List[str]:
        """Generate REINFORCE objective and loss formulas in LaTeX format."""

        formulas = []

        formulas.append("## Objective Function")
        formulas.append("$$\\text{Objective: } \\max \\mathbb{E}[R]$$")
        formulas.append("*Maximize expected return using policy gradient*\n")

        formulas.append("## Loss Function")
        formulas.append("$$\\text{Loss: } \\mathcal{L} = -\\log \\pi(a|s) \\cdot G$$")
        formulas.append("*Negative log-likelihood weighted by return*\n")

        formulas.append("## Return Calculation")
        formulas.append("$$G = \\sum_{t=0}^{T} \\gamma^t r_{t+1}$$")
        formulas.append("*Discounted future rewards*\n")

        return formulas

    def _generate_actor_critic_formulas(self, agent_name: str, config: Dict[str, Any]) -> List[str]:
        """Generate Actor-Critic objective and loss formulas in LaTeX format."""

        formulas = []

        formulas.append("## Actor Objective")
        formulas.append("$$\\text{Actor Objective: } \\max \\mathbb{E}[\\log \\pi(a|s) \\cdot A(s,a)]$$")
        formulas.append("*Maximize expected advantage-weighted log-probability*\n")

        formulas.append("## Critic Objective")
        formulas.append("$$\\text{Critic Objective: } \\min \\mathbb{E}[(V(s) - R)^2]$$")
        formulas.append("*Minimize value function prediction error*\n")

        formulas.append("## Combined Loss")
        formulas.append(
            "$$\\mathcal{L} = \\mathcal{L}_{\\text{actor}} + \\alpha \\mathcal{L}_{\\text{critic}} + \\beta \\mathcal{H}$$")
        formulas.append("*Actor loss + Value loss + Entropy regularization*\n")

        formulas.append("## Advantage Function")
        formulas.append("$$A(s,a) = R - V(s)$$")
        formulas.append("*Advantage as return minus baseline*\n")

        return formulas

    def _generate_generic_formulas(self, agent_name: str, config: Dict[str, Any]) -> List[str]:
        """Generate generic formulas for unknown agent types."""

        formulas = []

        formulas.append("## Agent Information")
        formulas.append(f"**Agent Type:** {agent_name}")
        formulas.append("**Objective and Loss Functions:** Agent-specific formulas")
        formulas.append("*Please refer to the specific agent documentation for detailed mathematical formulations*\n")

        return formulas

    def _draw_dqn_network(self, agent, agent_name: str, config: Dict[str, Any], base_filename: str) -> str:
        """使用Graphviz绘制专业的DQN网络图"""

        dot = Digraph(comment=f'{agent_name}')
        dot.attr(rankdir='LR', size='12,8')
        dot.attr('node', shape='box', style='filled', color='lightblue')

        # 输入层
        dot.node('input', f'Input\n{config.get("state_size", "State")}')

        # 隐藏层
        layers = list(agent.q_network.network.children())
        prev_node = 'input'

        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                node_name = f'fc{i}'
                dot.node(node_name, f'Linear\n{layer.in_features}→{layer.out_features}')
                dot.edge(prev_node, node_name)
                prev_node = node_name
            elif isinstance(layer, nn.ReLU):
                node_name = f'relu{i}'
                dot.node(node_name, 'ReLU', shape='ellipse', color='lightgreen')
                dot.edge(prev_node, node_name)
                prev_node = node_name

        # 输出层
        dot.node('output', f'Q-Values\n{config["n_cities"]}')
        dot.edge(prev_node, 'output')

        # 保存
        output_path = os.path.join(self.output_dir, f"{base_filename}_network")
        path = dot.render(output_path, format='png', cleanup=True)

        return path

    def _draw_lstm_dqn_network(self, agent, agent_name: str, config: Dict[str, Any], base_filename: str) -> str:
        """Draw LSTM-DQN network architecture diagram using Graphviz."""

        dot = Digraph(comment=f'{agent_name}')
        dot.attr(rankdir='LR', size='16,8')
        dot.attr('node', shape='box', style='filled')

        # Get network components
        lstm = agent.q_network.lstm
        mlp_head = agent.q_network.mlp_head

        # Sequence input
        dot.node('input', f'Sequence Input\n[T, {config["n_cities"]}]', fillcolor=self.colors['input'])

        # LSTM section
        lstm_info = f"LSTM\n{lstm.num_layers} layers\nHidden: {lstm.hidden_size}"
        dot.node('lstm', lstm_info, fillcolor=self.colors['lstm'])
        dot.edge('input', 'lstm')

        # Hidden state
        dot.node('hidden', 'h_n', shape='ellipse', fillcolor=self.colors['hidden'])
        dot.edge('lstm', 'hidden')

        # MLP head
        prev_node = 'hidden'
        mlp_layers = list(mlp_head.children())
        for i, layer in enumerate(mlp_layers):
            if isinstance(layer, nn.Linear):
                node_name = f'fc{i}'
                dot.node(node_name, f"Head{i // 2 + 1}\n{layer.in_features}→{layer.out_features}",
                         fillcolor=self.colors['hidden'])
                dot.edge(prev_node, node_name)
                prev_node = node_name
            elif isinstance(layer, nn.ReLU):
                node_name = f'relu{i}'
                dot.node(node_name, 'ReLU', shape='ellipse', fillcolor='lightgray')
                dot.edge(prev_node, node_name)
                prev_node = node_name
            elif isinstance(layer, nn.Dropout):
                node_name = f'dropout{i}'
                dot.node(node_name, f"Dropout\np={layer.p}", shape='ellipse', fillcolor='orange')
                dot.edge(prev_node, node_name)
                prev_node = node_name

        # Output section
        dot.node('output', f"Q-Values\n[{config['n_cities']}]", fillcolor=self.colors['output'])
        dot.edge(prev_node, 'output')

        # Save diagram
        output_path = os.path.join(self.output_dir, f"{base_filename}_network")
        path = dot.render(output_path, format='png', cleanup=True)

        return path

    def _draw_actor_critic_network(self, agent, agent_name: str, config: Dict[str, Any], base_filename: str) -> str:
        """Draw Actor-Critic network architecture diagram using Graphviz."""

        dot = Digraph(comment=f'{agent_name}')
        dot.attr(rankdir='LR', size='14,10')
        dot.attr('node', shape='box', style='filled')

        # Input section
        input_desc = f"Input\n[{config['n_cities']}]"
        if "OrderEmbedding" in agent_name:
            input_desc = f"Current + Order\n[{2 * config['n_cities']}]\nOne-hot + embedding"
        elif "Optimal" in agent_name:
            input_desc = f"Full State\n[{4 * config['n_cities'] + 1}]\nOne-hot + visited +\ndistances + progress"

        dot.node('input', input_desc, fillcolor=self.colors['input'])

        # Actor branch
        prev_node = 'input'
        dot.node('actor_start', 'Actor', fillcolor=self.colors['actor'])
        dot.edge('input', 'actor_start')
        prev_node = 'actor_start'

        actor_layers = list(agent.actor.network.children())
        for i, layer in enumerate(actor_layers):
            if isinstance(layer, nn.Linear):
                node_name = f'actor_fc{i}'
                dot.node(node_name, f"Actor\n{layer.in_features}→{layer.out_features}",
                         fillcolor=self.colors['actor'])
                dot.edge(prev_node, node_name)
                prev_node = node_name
            elif isinstance(layer, nn.ReLU):
                node_name = f'actor_relu{i}'
                dot.node(node_name, 'ReLU', shape='ellipse', fillcolor='lightgray')
                dot.edge(prev_node, node_name)
                prev_node = node_name

        dot.node('actor_output', f"π(a|s)\n[{config['n_cities']}]", fillcolor=self.colors['actor'])
        dot.edge(prev_node, 'actor_output')

        # Critic branch
        prev_node = 'input'
        dot.node('critic_start', 'Critic', fillcolor=self.colors['critic'])
        dot.edge('input', 'critic_start')
        prev_node = 'critic_start'

        critic_layers = list(agent.critic.network.children())
        for i, layer in enumerate(critic_layers):
            if isinstance(layer, nn.Linear):
                node_name = f'critic_fc{i}'
                dot.node(node_name, f"Critic\n{layer.in_features}→{layer.out_features}",
                         fillcolor=self.colors['critic'])
                dot.edge(prev_node, node_name)
                prev_node = node_name
            elif isinstance(layer, nn.ReLU):
                node_name = f'critic_relu{i}'
                dot.node(node_name, 'ReLU', shape='ellipse', fillcolor='lightgray')
                dot.edge(prev_node, node_name)
                prev_node = node_name

        dot.node('critic_output', "V(s)\n[1]", fillcolor=self.colors['critic'])
        dot.edge(prev_node, 'critic_output')

        # Save diagram
        output_path = os.path.join(self.output_dir, f"{base_filename}_network")
        path = dot.render(output_path, format='png', cleanup=True)

        return path

    def _draw_generic_network(self, agent, agent_name: str, config: Dict[str, Any], base_filename: str) -> str:
        """Draw generic network architecture diagram using Graphviz."""

        dot = Digraph(comment=f'{agent_name}')
        dot.attr(rankdir='LR', size='12,8')
        dot.attr('node', shape='box', style='filled')

        # Input section
        input_desc = f"Input\n[{config.get('state_size', 'Unknown')}]"
        dot.node('input', input_desc, fillcolor=self.colors['input'])

        # Generic network structure
        total_params = 0
        if hasattr(agent, 'q_network'):
            layers = list(agent.q_network.children())
            total_params += sum(p.numel() for p in agent.q_network.parameters())
        elif hasattr(agent, 'policy_network'):
            layers = list(agent.policy_network.children())
            total_params += sum(p.numel() for p in agent.policy_network.parameters())
        elif hasattr(agent, 'actor'):
            layers = list(agent.actor.network.children())
            total_params += sum(p.numel() for p in agent.actor.parameters())
            if hasattr(agent, 'critic'):
                total_params += sum(p.numel() for p in agent.critic.parameters())
        else:
            layers = []

        prev_node = 'input'
        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                node_name = f'fc{i}'
                dot.node(node_name, f"Linear\n{layer.in_features}→{layer.out_features}",
                         fillcolor=self.colors['hidden'])
                dot.edge(prev_node, node_name)
                prev_node = node_name
            elif isinstance(layer, nn.ReLU):
                node_name = f'relu{i}'
                dot.node(node_name, 'ReLU', shape='ellipse', fillcolor='lightgray')
                dot.edge(prev_node, node_name)
                prev_node = node_name
            elif isinstance(layer, nn.Dropout):
                node_name = f'dropout{i}'
                dot.node(node_name, f"Dropout\np={layer.p}", shape='ellipse', fillcolor='orange')
                dot.edge(prev_node, node_name)
                prev_node = node_name

        # Output section
        output_desc = f"Output\n[{config.get('n_cities', 'Unknown')}]"
        dot.node('output', output_desc, fillcolor=self.colors['output'])
        dot.edge(prev_node, 'output')

        # Add parameter count as a note
        dot.node('params', f"Total Parameters: {total_params:,}", shape='note',
                 fillcolor='lightgray')
        dot.edge('output', 'params', style='dashed')

        # Save diagram
        output_path = os.path.join(self.output_dir, f"{base_filename}_network")
        path = dot.render(output_path, format='png', cleanup=True)

        return path

    def _draw_policy_network(self, agent, agent_name: str, config: Dict[str, Any], base_filename: str) -> str:
        """使用Graphviz绘制专业的REINFORCE策略网络图"""

        dot = Digraph(comment=f'{agent_name}')
        dot.attr(rankdir='LR', size='12,8')
        dot.attr('node', shape='box', style='filled', color='lightblue')

        # 输入层
        dot.node('input', f'Input\n{config.get("state_size", "State")}')

        # 隐藏层
        layers = list(agent.policy_network.network.children())
        prev_node = 'input'

        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                node_name = f'fc{i}'
                dot.node(node_name, f'Linear\n{layer.in_features}→{layer.out_features}')
                dot.edge(prev_node, node_name)
                prev_node = node_name
            elif isinstance(layer, nn.ReLU):
                node_name = f'relu{i}'
                dot.node(node_name, 'ReLU', shape='ellipse', color='lightgreen')
                dot.edge(prev_node, node_name)
                prev_node = node_name

        # 输出层
        dot.node('output', f'Policy π(a|s)\n{config["n_cities"]}')
        dot.edge(prev_node, 'output')

        # 保存
        output_path = os.path.join(self.output_dir, f"{base_filename}_network")
        path = dot.render(output_path, format='png', cleanup=True)

        return path

    def _draw_input_section(self, ax, agent_name: str, config: Dict[str, Any], x_pos: float, node_size: float):
        """Draw input representation section."""

        # Determine input type based on agent
        if "LSTM" in agent_name:
            input_desc = f"Sequence\n[seq_len, {config['n_cities']}]\nOne-hot vectors"
        elif "OrderEmbedding" in agent_name:
            input_desc = f"Current + Order\n[{2 * config['n_cities']}]\nOne-hot + embedding"
        elif "Optimal" in agent_name:
            input_desc = f"Full State\n[{4 * config['n_cities'] + 1}]\nOne-hot + visited +\ndistances + progress"
        else:
            input_desc = f"Current City\n[{config['n_cities']}]\nOne-hot encoding"

        # Draw input box with black lines
        rect = FancyBboxPatch((x_pos - node_size / 2, -node_size / 2),
                              node_size * 1.5, node_size * 2,
                              boxstyle="round,pad=0.1",
                              facecolor='lightblue',
                              edgecolor='black',
                              linewidth=2)
        ax.add_patch(rect)

        ax.text(x_pos + node_size / 4, 0, input_desc,
                ha='center', va='center', fontsize=10, fontweight='bold', color='black')

    def _draw_sequence_input_section(self, ax, config: Dict[str, Any], x_pos: float, node_size: float):
        """Draw sequence input section for LSTM."""

        # Draw sequence of inputs
        for i in range(3):  # Show 3 time steps
            y_offset = (i - 1) * 1.5
            rect = FancyBboxPatch((x_pos - node_size / 4, y_offset - node_size / 4),
                                  node_size / 2, node_size / 2,
                                  boxstyle="round,pad=0.05",
                                  facecolor=self.colors['input'],
                                  edgecolor='black')
            ax.add_patch(rect)

            ax.text(x_pos, y_offset, f"t{i + 1}", ha='center', va='center', fontsize=8)

        # Add label
        ax.text(x_pos, -3, f"Sequence Input\n[T, {config['n_cities']}]",
                ha='center', va='center', fontsize=10, fontweight='bold')

    def _draw_lstm_section(self, ax, lstm: nn.LSTM, x_pos: float, node_size: float):
        """Draw LSTM section."""

        # LSTM cell
        rect = FancyBboxPatch((x_pos - node_size, -node_size),
                              node_size * 2, node_size * 2,
                              boxstyle="round,pad=0.1",
                              facecolor=self.colors['lstm'],
                              edgecolor='black',
                              linewidth=2)
        ax.add_patch(rect)

        # LSTM info
        lstm_info = f"LSTM\n{lstm.num_layers} layers\nHidden: {lstm.hidden_size}"
        ax.text(x_pos, 0, lstm_info, ha='center', va='center',
                fontsize=10, fontweight='bold')

        # Hidden state output
        h_rect = FancyBboxPatch((x_pos - node_size / 4, node_size * 1.5),
                                node_size / 2, node_size / 2,
                                boxstyle="round,pad=0.05",
                                facecolor=self.colors['hidden'],
                                edgecolor='black')
        ax.add_patch(h_rect)
        ax.text(x_pos, node_size * 1.75, "h_n", ha='center', va='center', fontsize=8)

    def _draw_activation_layer(self, ax, activation: str, x_pos: float, node_size: float):
        """Draw activation function."""

        circle = plt.Circle((x_pos, 0), node_size / 4,
                            facecolor='lightgray', edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x_pos, 0, activation, ha='center', va='center', fontsize=8, color='black')

    def _draw_dropout_layer(self, ax, dropout_p: float, x_pos: float, node_size: float):
        """Draw dropout layer."""

        circle = plt.Circle((x_pos, -0.8), node_size / 4,
                            facecolor='orange', edgecolor='black', alpha=0.7)
        ax.add_patch(circle)
        ax.text(x_pos, -0.8, f"Drop\n{dropout_p}", ha='center', va='center', fontsize=7)

    def _draw_output_section(self, ax, output_type: str, n_outputs: int, x_pos: float, node_size: float):
        """Draw output section."""

        rect = FancyBboxPatch((x_pos - node_size / 2, -node_size / 2),
                              node_size * 1.5, node_size * 2,
                              boxstyle="round,pad=0.1",
                              facecolor='lightcoral',
                              edgecolor='black',
                              linewidth=2)
        ax.add_patch(rect)

        output_desc = f"{output_type}\n[{n_outputs}]"
        ax.text(x_pos + node_size / 4, 0, output_desc,
                ha='center', va='center', fontsize=10, fontweight='bold', color='black')

    def _draw_actor_branch(self, ax, actor: nn.Module, x_start: float, y_offset: float, node_size: float,
                           n_actions: int):
        """Draw actor network branch."""

        layers = list(actor.network.children())
        x_pos = x_start

        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                rect = FancyBboxPatch((x_pos - node_size / 2, y_offset - node_size / 2),
                                      node_size, node_size,
                                      boxstyle="round,pad=0.1",
                                      facecolor='lightcoral',
                                      edgecolor='black',
                                      linewidth=2)
                ax.add_patch(rect)

                layer_info = f"Actor\n{layer.in_features}→{layer.out_features}"
                ax.text(x_pos, y_offset, layer_info, ha='center', va='center',
                        fontsize=9, color='black')
                x_pos += 2

        # Policy output
        rect = FancyBboxPatch((x_pos - node_size / 2, y_offset - node_size / 2),
                              node_size * 1.5, node_size,
                              boxstyle="round,pad=0.1",
                              facecolor='lightcoral',
                              edgecolor='black',
                              linewidth=2)
        ax.add_patch(rect)
        ax.text(x_pos + node_size / 4, y_offset, f"π(a|s)\n[{n_actions}]",
                ha='center', va='center', fontsize=10, fontweight='bold', color='black')

    def _draw_critic_branch(self, ax, critic: nn.Module, x_start: float, y_offset: float, node_size: float):
        """Draw critic network branch."""

        layers = list(critic.network.children())
        x_pos = x_start

        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                rect = FancyBboxPatch((x_pos - node_size / 2, y_offset - node_size / 2),
                                      node_size, node_size,
                                      boxstyle="round,pad=0.1",
                                      facecolor='lightsteelblue',
                                      edgecolor='black',
                                      linewidth=2)
                ax.add_patch(rect)

                layer_info = f"Critic\n{layer.in_features}→{layer.out_features}"
                ax.text(x_pos, y_offset, layer_info, ha='center', va='center',
                        fontsize=9, color='black')
                x_pos += 2

        # Value output
        rect = FancyBboxPatch((x_pos - node_size / 2, y_offset - node_size / 2),
                              node_size * 1.5, node_size,
                              boxstyle="round,pad=0.1",
                              facecolor='lightsteelblue',
                              edgecolor='black',
                              linewidth=2)
        ax.add_patch(rect)
        ax.text(x_pos + node_size / 4, y_offset, "V(s)\n[1]",
                ha='center', va='center', fontsize=10, fontweight='bold', color='black')

    def _draw_dqn_loss_objective(self, ax, x_pos: float, node_size: float):
        """Draw DQN loss and objective functions."""

        # Objective function
        obj_rect = FancyBboxPatch((x_pos - node_size, 1.5),
                                  node_size * 2, node_size,
                                  boxstyle="round,pad=0.1",
                                  facecolor=self.colors['objective'],
                                  edgecolor='black',
                                  linewidth=2)
        ax.add_patch(obj_rect)

        obj_text = "Objective: max Q(s,a)\nOptimal Q-function"
        ax.text(x_pos, 2, obj_text, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

        # Loss function
        loss_rect = FancyBboxPatch((x_pos - node_size, -1.5),
                                   node_size * 2, node_size,
                                   boxstyle="round,pad=0.1",
                                   facecolor=self.colors['loss'],
                                   edgecolor='black',
                                   linewidth=2)
        ax.add_patch(loss_rect)

        loss_text = "Loss: MSE\n(Q(s,a) - target)²"
        ax.text(x_pos, -1.5, loss_text, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

        # Bellman equation
        ax.text(x_pos, -3, "Target = r + γ max Q(s',a')",
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))

    def _draw_actor_critic_loss_objective(self, ax, x_pos: float, node_size: float):
        """Draw Actor-Critic loss and objective functions."""

        # Actor objective
        actor_obj_rect = FancyBboxPatch((x_pos - node_size, 2.5),
                                        node_size * 2, node_size * 0.8,
                                        boxstyle="round,pad=0.1",
                                        facecolor=self.colors['actor'],
                                        edgecolor='black',
                                        linewidth=2)
        ax.add_patch(actor_obj_rect)

        ax.text(x_pos, 2.9, "Actor Objective:\nmax E[log π(a|s) * A(s,a)]",
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')

        # Critic objective
        critic_obj_rect = FancyBboxPatch((x_pos - node_size, 1.2),
                                         node_size * 2, node_size * 0.8,
                                         boxstyle="round,pad=0.1",
                                         facecolor=self.colors['critic'],
                                         edgecolor='black',
                                         linewidth=2)
        ax.add_patch(critic_obj_rect)

        ax.text(x_pos, 1.6, "Critic Objective:\nmin E[(V(s) - R)²]",
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')

        # Combined loss
        loss_rect = FancyBboxPatch((x_pos - node_size, -0.5),
                                   node_size * 2, node_size,
                                   boxstyle="round,pad=0.1",
                                   facecolor=self.colors['loss'],
                                   edgecolor='black',
                                   linewidth=2)
        ax.add_patch(loss_rect)

        ax.text(x_pos, -0.5, "Combined Loss:\nActor + α*Critic + β*Entropy",
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    def _draw_reinforce_loss_objective(self, ax, x_pos: float, node_size: float):
        """Draw REINFORCE loss and objective functions."""

        # Objective function
        obj_rect = FancyBboxPatch((x_pos - node_size, 1.5),
                                  node_size * 2, node_size,
                                  boxstyle="round,pad=0.1",
                                  facecolor=self.colors['objective'],
                                  edgecolor='black',
                                  linewidth=2)
        ax.add_patch(obj_rect)

        obj_text = "Objective: max E[R]\nPolicy gradient"
        ax.text(x_pos, 2, obj_text, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

        # Loss function
        loss_rect = FancyBboxPatch((x_pos - node_size, -1.5),
                                   node_size * 2, node_size,
                                   boxstyle="round,pad=0.1",
                                   facecolor=self.colors['loss'],
                                   edgecolor='black',
                                   linewidth=2)
        ax.add_patch(loss_rect)

        loss_text = "Loss: -log π(a|s) * G"
        ax.text(x_pos, -1.5, loss_text, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

        # Return calculation
        ax.text(x_pos, -3, "G = Σ γᵗ r_{t+1}",
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))

    def _add_diagram_title(self, ax, title: str, config: Dict[str, Any]):
        """Add title and configuration info to diagram."""

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        # Configuration info
        config_text = f"Cities: {config['n_cities']} | LR: {config.get('lr', 'N/A')} | Seed: {config.get('seed', 'N/A')}"
        ax.text(0.5, 0.95, config_text, transform=ax.transAxes,
                ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))

        # Remove axes
        ax.set_xlim(-2, max(12, len(config.get('hidden_sizes', [256])) * 2 + 8))
        ax.set_ylim(-4, 4)
        ax.axis('off')

    def _save_model_summary(self, agent, agent_name: str, config: Dict[str, Any], base_filename: str):
        """Save text summary of model architecture."""

        summary_path = os.path.join(self.output_dir, f"{base_filename}_summary.txt")

        with open(summary_path, 'w') as f:
            f.write(f"Model Architecture Summary\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Agent: {agent_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Configuration
            f.write("Configuration:\n")
            f.write("-" * 20 + "\n")
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

            # Model parameters
            if hasattr(agent, 'q_network'):
                f.write("Q-Network Architecture:\n")
                f.write("-" * 25 + "\n")
                f.write(str(agent.q_network))
                f.write(f"\nTotal parameters: {sum(p.numel() for p in agent.q_network.parameters())}\n\n")

                if hasattr(agent, 'target_network'):
                    f.write("Target Network: Yes\n")
                    f.write(f"Update frequency: {getattr(agent, 'target_update_freq', 'N/A')}\n\n")

            if hasattr(agent, 'actor'):
                f.write("Actor Network:\n")
                f.write("-" * 15 + "\n")
                f.write(str(agent.actor))
                f.write(f"\nActor parameters: {sum(p.numel() for p in agent.actor.parameters())}\n\n")

                f.write("Critic Network:\n")
                f.write("-" * 16 + "\n")
                f.write(str(agent.critic))
                f.write(f"\nCritic parameters: {sum(p.numel() for p in agent.critic.parameters())}\n\n")

            if hasattr(agent, 'policy_network'):
                f.write("Policy Network:\n")
                f.write("-" * 16 + "\n")
                f.write(str(agent.policy_network))
                f.write(f"\nTotal parameters: {sum(p.numel() for p in agent.policy_network.parameters())}\n\n")

            # Loss and objective functions
            f.write("Loss and Objective Functions:\n")
            f.write("-" * 30 + "\n")

            if "DQN" in agent_name:
                f.write("Objective: Maximize Q(s,a) - Learn optimal action-value function\n")
                f.write("Loss: Mean Squared Error between Q(s,a) and target\n")
                f.write("Target: r + γ * max_a' Q(s', a') (Bellman equation)\n")
                if hasattr(agent, 'gamma'):
                    f.write(f"Discount factor (γ): {agent.gamma}\n")

            elif "REINFORCE" in agent_name:
                f.write("Objective: Maximize expected return E[R]\n")
                f.write("Loss: -log π(a|s) * G (Policy gradient)\n")
                f.write("Return: G = Σ γᵗ r_{t+1} (Discounted future rewards)\n")
                if hasattr(agent, 'gamma'):
                    f.write(f"Discount factor (γ): {agent.gamma}\n")

            elif "ActorCritic" in agent_name:
                f.write("Actor Objective: Maximize E[log π(a|s) * A(s,a)]\n")
                f.write("Critic Objective: Minimize E[(V(s) - R)²]\n")
                f.write("Combined Loss: Actor loss + α*Critic loss + β*Entropy\n")
                f.write("Advantage: A(s,a) = R - V(s)\n")
                if hasattr(agent, 'entropy_coeff'):
                    f.write(f"Entropy coefficient (β): {agent.entropy_coeff}\n")
                if hasattr(agent, 'value_loss_coeff'):
                    f.write(f"Value loss coefficient (α): {agent.value_loss_coeff}\n")

    def _create_torchviz_diagram(self, agent, agent_name: str, base_filename: str):
        """Create computational graph using torchviz."""

        try:
            from torchviz import make_dot

            # Create dummy input
            if hasattr(agent, 'q_network'):
                if hasattr(agent.q_network, 'lstm'):
                    # LSTM input: [batch, seq_len, features]
                    dummy_input = torch.randn(1, 5, agent.n_cities)
                    output = agent.q_network(dummy_input)
                else:
                    # Regular MLP input
                    if "OrderEmbedding" in agent_name:
                        dummy_input = torch.randn(1, 2 * agent.n_cities)
                    elif "Optimal" in agent_name:
                        dummy_input = torch.randn(1, 4 * agent.n_cities + 1)
                    else:
                        dummy_input = torch.randn(1, agent.n_cities)
                    output = agent.q_network(dummy_input)

                # Create computational graph
                dot = make_dot(output, params=dict(agent.q_network.named_parameters()),
                               show_attrs=True, show_saved=True)

                # Save graph
                graph_path = os.path.join(self.output_dir, f"{base_filename}_computational_graph")
                dot.render(graph_path, format='png', cleanup=True)

            elif hasattr(agent, 'actor') and hasattr(agent, 'critic'):
                # Actor-Critic
                dummy_input = torch.randn(1, agent.n_cities)

                actor_output = agent.actor(dummy_input)
                critic_output = agent.critic(dummy_input)

                # Create separate graphs
                actor_dot = make_dot(actor_output, params=dict(agent.actor.named_parameters()),
                                     show_attrs=True, show_saved=True)
                actor_path = os.path.join(self.output_dir, f"{base_filename}_actor_graph")
                actor_dot.render(actor_path, format='png', cleanup=True)

                critic_dot = make_dot(critic_output, params=dict(agent.critic.named_parameters()),
                                      show_attrs=True, show_saved=True)
                critic_path = os.path.join(self.output_dir, f"{base_filename}_critic_graph")
                critic_dot.render(critic_path, format='png', cleanup=True)

        except Exception as e:
            print(f"Could not create torchviz diagram: {e}")


def create_model_visualizer(base_output_dir: str = "models") -> ModelVisualizer:

    return ModelVisualizer(base_output_dir)



---plotter.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import os
from datetime import datetime

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'lines.linewidth': 2,
    'lines.markersize': 6
})


class TSPPlotter:
    """
    Publication-quality plotting utilities for TSP experiments.
    """

    def __init__(self,
                 output_dir: str = "plots",
                 dpi: int = 300,
                 format: str = "png"):
        """
        Initialize TSP plotter.

        Args:
            output_dir: Directory to save plots
            dpi: Resolution for saved plots
            format: File format for saved plots
        """
        self.output_dir = output_dir
        self.dpi = dpi
        self.format = format

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Color palette for different algorithms
        self.colors = {
            'DQN Basic': '#1f77b4',
            'REINFORCE': '#ff7f0e',
            'DQN LSTM': '#2ca02c',
            'DQN Order Embedding': '#d62728',
            'Actor-Critic': '#9467bd',
            'DQN Optimal': '#8c564b',
            'Optimal': '#e377c2',
            'Random': '#7f7f7f'
        }

    def plot_learning_curves(self,
                             training_data: Dict[str, pd.DataFrame],
                             metric: str = 'total_distance',
                             window: int = 100,
                             title: Optional[str] = None,
                             save_name: Optional[str] = None) -> None:
        """
        Plot learning curves for multiple algorithms.

        Args:
            training_data: Dictionary of {algorithm_name: dataframe}
            metric: Metric to plot ('total_distance', 'total_reward', 'loss')
            window: Rolling window size for smoothing
            title: Plot title
            save_name: Name for saved plot
        """
        plt.figure(figsize=(12, 8))

        for agent_name, data in training_data.items():
            if metric in data.columns:
                # Apply rolling mean for smoothing
                smoothed = data[metric].rolling(window=window, min_periods=1).mean()
                episodes = data['episode'] if 'episode' in data.columns else range(len(data))

                color = self.colors.get(agent_name, np.random.rand(3, ))
                plt.plot(episodes, smoothed, label=agent_name, color=color, alpha=0.8)

                # Add confidence bands (std)
                if len(data) > window:
                    std = data[metric].rolling(window=window, min_periods=1).std()
                    plt.fill_between(episodes, smoothed - std, smoothed + std,
                                     color=color, alpha=0.2)

        plt.xlabel('Episode')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(title or f'Learning Curves - {metric.replace("_", " ").title()}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_name:
            self._save_plot(save_name)
        plt.show()

    def plot_performance_comparison(self,
                                    results: Dict[str, Dict[str, float]],
                                    metrics: List[str] = ['avg_distance', 'gap_percentage'],
                                    title: Optional[str] = None,
                                    save_name: Optional[str] = None) -> None:
        """
        Plot performance comparison across algorithms.

        Args:
            results: Dictionary of {algorithm_name: {metric: value}}
            metrics: List of metrics to plot
            title: Plot title
            save_name: Name for saved plot
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))

        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            agent_names = list(results.keys())
            values = [results[agent][metric] for agent in agent_names]
            colors = [self.colors.get(agent, np.random.rand(3, )) for agent in agent_names]

            bars = axes[i].bar(agent_names, values, color=colors, alpha=0.8)
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                             f'{value:.3f}', ha='center', va='bottom')

        plt.suptitle(title or 'Performance Comparison')
        plt.tight_layout()

        if save_name:
            self._save_plot(save_name)
        plt.show()

    def plot_convergence_analysis(self,
                                  training_data: Dict[str, pd.DataFrame],
                                  convergence_threshold: float = 0.01,
                                  window: int = 200,
                                  title: Optional[str] = None,
                                  save_name: Optional[str] = None) -> None:
        """
        Plot convergence analysis showing when algorithms converge.

        Args:
            training_data: Dictionary of {algorithm_name: dataframe}
            convergence_threshold: Threshold for convergence detection
            window: Window size for convergence detection
            title: Plot title
            save_name: Name for saved plot
        """
        plt.figure(figsize=(14, 8))

        convergence_episodes = {}

        for agent_name, data in training_data.items():
            if 'total_distance' in data.columns:
                distances = data['total_distance'].values
                episodes = data['episode'].values if 'episode' in data.columns else np.arange(len(distances))

                # Detect convergence
                convergence_ep = self._detect_convergence(distances, convergence_threshold, window)
                convergence_episodes[agent_name] = convergence_ep

                # Plot smoothed curve
                # [1, 2, 3, 4, 5]pd.Series(distances).rolling(window=3).mean() =>
                # 0    NaN
                # 1    NaN
                # 2    2.0   # (1+2+3)/3
                # 3    3.0   # (2+3+4)/3
                # 4    4.0   # (3+4+5)/3
                smoothed = pd.Series(distances).rolling(window=50, min_periods=1).mean()
                color = self.colors.get(agent_name, np.random.rand(3, ))
                plt.plot(episodes, smoothed, label=agent_name, color=color)

                # Mark convergence point
                if convergence_ep is not None:
                    plt.axvline(x=convergence_ep, color=color, linestyle='--', alpha=0.7)
                    plt.scatter([convergence_ep], [smoothed.iloc[convergence_ep]],
                                color=color, s=100, marker='o', zorder=5)

        plt.xlabel('Episode')
        plt.ylabel('Total Distance')
        plt.title(title or f'Convergence Analysis (threshold={convergence_threshold})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Print convergence episodes
        print("Convergence Episodes:")
        for agent, ep in convergence_episodes.items():
            print(f"  {agent}: {ep if ep is not None else 'Not converged'}")

        if save_name:
            self._save_plot(save_name)
        plt.show()

    def plot_tsp_solution(self,
                          coordinates: np.ndarray,
                          path: List[int],
                          title: Optional[str] = None,
                          save_name: Optional[str] = None,
                          show_city_labels: bool = True) -> None:
        """
        Plot TSP solution path.

        Args:
            coordinates: City coordinates (n_cities x 2)
            path: Solution path
            title: Plot title
            save_name: Name for saved plot
            show_city_labels: Whether to show city labels
        """
        plt.figure(figsize=(10, 10))

        # Plot cities
        plt.scatter(coordinates[:, 0], coordinates[:, 1],
                    c='red', s=100, zorder=3, label='Cities')

        # Plot path
        path_coords = coordinates[path]
        plt.plot(path_coords[:, 0], path_coords[:, 1],
                 'b-', linewidth=2, alpha=0.7, label='Path')

        # Highlight start/end
        start_coord = coordinates[path[0]]
        plt.scatter(start_coord[0], start_coord[1],
                    c='green', s=200, marker='*', zorder=4, label='Start/End')

        # City labels
        if show_city_labels:
            for i, (x, y) in enumerate(coordinates):
                plt.annotate(str(i), (x, y), xytext=(5, 5),
                             textcoords='offset points', fontsize=10)

        # Calculate total distance
        total_distance = sum(np.linalg.norm(coordinates[path[i]] - coordinates[path[i + 1]])
                             for i in range(len(path) - 1))

        plt.title(title or f'TSP Solution (Distance: {total_distance:.3f})')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()

        if save_name:
            self._save_plot(save_name)
        plt.show()

    def plot_algorithm_comparison_radar(self,
                                        results: Dict[str, Dict[str, float]],
                                        metrics: List[str],
                                        title: Optional[str] = None,
                                        save_name: Optional[str] = None) -> None:
        """
        Create radar chart comparing algorithms across multiple metrics.

        Args:
            results: Dictionary of {algorithm_name: {metric: value}}
            metrics: List of metrics to include
            title: Plot title
            save_name: Name for saved plot
        """
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        for agent_name, data in results.items():
            values = [data.get(metric, 0) for metric in metrics]

            # Normalize values to 0-1 scale for radar chart
            normalized_values = self._normalize_for_radar(values, metrics)
            normalized_values += normalized_values[:1]  # Complete the circle

            color = self.colors.get(agent_name, np.random.rand(3, ))
            ax.plot(angles, normalized_values, 'o-', linewidth=2,
                    label=agent_name, color=color)
            ax.fill(angles, normalized_values, alpha=0.25, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title(title or 'Algorithm Comparison - Radar Chart')

        if save_name:
            self._save_plot(save_name)
        plt.show()

    def plot_hyperparameter_sensitivity(self,
                                        sensitivity_data: pd.DataFrame,
                                        param_name: str,
                                        metric: str = 'avg_distance',
                                        title: Optional[str] = None,
                                        save_name: Optional[str] = None) -> None:
       
        plt.figure(figsize=(12, 8))

        if 'algorithm' in sensitivity_data.columns:
            # Multiple algorithms
            for algorithm in sensitivity_data['algorithm'].unique():
                data = sensitivity_data[sensitivity_data['algorithm'] == algorithm]
                color = self.colors.get(algorithm, np.random.rand(3, ))
                plt.plot(data[param_name], data[metric], 'o-',
                         label=algorithm, color=color, linewidth=2, markersize=8)
        else:
            # Single algorithm
            plt.plot(sensitivity_data[param_name], sensitivity_data[metric],
                     'o-', linewidth=2, markersize=8)

        plt.xlabel(param_name.replace('_', ' ').title())
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(title or f'Hyperparameter Sensitivity - {param_name}')
        plt.grid(True, alpha=0.3)

        if 'algorithm' in sensitivity_data.columns:
            plt.legend()

        plt.tight_layout()

        if save_name:
            self._save_plot(save_name)
        plt.show()

    def create_paper_figure(self,
                            training_data: Dict[str, pd.DataFrame],
                            evaluation_results: Dict[str, Dict[str, float]],
                            optimal_distances: Dict[int, float],
                            title: str = "TSP Reinforcement Learning Results",
                            save_name: str = "paper_figure") -> None:

        fig = plt.figure(figsize=(20, 12))

        # Learning curves
        ax1 = plt.subplot(2, 3, 1)
        for agent_name, data in training_data.items():
            if 'total_distance' in data.columns:
                smoothed = data['total_distance'].rolling(window=100, min_periods=1).mean()
                episodes = data['episode'] if 'episode' in data.columns else range(len(data))
                color = self.colors.get(agent_name, np.random.rand(3, ))
                plt.plot(episodes, smoothed, label=agent_name, color=color)
        plt.xlabel('Episode')
        plt.ylabel('Total Distance')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Performance comparison
        ax2 = plt.subplot(2, 3, 2)
        agents = list(evaluation_results.keys())
        distances = [evaluation_results[agent]['avg_distance'] for agent in agents]
        colors = [self.colors.get(agent, np.random.rand(3, )) for agent in agents]
        bars = plt.bar(agents, distances, color=colors, alpha=0.8)
        plt.ylabel('Average Distance')
        plt.title('Final Performance')
        plt.xticks(rotation=45)
        for bar, dist in zip(bars, distances):
            plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                     f'{dist:.3f}', ha='center', va='bottom')

        # Gap from optimal
        ax3 = plt.subplot(2, 3, 3)
        gaps = [evaluation_results[agent]['avg_gap'] for agent in agents]
        bars = plt.bar(agents, gaps, color=colors, alpha=0.8)
        plt.ylabel('Avg Gap from Optimal (%)')
        plt.title('Optimality Gap')
        plt.xticks(rotation=45)
        for bar, gap in zip(bars, gaps):
            plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                     f'{gap:.1f}%', ha='center', va='bottom')

        # Convergence analysis
        ax4 = plt.subplot(2, 3, 4)
        convergence_episodes = {}
        for agent_name, data in training_data.items():
            if 'total_distance' in data.columns:
                distances = data['total_distance'].values
                conv_ep = self._detect_convergence(distances, 0.01, 200)
                convergence_episodes[agent_name] = conv_ep if conv_ep is not None else len(distances)

        agents = list(convergence_episodes.keys())
        conv_eps = list(convergence_episodes.values())
        colors = [self.colors.get(agent, np.random.rand(3, )) for agent in agents]
        plt.bar(agents, conv_eps, color=colors, alpha=0.8)
        plt.ylabel('Episodes to Convergence')
        plt.title('Convergence Speed')
        plt.xticks(rotation=45)

        # Sample solution visualization
        ax5 = plt.subplot(2, 3, 5)
        # Use dummy coordinates for visualization
        n_cities = 10
        np.random.seed(42)
        coords = np.random.uniform(0, 1, (n_cities, 2))
        path = list(range(n_cities)) + [0]  # Simple path

        plt.scatter(coords[:, 0], coords[:, 1], c='red', s=100, zorder=3)
        path_coords = coords[path]
        plt.plot(path_coords[:, 0], path_coords[:, 1], 'b-', linewidth=2, alpha=0.7)
        plt.scatter(coords[0, 0], coords[0, 1], c='green', s=200, marker='*', zorder=4)
        plt.title('Sample TSP Solution')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.axis('equal')

        # Performance summary table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('tight')
        ax6.axis('off')

        table_data = []
        headers = ['Algorithm', 'Avg Distance', 'Gap (%)', 'Convergence']
        for agent in agents:
            row = [
                agent,
                f"{evaluation_results[agent]['avg_distance']:.3f}",
                f"{evaluation_results[agent]['avg_gap']:.1f}",
                f"{convergence_episodes.get(agent, 'N/A')}"
            ]
            table_data.append(row)

        table = ax6.table(cellText=table_data, colLabels=headers,
                          cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax6.set_title('Performance Summary')

        plt.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()

        self._save_plot(save_name)
        plt.show()

    def _detect_convergence(self, values: np.ndarray, threshold: float, window: int) -> Optional[int]:
        """Detect convergence point in a series."""
        if len(values) < window * 2:
            return None

        smoothed = pd.Series(values).rolling(window=window, min_periods=1).mean()

        for i in range(window, len(smoothed) - window):
            recent_std = smoothed[i - window:i + window].std()
            if recent_std < threshold:
                return i

        return None

    def _normalize_for_radar(self, values: List[float], metrics: List[str]) -> List[float]:
        """Normalize values for radar chart (0-1 scale)."""
        normalized = []
        for i, (value, metric) in enumerate(zip(values, metrics)):
            # For metrics where lower is better (like distance, gap), invert
            if 'distance' in metric.lower() or 'gap' in metric.lower() or 'loss' in metric.lower():
                # Invert and normalize
                max_val = max(values) if max(values) > 0 else 1
                normalized.append(1 - (value / max_val))
            else:
                # For metrics where higher is better
                max_val = max(values) if max(values) > 0 else 1
                normalized.append(value / max_val)
        return normalized

    def _save_plot(self, filename: str) -> None:
        """Save plot to output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_path = os.path.join(self.output_dir, f"{filename}_{timestamp}.{self.format}")
        plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight', format=self.format)
        print(f"Plot saved to: {full_path}")


def create_plotter(output_dir: str = "plots") -> TSPPlotter:

    return TSPPlotter(output_dir)