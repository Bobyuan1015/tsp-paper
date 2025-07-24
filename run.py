#!/usr/bin/env python3
"""
Main entry point for TSP RL experiments.
This script provides a unified interface to run all experiments.
"""

import sys
import os
import traceback
import random
import numpy as np
import torch
from typing import Dict, Any, List
from tqdm import tqdm

import yaml

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from confs import config_loader
from environments import TSPEnvironment
from agents import DQNAgent, DQNLSTMAgent, ReinforceAgent, ActorCriticAgent, PPOAgent
from utils import Logger, Evaluator, RewardCalculator, DatasetGenerator, DatasetLoader, ModelManager


class ExperimentRunner:
    """Main experiment runner for TSP RL experiments."""
    
    def __init__(self, config_file: str = "experiment_config.yaml"):
        """
        Initialize experiment runner.
        
        Args:
            config_file: Configuration file name
        """
        # Load configurations
        self.experiment_config = config_loader.load_experiment_config(config_file)
        self.model_config = config_loader.load_model_config()
        
        # Set global seed
        self.seed = self.experiment_config['experiment']['seed']
        self._set_global_seed(self.seed)
        
        # Initialize components
        self.logger = Logger(
            experiment_name=self.experiment_config['experiment']['name'],
            log_level=self.experiment_config['logging']['level'],
            save_to_file=self.experiment_config['logging']['save_to_file'],
            console_output=self.experiment_config['logging']['console_output']
        )
        
        self.evaluator = Evaluator()
        self.reward_calculator = RewardCalculator()
        
        # Dataset components
        self.dataset_generator = DatasetGenerator(seed=self.seed)
        self.dataset_loader = DatasetLoader()
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.logger.info(f"Using device: {self.device}")
        
        # Algorithm mapping
        self.algorithm_classes = {
            'DQN': DQNAgent,
            'DQN_LSTM': DQNLSTMAgent,
            'Reinforce': ReinforceAgent,
            'ActorCritic': ActorCriticAgent,
            'PPO': PPOAgent
        }
        
        # Progress tracking
        self.total_tasks = 0
        self.completed_tasks = 0
        
    def _set_global_seed(self, seed: int):
        """Set global random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def prepare_datasets(self):
        """Generate and prepare datasets for experiments."""
        self.logger.logger.info("Preparing datasets...")
        
        # Generate random datasets with train/test split
        city_nums = self.experiment_config['dataset']['city_nums']
        total_instances = self.experiment_config['dataset']['total_instances']
        train_ratio = self.experiment_config['dataset']['train_ratio']
        
        random_datasets = self.dataset_generator.generate_random_datasets(
            city_nums=city_nums,
            num_instances=total_instances,
            train_ratio=train_ratio,
            force_regenerate=False
        )
        
        # Validate datasets
        for city_num, split_data in random_datasets.items():
            # Validate both train and test splits
            all_instances = split_data['train'] + split_data['test']
            if not self.dataset_generator.validate_dataset(all_instances):
                raise ValueError(f"Dataset validation failed for {city_num} cities")
        
        # Generate TSPLIB datasets if specified
        if self.experiment_config['dataset']['use_tsplib']:
            tsplib_problems = self.experiment_config['dataset']['tsplib_problems']
            tsplib_datasets = self.dataset_generator.generate_tsplib_datasets(
                tsplib_problems=tsplib_problems,
                force_regenerate=False
            )
        
        self.logger.logger.info("Dataset preparation completed")
    
    def create_agent(self, algorithm: str, state_dim: int, action_dim: int, 
                    state_components: List[str]) -> Any:
        """Create agent instance."""
        if algorithm not in self.algorithm_classes:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        agent_config = config_loader.get_algorithm_config(algorithm)
        agent_class = self.algorithm_classes[algorithm]
        
        return agent_class(
            state_dim=state_dim,
            action_dim=action_dim,
            config=agent_config,
            state_components=state_components,
            device=str(self.device)
        )
    
    def calculate_state_dim(self, state_components: List[str], n_cities: int) -> int:
        """Calculate state dimension based on components."""
        dim = 0
        
        for component in state_components:
            if component in ['current_city_onehot', 'visited_mask', 'order_embedding', 'distances_from_current']:
                dim += n_cities
            elif component == 'step_count':
                dim += 1
        
        return dim
    
    def _calculate_total_tasks(self):
        """Calculate total number of tasks for progress tracking."""
        algorithms = self.experiment_config['algorithms']
        city_nums = self.experiment_config['dataset']['city_nums']
        state_types = list(self.experiment_config['state_types'].keys())
        repeat_runs = self.experiment_config['training']['repeat_runs']
        modes = self.experiment_config['modes']
        
        total_tasks = 0
        
        for mode in modes:
            if mode == "per_instance":
                # Load dataset to get number of training instances
                for algorithm in algorithms:
                    for city_num in city_nums:
                        dataset = self.dataset_loader.load_random_dataset(city_num, split='both')
                        train_instances = len(dataset['train'])
                        test_instances = len(dataset['test'])

                        for state_type in state_types:
                            total_tasks += train_instances * repeat_runs*(1+test_instances)


            
            elif mode == "cross_instance":
                for algorithm in algorithms:
                    for city_num in city_nums:
                        for state_type in state_types:
                            # Cross-instance training: repeat_runs tasks
                            # Testing on test instances
                            dataset = self.dataset_loader.load_random_dataset(city_num, split='both')
                            total_tasks += repeat_runs + len(dataset['test'])*repeat_runs

        
        self.total_tasks = total_tasks
        self.logger.logger.info(f"Total tasks calculated: {total_tasks}")
    
    def run_mode(self, mode: str):
        """Run experiments in specified mode (per_instance or cross_instance)."""
        self.logger.logger.info(f"Starting {mode} mode experiments")
        
        algorithms = self.experiment_config['algorithms']
        city_nums = self.experiment_config['dataset']['city_nums']
        state_types = list(self.experiment_config['state_types'].keys())
        repeat_runs = self.experiment_config['training']['repeat_runs']
        save_model_every = self.experiment_config['training'].get('save_model_every', 100)
        evaluate_every = self.experiment_config['training'].get('evaluate_every', 100)
        
        if mode == "per_instance":
            episodes_training = self.experiment_config['training']['episodes_per_instance']
        else:  # cross_instance
            episodes_training = self.experiment_config['training']['episodes_cross_instance']
        
        for algorithm in algorithms:
            self.logger.logger.info(f"Running algorithm: {algorithm}")
            
            for city_num in city_nums:
                self.logger.logger.info(f"Processing {city_num} cities")
                
                # Load pre-split dataset
                dataset = self.dataset_loader.load_random_dataset(city_num, split='both')
                train_data = dataset['train']
                test_data = dataset['test']
                
                for state_type in state_types:
                    self.logger.logger.info(f"State type: {state_type}")
                    
                    state_components = config_loader.get_state_type_config(state_type)
                    state_dim = self.calculate_state_dim(state_components, city_num)
                    
                    if mode == "per_instance":
                        self._run_per_instance_training(
                            algorithm, city_num, state_type, state_components, state_dim,
                            train_data, test_data, repeat_runs, episodes_training,
                            save_model_every, evaluate_every
                        )
                    else:  # cross_instance
                        self._run_cross_instance_training(
                            algorithm, city_num, state_type, state_components, state_dim,
                            train_data, test_data, repeat_runs, episodes_training,
                            save_model_every, evaluate_every
                        )

    def _run_per_instance_training(self, algorithm: str, city_num: int, state_type: str,
                                  state_components: List[str], state_dim: int, train_data: List,
                                  test_data: List, repeat_runs: int, episodes_training: int,
                                  save_model_every: int, evaluate_every: int):
        """Handle per-instance training logic."""
        # Per-instance training  
        for instance_id in range(len(train_data)):
            instance_data = train_data[instance_id]
            
            for run_id in range(repeat_runs):
                self.completed_tasks += 1
                self.logger.log_experiment_progress(
                    algorithm, city_num, "per_instance", 
                    instance_id + 1, len(train_data), run_id + 1, repeat_runs,
                    self.completed_tasks, self.total_tasks
                )
                
                # Create environment and agent
                env = self.dataset_loader.create_env_from_instance(
                    instance_data, state_components
                )
                
                agent = self.create_agent(
                    algorithm, state_dim, city_num, state_components
                )
                
                # Create model manager
                model_dir = os.path.join(
                    self.logger.get_experiment_dir(), "models", 
                    algorithm, f"{city_num}cities", state_type, "per_instance",
                    f"instance_{instance_id}_run_{run_id}"
                )
                model_manager = ModelManager(model_dir, keep_recent=10)
                
                # Train on single instance
                self._train_agent(
                    agent, env, episodes_training, algorithm, 
                    city_num, "per_instance", instance_id, run_id, 
                    state_type, "train", model_manager, 
                    save_model_every, evaluate_every,state_components
                )
        
                # Zero-shot testing on test instances
                for instance_id in range(len(test_data)):
                    instance_data = test_data[instance_id]
                    self.completed_tasks += 1
                    # Create fresh environment and agent for zero-shot test
                    env = self.dataset_loader.create_env_from_instance(
                        instance_data, state_components
                    )

                    # Zero-shot test (single episode with untrained agent)
                    self._zero_shot_test(
                        agent, env, algorithm, city_num, "per_instance",
                        instance_id, run_id, state_type, "test"
                    )

    def _run_cross_instance_training(self, algorithm: str, city_num: int, state_type: str,
                                   state_components: List[str], state_dim: int, train_data: List,
                                   test_data: List, repeat_runs: int, episodes_training: int,
                                   save_model_every: int, evaluate_every: int):
        """Handle cross-instance training logic."""
        for run_id in range(repeat_runs):
            self.completed_tasks += 1
            self.logger.log_experiment_progress(
                algorithm, city_num, "cross_instance", 
                1, 1, run_id + 1, repeat_runs,
                self.completed_tasks, self.total_tasks
            )
            
            # Create agent once for all training instances
            agent = self.create_agent(
                algorithm, state_dim, city_num, state_components
            )
            
            # Create model manager
            model_dir = os.path.join(
                self.logger.get_experiment_dir(), "models", 
                algorithm, f"{city_num}cities", state_type, "cross_instance",
                f"run_{run_id}"
            )
            model_manager = ModelManager(model_dir, keep_recent=10)
            
            # Cross-instance training
            self._train_agent(
                agent, train_data, episodes_training,
                algorithm, city_num, "cross_instance", 0, run_id, 
                state_type, "train", model_manager,
                save_model_every, evaluate_every, state_components
            )
            
            # Test on test instances
            for instance_id in range(len(test_data)):
                self.completed_tasks += 1
                instance_data = test_data[instance_id]
                
                env = self.dataset_loader.create_env_from_instance(
                    instance_data, state_components
                )
                
                # Use _zero_shot_test to ensure detailed step logging to CSV
                self._zero_shot_test(
                    agent, env, algorithm, city_num, "cross_instance", 
                    instance_id, run_id, state_type, "test"
                )

    def run_per_instance_mode(self):
        """Run per-instance training mode."""
        return self.run_mode("per_instance")

    def run_cross_instance_mode(self):
        """Run cross-instance training mode."""
        return self.run_mode("cross_instance")

    def _train_agent(self, agent, env_or_data, num_episodes: int, algorithm: str,
                    city_num: int, mode: str, instance_id: int, run_id: int, 
                    state_type: str, train_test: str, model_manager: ModelManager = None,
                    save_model_every: int = 100, evaluate_every: int = 100,
                    state_components: List[str] = None):
        """Train agent on environment or across multiple instances."""
        for episode in range(num_episodes):
            # Handle cross-instance vs single instance training
            if mode == "cross_instance" and isinstance(env_or_data, list):
                # Randomly sample instance for cross-instance training
                instance_data = random.choice(env_or_data)
                env = self.dataset_loader.create_env_from_instance(
                    instance_data, state_components
                )
                current_instance_id = 0  # Use 0 for cross-instance logging
            else:
                # Single instance training
                env = env_or_data
                current_instance_id = instance_id
            
            state = env.reset()
            agent.reset_episode()
            
            total_reward = 0.0
            step = 0

            while True:
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                # Update agent
                experience = {
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done
                }
                
                update_info = agent.update(experience)
                
                # Extract coordinates and state values for CSV logging
                coordinates = env.coordinates.tolist() if hasattr(env, 'coordinates') else None
                state_values = state_components
                
                # Log step to CSV for both modes
                self.logger.log_training_step(
                    algorithm, city_num, mode, current_instance_id, run_id, state_type,
                    train_test, episode, step, action, state, done, reward,
                    update_info.get('loss', None), total_reward + reward, 
                    env.total_distance, env.optimal_distance, env.optimal_path,
                    coordinates, state_values
                )

                total_reward += reward
                step += 1
                
                if done:
                    break
                
                state = next_state
            
            # Log episode summary periodically
            if episode % evaluate_every == 0:
                self.logger.log_episode_summary(
                    algorithm, city_num, mode, current_instance_id, run_id, state_type,
                    train_test, episode, total_reward, step, update_info
                )
            
            # Save model periodically
            if episode % save_model_every == 0 and model_manager is not None:
                model_manager.save_model(
                    agent, episode, total_reward, 
                    metric_name="reward", is_higher_better=True
                )
    
    def _extract_state_values(self, state: Dict[str, Any], state_components: List[str]) -> Dict[str, Any]:
        """
        Extract state values based on state_type configuration.
        
        Args:
            state: Current state dictionary
            state_components: List of state components to extract
            
        Returns:
            Dictionary with state component values
        """
        state_values = {}
        
        for component in state_components:
            if component in state:
                value = state[component]
                # Convert numpy arrays to lists for JSON serialization
                if hasattr(value, 'tolist'):
                    state_values[component] = value.tolist()
                elif isinstance(value, (int, float, bool)):
                    state_values[component] = value
                else:
                    # For other types, convert to string
                    state_values[component] = str(value)
            else:
                # Component not found in state
                state_values[component] = None
                
        return state_values

    def _zero_shot_test(self, agent, env, algorithm: str, city_num: int, 
                       mode: str, instance_id: int, run_id: int, state_type: str, 
                       train_test: str):
        """Perform test with agent (trained or untrained) and log detailed steps to CSV."""
        state = env.reset()
        agent.reset_episode()
        
        total_reward = 0.0
        step = 0
        
        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)

            # Extract coordinates and state values for CSV logging
            coordinates = env.coordinates.tolist() if hasattr(env, 'coordinates') else None
            # For _zero_shot_test, we don't have state_components passed, so we need to get them
            state_components = config_loader.get_state_type_config(state_type)


            self.logger.log_training_step(
                algorithm, city_num, mode, instance_id, run_id, state_type,
                train_test, 0, step, action, state, done, reward,
                None, total_reward + reward, env.total_distance, env.optimal_distance, env.optimal_path,
                coordinates, state_components
            )

            total_reward += reward
            step += 1
            # Log step

            
            if done:
                break
            
            state = next_state
        
        # Log episode summary
        self.logger.log_episode_summary(
            algorithm, city_num, mode, instance_id, run_id, state_type,
            train_test, 0, total_reward, step
        )
    #
    # def run_experiments(self):
    #     """Run all experiments."""
    #     self.logger.logger.info("Starting TSP RL experiments")
    #
    #     try:
    #         # Prepare datasets
    #         self.prepare_datasets()
    #
    #         # Calculate total tasks for progress tracking
    #         self._calculate_total_tasks()
    #
    #         # Run experiments
    #         modes = self.experiment_config['modes']
    #
    #         # if "per_instance" in modes:
    #         #     self.run_mode("per_instance")
    #         #
    #         if "cross_instance" in modes:
    #             self.run_mode("cross_instance")
    #
    #         # Analyze results
    #         analysis = self.logger.analyze_results()
    #         self.logger.save_experiment_summary(analysis)
    #
    #         self.logger.logger.info("All experiments completed successfully")
    #
    #     except Exception as e:
    #         self.logger.logger.error(f"Experiment failed: {e}")
    #         raise



def main():
    """Main function to run experiments."""

    # Load configuration from YAML file (for backward compatibility)
    config_path = "confs/experiment_config.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        sys.exit(1)
    
    # Create experiment runner with relative config path for backward compatibility
    runner = ExperimentRunner(config_file="experiment_config.yaml")
    
    try:
        # Get modes from config
        modes = config.get('modes', ['per_instance'])
        
        # Prepare datasets first
        print("Preparing datasets...")
        runner.prepare_datasets()
        
        # Calculate total tasks for progress tracking
        runner._calculate_total_tasks()
        
        # Run experiments based on configured modes
        for mode in modes:
            if mode == "per_instance":
                print("Running per-instance experiments...")
                runner.run_per_instance_mode()
                
            elif mode == "cross_instance":
                print("Running cross-instance experiments...")
                runner.run_cross_instance_mode()

                
        # Analyze results
        analysis = runner.logger.analyze_results()
        runner.logger.save_experiment_summary(analysis)
        
        print("All experiments completed successfully!")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"Experiment failed with error: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()