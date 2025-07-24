import os
import logging
import csv
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

from confs.path import project_root


class Logger:
    """Comprehensive logging system for TSP experiments."""
    
    def __init__(self, 
                 experiment_name: str,
                 log_level: str = "INFO",
                 save_to_file: bool = True,
                 console_output: bool = True):
        """
        Initialize logger.
        
        Args:
            experiment_name: Name of the experiment
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            save_to_file: Whether to save logs to file
            console_output: Whether to output logs to console
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment directory
        self.experiment_dir = os.path.join(project_root, "results", experiment_name, self.timestamp)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Setup standard logger
        self.logger = self._setup_logger(log_level, save_to_file, console_output)
        
        # CSV data storage
        self.csv_data = []
        self.csv_headers = [
            'algorithm', 'city_num', 'mode', 'instance_id', 'run_id', 'state_type',
            'train_test', 'episode', 'step', 'action', 'state', 'done', 'reward', 
            'loss', 'total_reward', 'current_length','optimal_distance','optimal_path',
            'coordinates', 'state_values', 'timestamp'
        ]
        
        # Initialize CSV file
        self.csv_file = os.path.join(self.experiment_dir, "experiment_data.csv")
        self._init_csv_file()
        
        # Experiment metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': self.timestamp,
            'experiment_dir': self.experiment_dir
        }
        
        self.logger.info(f"Initialized logger for experiment: {experiment_name}")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def _setup_logger(self, log_level: str, save_to_file: bool, console_output: bool) -> logging.Logger:
        """Setup logger with file and console handlers."""
        logger = logging.getLogger(f"tsp_experiment_{self.experiment_name}")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s][%(name)s][%(levelname)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        if save_to_file:
            log_file = os.path.join(self.experiment_dir, "experiment.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _init_csv_file(self):
        """Initialize CSV file with headers."""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_headers)
    
    def log_training_step(self, 
                         algorithm: str,
                         city_num: int,
                         mode: str,
                         instance_id: int,
                         run_id: int,
                         state_type: str,
                         train_test: str,
                         episode: int,
                         step: int,
                         action: int,
                         state: Dict[str, Any],
                         done: bool,
                         reward: float,
                         loss: Optional[float] = None,
                         total_reward: Optional[float] = None,
                         current_length=None,
                         optimal_distance=None,
                         optimal_path=[],
                         coordinates=None,
                         state_values=None):
        """
        Log a single training/testing step.
        
        Args:
            algorithm: Algorithm name
            city_num: Number of cities
            mode: Training mode (per_instance or cross_instance)
            instance_id: Instance ID
            run_id: Run ID
            state_type: State representation type
            train_test: Whether this is training or testing data
            episode: Episode number
            step: Step number within episode
            action: Action taken
            state: State representation
            done: Whether episode is done
            reward: Reward received
            loss: Loss value (if available)
            total_reward: Total reward so far in episode
            current_length: Current tour length
            optimal_distance: Optimal distance for this instance
            optimal_path: Optimal path for this instance
            coordinates: City coordinates for this instance
            state_values: State type components values
        """
        # Format state as JSON string (only include numeric values for CSV)
        state_json = {}
        for key, value in state.items():
            if isinstance(value, (int, float)):
                state_json[key] = float(value)
            elif hasattr(value, 'tolist'):  # numpy array
                state_json[key] = value.tolist()
        
        # Format coordinates and state_values as JSON strings
        coordinates_json = json.dumps(coordinates) if coordinates is not None else ''
        state_values_json = json.dumps(state_values) if state_values is not None else ''
        
        # Create row data
        row_data = [
            algorithm, city_num, mode, instance_id, run_id, state_type,
            train_test, episode, step, action, json.dumps(state_json), 
            int(done), reward, loss or '', total_reward or '', current_length,optimal_distance, optimal_path,
            coordinates_json, state_values_json, datetime.now().isoformat()
        ]
        
        # Append to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
        
        # Log to standard logger
        log_msg = (f"[{algorithm}][{city_num}cities][{state_type}][{mode}]"
                  f"[Episode {episode}][Step {step}]: "
                  f"Action={action}, Reward={reward:.4f}, Done={done}")
        
        if loss is not None:
            log_msg += f", Loss={loss:.6f}"
        
        self.logger.debug(log_msg)
    
    def log_episode_summary(self,
                           algorithm: str,
                           city_num: int,
                           mode: str,
                           instance_id: int,
                           run_id: int,
                           state_type: str,
                           train_test: str,
                           episode: int,
                           total_reward: float,
                           episode_length: int,
                           metrics: Dict[str, float] = None):
        """
        Log episode summary.
        
        Args:
            algorithm: Algorithm name
            city_num: Number of cities
            mode: Training mode
            instance_id: Instance ID
            run_id: Run ID
            state_type: State representation type
            train_test: Training or testing
            episode: Episode number
            total_reward: Total episode reward
            episode_length: Number of steps in episode
            metrics: Additional metrics (loss, etc.)
        """
        log_msg = (f"[{algorithm}][{city_num}cities][{state_type}][{mode}]"
                  f"[Episode {episode} SUMMARY]: "
                  f"Total Reward={total_reward:.4f}, "
                  f"Length={episode_length}")
        
        if metrics:
            metric_strs = [f"{k}={v:.6f}" for k, v in metrics.items()]
            log_msg += f", {', '.join(metric_strs)}"
        
        self.logger.info(log_msg)
    
    def log_experiment_progress(self,
                              algorithm: str,
                              city_num: int,
                              mode: str,
                              current_instance: int,
                              total_instances: int,
                              current_run: int,
                              total_runs: int,
                              current_task: int = None,
                              total_tasks: int = None):
        """Log overall experiment progress."""
        progress_msg = (f"[{algorithm}][{city_num}cities][{mode}] "
                       f"Progress: Instance {current_instance}/{total_instances}, "
                       f"Run {current_run}/{total_runs}")
        
        # Add total progress if provided
        if current_task is not None and total_tasks is not None and total_tasks > 0:
            percentage = (current_task / total_tasks) * 100
            progress_msg += f" | 总进度: {current_task}/{total_tasks} ({percentage:.1f}%)"
        
        self.logger.info(progress_msg)
    
    def log_performance_metrics(self,
                              algorithm: str,
                              city_num: int,
                              state_type: str,
                              mode: str,
                              metrics: Dict[str, float]):
        """Log performance metrics."""
        metrics_msg = (f"[{algorithm}][{city_num}cities][{state_type}][{mode}] "
                      f"Performance Metrics: ")
        
        metric_strs = [f"{k}={v:.6f}" for k, v in metrics.items()]
        metrics_msg += ", ".join(metric_strs)
        
        self.logger.info(metrics_msg)
    
    def save_experiment_summary(self, summary_data: Dict[str, Any]):
        """Save experiment summary to JSON file."""
        summary_file = os.path.join(self.experiment_dir, "experiment_summary.json")
        
        summary_data.update(self.metadata)
        summary_data['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        self.logger.info(f"Experiment summary saved to {summary_file}")
    
    def create_algorithm_csv(self, algorithm: str) -> str:
        """Create separate CSV file for specific algorithm."""
        algorithm_dir = os.path.join(self.experiment_dir, algorithm)
        os.makedirs(algorithm_dir, exist_ok=True)
        
        algorithm_csv = os.path.join(algorithm_dir, f"{algorithm}_data.csv")
        
        # Initialize with headers
        with open(algorithm_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_headers)
        
        return algorithm_csv
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze experiment results from CSV data."""
        if not os.path.exists(self.csv_file):
            self.logger.warning("No CSV data file found for analysis")
            return {}
        
        try:
            df = pd.read_csv(self.csv_file)
            
            # Basic statistics
            analysis = {
                'total_steps': len(df),
                'algorithms': df['algorithm'].unique().tolist(),
                'city_nums': df['city_num'].unique().tolist(),
                'state_types': df['state_type'].unique().tolist(),
                'modes': df['mode'].unique().tolist()
            }
            
            # Performance by algorithm
            if 'total_reward' in df.columns:
                # Filter for episode completion (done=True)
                episode_data = df[df['done'] == 1]
                
                if not episode_data.empty:
                    analysis['performance_by_algorithm'] = {}
                    for algorithm in episode_data['algorithm'].unique():
                        alg_data = episode_data[episode_data['algorithm'] == algorithm]
                        analysis['performance_by_algorithm'][algorithm] = {
                            'mean_reward': float(alg_data['total_reward'].mean()),
                            'std_reward': float(alg_data['total_reward'].std()),
                            'episodes_completed': len(alg_data)
                        }
            
            self.logger.info("Results analysis completed")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing results: {e}")
            return {}
    
    def get_experiment_dir(self) -> str:
        """Get experiment directory path."""
        return self.experiment_dir
    
    def get_csv_file(self) -> str:
        """Get CSV file path."""
        return self.csv_file