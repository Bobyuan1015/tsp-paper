import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist as distance_matrix
from heapq import heappop, heappush, heapify
import random
from typing import Optional, Dict, Any, Tuple, List
import os
import urllib.request
import gzip
import shutil

from confs.path import project_root

try:
    import tsplib95
except ImportError:
    print("Warning: tsplib95 not installed. Only random mode will work.")
    print("Install with: pip install tsplib95")
    tsplib95 = None


class TSPEnvironment:
    """
    Enhanced Traveling Salesman Problem environment for reinforcement learning.
    
    The agent starts at city 0 and must visit all cities exactly once before returning to the start.
    Enhanced with improved state representation and error handling.
    """

    def __init__(self,
                 n_cities: int = 10,
                 coordinates: Optional[np.ndarray] = None,
                 seed: Optional[int] = None,
                 use_tsplib: bool = False,
                 tsplib_name: Optional[str] = None,
                 tsplib_path: str = None,
                 state_components: List[str] = None):
        """
        Initialize TSP environment.

        Args:
            n_cities: Number of cities (used for random mode or if tsplib problem has different size)
            coordinates: Optional pre-defined city coordinates (n_cities x 2)
            seed: Random seed for reproducibility
            use_tsplib: Whether to use TSPLIB95 dataset
            tsplib_name: Name of TSPLIB problem (e.g., 'berlin52', 'eil51')
            tsplib_path: Path to TSPLIB data files
            state_components: List of state components to include
        """
        self.n_cities = n_cities
        self.seed = seed
        self.use_tsplib = use_tsplib
        self.tsplib_name = tsplib_name
        self.tsplib_path = tsplib_path or os.path.join(project_root, "data", "tsplib")
        self.optimal_distance = None
        self.optimal_path = None
        
        # State components configuration
        self.state_components = state_components or [
            "current_city_onehot", "visited_mask", "order_embedding", 
            "distances_from_current", "step_count"
        ]

        # Download and prepare TSPLIB data if needed
        if use_tsplib and tsplib95 is not None:
            self._prepare_tsplib_data()

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Initialize city coordinates with uniqueness check
        if coordinates is not None:
            assert coordinates.shape == (n_cities, 2), f"Coordinates must be {n_cities}x2"
            self.coordinates = coordinates.copy()
        elif use_tsplib and tsplib95 is not None and tsplib_name:
            try:
                self.coordinates, self.optimal_path = self._load_tsplib_data()
                self.n_cities = len(self.coordinates)
            except Exception as e:
                print(f"Failed to load TSPLIB data: {e}, falling back to random generation")
                self.coordinates = self._generate_unique_coordinates()
                self.use_tsplib = False
        else:
            self.coordinates = self._generate_unique_coordinates()

        # Precompute distance matrix
        self.distance_matrix = self._compute_distance_matrix()
        self.optimal_path, self.optimal_distance = self.get_optimal_solution()

        # State variables
        self.current_city = 0
        self.visited = set([0])  # Start at city 0
        self.path = [0]
        self.total_distance = 0.0
        self.step_count = 0
        self.done = False

    def _generate_unique_coordinates(self) -> np.ndarray:
        """Generate unique coordinates with validation."""
        max_attempts = 1000
        for attempt in range(max_attempts):
            coordinates = np.random.uniform(0, 1, (self.n_cities, 2))
            
            # Check uniqueness by computing distance matrix
            dist_matrix = distance_matrix(coordinates, coordinates)
            
            # Check if any off-diagonal elements are zero (duplicate coordinates)
            np.fill_diagonal(dist_matrix, 1)  # Set diagonal to 1 to ignore self-distances
            if np.min(dist_matrix) > 1e-6:  # No duplicate coordinates
                return coordinates
                
        # If we couldn't generate unique coordinates, use grid-based generation
        print(f"Warning: Could not generate unique random coordinates after {max_attempts} attempts. Using grid-based generation.")
        return self._generate_grid_coordinates()
    
    def _generate_grid_coordinates(self) -> np.ndarray:
        """Generate coordinates on a grid to ensure uniqueness."""
        grid_size = int(np.ceil(np.sqrt(self.n_cities)))
        coordinates = []
        
        for i in range(self.n_cities):
            x = (i % grid_size) / (grid_size - 1) if grid_size > 1 else 0.5
            y = (i // grid_size) / (grid_size - 1) if grid_size > 1 else 0.5
            coordinates.append([x, y])
            
        # Add small random noise to avoid perfect grid
        coordinates = np.array(coordinates)
        noise = np.random.normal(0, 0.05, coordinates.shape)
        coordinates = np.clip(coordinates + noise, 0, 1)
        
        return coordinates

    def _prepare_tsplib_data(self):
        """Download and prepare TSPLIB data if not exists."""
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
                    url = f"{base_url}{tsp_gz_file}"
                    urllib.request.urlretrieve(url, tsp_gz_path)
                    print(f"Downloaded {tsp_gz_file} from {url}")

                    with gzip.open(tsp_gz_path, 'rb') as f_in:
                        with open(tsp_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    print(f"Extracted {tsp_file}")

                    os.remove(tsp_gz_path)
                except Exception as e:
                    print(f"Failed to download or extract {tsp_gz_file}: {e}")

            # Download and extract .opt.tour file
            if not os.path.exists(opt_path):
                try:
                    url = f"{base_url}{opt_gz_file}"
                    urllib.request.urlretrieve(url, opt_gz_path)
                    print(f"Downloaded {opt_gz_file} from {url}")

                    with gzip.open(opt_gz_path, 'rb') as f_in:
                        with open(opt_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    print(f"Extracted {opt_file}")

                    os.remove(opt_gz_path)
                except Exception as e:
                    print(f"Failed to download or extract {opt_gz_file}: {e}")

    def _load_tsplib_data(self) -> Tuple[np.ndarray, Optional[List[int]]]:
        """Load coordinates from TSPLIB dataset with error handling."""
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
        
        # Avoid division by zero
        coord_range = np.where(coord_range == 0, 1, coord_range)
        normalized_coords = (coordinates - min_coords) / coord_range

        # Load optimal tour and distance if available
        optimal_path = None
        if os.path.exists(opt_file):
            try:
                solution = tsplib95.load(opt_file)
                opt_tour = solution.tours[0]

                # Convert to 0-based indexing and ensure starts with 0
                tour = [city - 1 for city in opt_tour]
                if tour[0] != 0:
                    # Rotate tour to start with city 0
                    start_idx = tour.index(0)
                    tour = tour[start_idx:] + tour[:start_idx]
                tour.append(0)  # Return to start
                optimal_path = tour
            except Exception as e:
                print(f"Warning: Failed to load optimal solution for {self.tsplib_name}: {e}")
                print(f"Optimal tour file will be ignored for this instance.")
        else:
            print(f"Warning: Optimal tour file not found for {self.tsplib_name}: {opt_file}")
            print(f"This instance will be skipped or use heuristic solution.")

        return normalized_coords, optimal_path

    def _compute_distance_matrix(self) -> np.ndarray:
        """Compute distance matrix between all city pairs."""
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
                self.coordinates = self._generate_unique_coordinates()
                self.distance_matrix = self._compute_distance_matrix()
                self.optimal_path, self.optimal_distance = self.get_optimal_solution()

        # Reset state
        self.current_city = 0
        self.visited = set([0])
        self.path = [0]
        self.total_distance = 0.0
        self.step_count = 0
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
                action = random.choice(unvisited) if unvisited else 0

        # Calculate reward (negative distance)
        distance = self.distance_matrix[self.current_city, action]
        reward = -distance
        self.total_distance += distance

        # Update state
        self.current_city = action
        self.visited.add(action)
        self.path.append(action)
        self.step_count += 1

        # Check if episode is done
        if len(self.visited) == self.n_cities and action == 0:
            self.done = True

        info = {
            'total_distance': self.total_distance,
            'path': self.path.copy(),
            'visited_all': len(self.visited) == self.n_cities,
            'is_valid_path': self._is_valid_path(),
            'optimal_distance': self.optimal_distance,
            'gap_to_optimal': (self.total_distance - self.optimal_distance) / self.optimal_distance if self.optimal_distance else None,
            'step_count': self.step_count
        }

        return self._get_state(), reward, self.done, info

    def _get_state(self) -> Dict[str, Any]:
        """Get current state representation based on configured components."""
        state = {}
        
        # Always include basic info
        state['current_city'] = self.current_city
        state['visited'] = self.visited.copy()
        state['path_sequence'] = self.path.copy()
        state['coordinates'] = self.coordinates.copy()
        
        if "current_city_onehot" in self.state_components:
            current_city_onehot = np.zeros(self.n_cities)
            current_city_onehot[self.current_city] = 1.0
            state['current_city_onehot'] = current_city_onehot

        if "visited_mask" in self.state_components:
            visited_mask = np.zeros(self.n_cities)
            for city in self.visited:
                visited_mask[city] = 1.0
            state['visited_mask'] = visited_mask

        if "order_embedding" in self.state_components:
            order_embedding = np.zeros(self.n_cities)
            for i, city in enumerate(self.path):
                order_embedding[city] = min((i + 1) / self.n_cities, 1)
            state['order_embedding'] = order_embedding

        if "distances_from_current" in self.state_components:
            distances_from_current = self.distance_matrix[self.current_city].copy()
            
            # Dynamic normalization: divide by max distance from current city to unvisited cities
            unvisited_cities = [i for i in range(self.n_cities) if i not in self.visited]
            if unvisited_cities:
                max_dist_current = np.max(distances_from_current[unvisited_cities])
                if max_dist_current > 0:
                    distances_from_current = distances_from_current / max_dist_current
            else:
                # If all cities visited, normalize by global max
                max_dist = np.max(self.distance_matrix)
                if max_dist > 0:
                    distances_from_current = distances_from_current / max_dist
            
            state['distances_from_current'] = distances_from_current

        if "step_count" in self.state_components:
            # Normalize step count by total number of cities
            step_count_normalized = self.step_count / self.n_cities
            state['step_count'] = step_count_normalized

        # For LSTM models
        if "sequence_onehot" in self.state_components:
            sequence_length = len(self.path)
            sequence_onehot = np.zeros((sequence_length, self.n_cities))
            for i, city in enumerate(self.path):
                sequence_onehot[i, city] = 1.0
            state['sequence_onehot'] = sequence_onehot
            state['sequence_length'] = sequence_length

        return state

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
        Get optimal solution using improved heuristics.
        For TSPLIB problems, returns the known optimal if available.
        For random instances, uses 2-opt improvement on nearest neighbor.
        """
        # If using TSPLIB and optimal solution is known, return it
        if self.use_tsplib and self.tsplib_name and self.optimal_path:
            total_dist = 0
            for i in range(len(self.optimal_path) - 1):
                current = self.optimal_path[i]
                next_city = self.optimal_path[i + 1]
                total_dist += self.distance_matrix[current, next_city]
            return self.optimal_path, total_dist

        # Use nearest neighbor + 2-opt for better heuristic solution
        nn_path, nn_dist = self._nearest_neighbor_solution()
        opt_path, opt_dist = self._two_opt_improvement(nn_path, nn_dist)

        return opt_path, opt_dist

    def _nearest_neighbor_solution(self) -> Tuple[List[int], float]:
        """Get solution using nearest neighbor heuristic."""
        heap = [(self.distance_matrix[0, i], i) for i in range(1, self.n_cities)]
        heapify(heap)
        path = [0]
        total_dist = 0.0
        visited = {0}

        current = 0
        while len(visited) < self.n_cities:
            while heap:
                dist, next_city = heappop(heap)
                if next_city not in visited:
                    break
            else:
                break

            total_dist += dist
            path.append(next_city)
            visited.add(next_city)

            current = next_city
            for i in range(1, self.n_cities):
                if i not in visited:
                    heappush(heap, (self.distance_matrix[current, i], i))

        # Return to start
        total_dist += self.distance_matrix[current, 0]
        path.append(0)

        return path, total_dist

    def _two_opt_improvement(self, path: List[int], distance: float) -> Tuple[List[int], float]:
        """Improve solution using 2-opt local search."""
        best_path = path[:]
        best_distance = distance
        improved = True
        
        while improved:
            improved = False
            for i in range(1, len(path) - 2):
                for j in range(i + 1, len(path) - 1):
                    # Try 2-opt swap
                    new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                    new_distance = self._calculate_path_distance(new_path)
                    
                    if new_distance < best_distance:
                        best_path = new_path
                        best_distance = new_distance
                        path = new_path
                        improved = True
                        break
                if improved:
                    break
                    
        return best_path, best_distance

    def _calculate_path_distance(self, path: List[int]) -> float:
        """Calculate total distance for given path."""
        total_dist = 0.0
        for i in range(len(path) - 1):
            total_dist += self.distance_matrix[path[i], path[i + 1]]
        return total_dist


# Usage examples:
def create_random_env(n_cities=10, seed=42, state_components=None):
    """Create environment with random coordinates."""
    return TSPEnvironment(n_cities=n_cities, seed=seed, use_tsplib=False, 
                         state_components=state_components)


def create_tsplib_env(problem_name='berlin52', state_components=None):
    """Create environment with TSPLIB dataset."""
    return TSPEnvironment(use_tsplib=True, tsplib_name=problem_name,
                         state_components=state_components)