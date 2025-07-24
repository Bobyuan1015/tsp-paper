下面代码中TSPEnvironment产生城市坐标是随机数，现在要求支持tsplib95数据集，通用标志位自动切换 随机 还是饮用库数据。
TSPEnvironment代码中变量，函数，函数入参不能变，因为有代码依赖的关系。
要求注意下面问题问题：
1.只选取中TSPlib EUC_2D的类别数据，其他数据不做支持
2.tsplib中城市距离 与 随机生成方案的距离要求统一（是原始坐标被归一化到0到1之间），此时最优距离如何处理。
3.tsplib是不是还要下载文件，下载地址在哪，下载哪些文件
4.TSPEnvironment中 n_cities要求匹配



class TSPEnvironment:
    """
    Traveling Salesman Problem environment for reinforcement learning.

    The agent starts at city 0 and must visit all cities exactly once before returning to the start.
    """

    def __init__(self,
                 n_cities: int = 10,
                 coordinates: Optional[np.ndarray] = None,
                 seed: Optional[int] = None):
        """
        Initialize TSP environment.

        Args:
            n_cities: Number of cities
            coordinates: Optional pre-defined city coordinates (n_cities x 2)
            seed: Random seed for reproducibility
        """
        self.n_cities = n_cities
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Initialize city coordinates
        if coordinates is not None:
            assert coordinates.shape == (n_cities, 2), f"Coordinates must be {n_cities}x2"
            self.coordinates = coordinates.copy()
        else:
            self.coordinates = np.random.uniform(0, 1, (n_cities, 2))

        # Precompute distance matrix
        self.distance_matrix = self._compute_distance_matrix()

        # State variables
        self.current_city = 0
        self.visited = set([0])  # Start at city 0
        self.path = [0]
        self.total_distance = 0.0
        self.done = False



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
            # Regenerate coordinates if new seed provided
            self.coordinates = np.random.uniform(0, 1, (self.n_cities, 2))
            self.distance_matrix = self._compute_distance_matrix()

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
            self.done = True

        info = {
            'total_distance': self.total_distance,
            'path': self.path.copy(),
            'visited_all': len(self.visited) == self.n_cities,
            'is_valid_path': self._is_valid_path()
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

        # Additional features
        progress = len(self.visited) / self.n_cities

        # Order embedding: use visit order as values
        order_embedding = np.zeros(self.n_cities)
        for i, city in enumerate(self.path):
            if i == 0:
                order_embedding[city] = 1.0  # Current city
            else:
                order_embedding[city] = 1.0 / i  # Inverse of visit step

        return {
            'current_city_onehot': current_city_onehot, # basic
            'sequence_onehot': sequence_onehot, # lstm
            'sequence_length': sequence_length  ,# lstm
            'visited_mask': visited_mask,  # basic
            'order_embedding': order_embedding, # order_embedding
            'distances_from_current': distances_from_current,
            'progress': progress,
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

        plt.title(f'TSP Environment - {len(self.visited)}/{self.n_cities} cities visited\n'
                  f'Current city: {self.current_city}, Total distance: {self.total_distance:.3f}')
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
        """
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