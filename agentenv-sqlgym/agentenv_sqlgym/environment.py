"""
SqlGymEnvServer
"""

import os
import random
import time
from typing import Literal, Mapping, Optional, Tuple

from sqlgym import SqlGymEnv
from sqlgym.datasets import BirdDataset


class NotInitializedError(Exception):
    pass


SqlGymMode = Literal["not_initialized"] | Literal["bird_train"] | Literal["bird_dev"]

ITEM_RANGE = {
    "bird_train": (0, 9428),  # 0 <= item_id < 9428
    "bird_dev": (9428, 10962),
}


class SqlGymEnvServer:
    """
    Server class for managing multiple SqlGym environment instances.
    
    Provides a centralized interface for creating, resetting, and stepping through
    SQL-based reinforcement learning environments with capacity management and
    dataset switching between BIRD train/dev splits.
    """

    def __init__(self) -> None:
        """
        Initialize the SqlGymEnvServer with environment registry and capacity management.
        
        Sets up the server with:
        - Empty environment dictionary mapping env_idx to (SqlGymEnv, mode) tuples
        - Environment list for round-robin reuse when at capacity
        - Maximum capacity of 8 environments
        - Current index tracker for environment reuse
        """
        self.env: Mapping[int, Tuple[SqlGymEnv | None, SqlGymMode]] = {}
        self.ls = []
        self.sz = 8
        self.now = -1

    def create(self) -> int:
        """
        Create a new SqlGym environment instance or reuse an existing one if at capacity.
        
        Returns:
            int: Unique environment index. If server is at capacity (8 environments),
                 returns an existing environment ID using round-robin selection.
        
        Note:
            - Generates random environment ID between 0-489576
            - Environments are created in "not_initialized" state
            - When capacity is reached, existing environments are reused cyclically
        """
        random.seed(time.time())
        idx = random.randint(0, 489576)
        print(f"-------Env {idx} created--------")
        if len(self.env) == self.sz:
            self.now = self.now + 1
            if self.now == self.sz:
                self.now = 0
            return self.ls[self.now]

        self.env[idx] = (None, "not_initialized")
        self.ls.append(idx)
        return idx

    def observation(self, env_idx):
        """
        Get the current observation from the specified SqlGym environment.
        
        Args:
            env_idx (int): Index of the environment to get observation from.
        
        Returns:
            str: Current SQL environment observation, typically containing:
                 - Database schema information
                 - Current task/question description
                 - Available tables and columns
        
        Raises:
            IndexError: If env_idx doesn't exist
            NotInitializedError: If environment hasn't been initialized
        """
        self._check_env_idx(env_idx)
        return self.env[env_idx][0].observation

    def step(self, env_idx, action: str):
        """
        Execute a SQL action in the specified environment.
        
        Args:
            env_idx (int): Index of the environment to step through.
            action (str): SQL query string to execute.
        
        Returns:
            tuple: (execution_result, reward, terminated, info) where:
                - execution_result (str): SQL execution result, truncated to 100 chars if too long
                - reward (float): Reward for the SQL query execution
                - terminated (bool): Whether the SQL task is completed
                - info (dict): Additional execution information
        
        Raises:
            IndexError: If env_idx doesn't exist
            NotInitializedError: If environment hasn't been initialized
        
        Note:
            Long execution results (>100 characters) are truncated with "..." suffix.
        """
        self._check_env_idx(env_idx)
        execution_result, reward, terminated, info, _ = self.env[env_idx][0].step(
            action
        )
        execution_result = str(execution_result)
        if len(execution_result) > 100:
            execution_result = execution_result[:100] + "..."
        return execution_result, reward, terminated, info

    def reset(self, env_idx, item_id: Optional[int]):
        """
        Reset the environment to a specific BIRD dataset item.
        
        Args:
            env_idx (int): Index of the environment to reset.
            item_id (Optional[int]): BIRD dataset item ID to reset to.
                                   - 0-9427: BIRD train dataset
                                   - 9428-10961: BIRD dev dataset
        
        Returns:
            Initial observation after reset.
        
        Raises:
            ValueError: If item_id is outside valid ranges
            IndexError: If env_idx doesn't exist
        
        Note:
            - Automatically switches between train/dev datasets based on item_id
            - Creates new SqlGymEnv instance if switching dataset modes
            - Handles uninitialized environments by initializing them
        """
        try:
            self._check_env_idx(env_idx)
        except NotInitializedError:
            print(f"env_idx {env_idx} not initialized, initializing...")

        _id = None
        for mode, r in ITEM_RANGE.items():
            if r[0] <= item_id < r[1]:
                if self.env[env_idx][1] != mode:
                    self.env[env_idx] = (
                        SqlGymEnv(self._get_dataset_from_mode(mode)),
                        mode,
                    )
                _id = item_id - r[0]
                break
        if _id is None:
            raise ValueError(f"Item id {item_id} is out of range.")

        return self.env[env_idx][0].reset(_id)

    def _get_dataset_from_mode(self, mode: SqlGymMode) -> SqlGymEnv:
        """
        Create a BIRD dataset instance for the specified mode.
        
        Args:
            mode (SqlGymMode): Dataset mode, either "bird_train" or "bird_dev".
        
        Returns:
            BirdDataset: Configured dataset instance for the specified split.
        
        Raises:
            ValueError: If mode is not supported or BIRD path not configured.
        
        Note:
            Requires AGENTENV_SQLGYM_BIRD_PATH environment variable to be set.
        """
        if mode == "bird_train":
            bird_path = self._get_bird_path()
            return BirdDataset(bird_path, "train")
        elif mode == "bird_dev":
            bird_path = self._get_bird_path()
            return BirdDataset(bird_path, "dev")
        else:
            raise ValueError(f"Mode {mode} not supported")

    def _get_bird_path(self):
        """
        Retrieve the BIRD dataset path from environment variables.
        
        Returns:
            str: Path to the BIRD dataset directory.
        
        Raises:
            ValueError: If AGENTENV_SQLGYM_BIRD_PATH environment variable is not set.
        
        Note:
            This path should point to the root directory containing BIRD dataset files.
        """
        bird_path = os.environ.get("AGENTENV_SQLGYM_BIRD_PATH", None)
        if bird_path is None:
            raise ValueError("Please set AGENTENV_SQLGYM_BIRD_PATH")
        return bird_path

    def _check_env_idx(self, env_idx):
        """
        Validate that the environment index exists and is properly initialized.
        
        Args:
            env_idx (int): Environment index to validate.
        
        Raises:
            IndexError: If env_idx doesn't exist in the environment registry.
            NotInitializedError: If environment exists but hasn't been initialized.
        
        Note:
            This is an internal validation method used by other public methods.
        """
        if env_idx not in self.env:
            raise IndexError(f"Env {env_idx} not found")
        if self.env[env_idx] is None:
            raise NotInitializedError(f"Env {env_idx} not initialized")


sqlgym_env_server = SqlGymEnvServer()
