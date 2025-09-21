"""
AcademiaEnvServer
"""

from typing import Optional

from environment.academia_env import AcademiaEnv
from utils.tool.data_utils import ToolDataset
from utils.tool.helpers import extract_action_name_and_action_input
import threading

class AcademiaEnvServer:
    """
    Server class for managing multiple Academia environment instances.
    
    Provides a centralized interface for creating, resetting, and stepping through
    academia-based reinforcement learning environments with thread-safe operations.
    """

    def __init__(self) -> None:
        """
        Initialize the AcademiaEnvServer with an empty environment registry,
        dataset loader, and thread lock for concurrent access.
        
        Sets up the server with:
        - Environment counter starting at 0
        - Empty environment dictionary
        - Academia dataset from jsonl file
        - Threading lock for thread-safe operations
        """
        self._max_id = 0
        self.env = {}
        dataset_path = "Toolusage/data/academia.jsonl"
        self.dataset = ToolDataset(test_file=dataset_path)
        self._lock = threading.Lock()

    def create(self, id: int = 0) -> int:
        """
        Create a new AcademiaEnv instance with data from the specified dataset entry.
        
        Args:
            id (int): Index of the dataset entry to use for environment initialization.
                     Defaults to 0.
        
        Returns:
            int: Unique environment index that can be used to reference this environment
                 in subsequent operations.
        
        Note:
            Thread-safe operation using internal lock to ensure unique environment IDs.
        """
        with self._lock:
            env_idx = self._max_id
            self._max_id += 1
        dataset = self.dataset
        dataset_i = dict()
        dataset_i["goal"] = dataset.goals[id]
        dataset_i["ground_truth"] = dataset.ground_truths[id]
        dataset_i["ground_truth_subgoals"] = dataset.ground_truth_subgoals[id]
        dataset_i["tool"] = dataset.tools[id]

        self.env[env_idx] = AcademiaEnv(dataset=dataset_i)
        return env_idx

    def reset(self, env_idx, id: Optional[int] = None):
        """
        Reset an existing environment to its initial state.
        
        Args:
            env_idx (int): Index of the environment to reset.
            id (Optional[int]): If provided, reset the environment with new dataset entry.
                               If None, reset with the same dataset the environment was using.
        
        Note:
            When id is provided, the environment is reinitialized with new goal,
            ground truth, subgoals, and tools from the specified dataset entry.
            When id is None, the environment resets to its original state.
        """
        if id is not None:
            print(id)
            dataset = self.dataset
            dataset_i = dict()
            dataset_i["goal"] = dataset.goals[id]
            dataset_i["ground_truth"] = dataset.ground_truths[id]
            dataset_i["ground_truth_subgoals"] = dataset.ground_truth_subgoals[id]
            dataset_i["tool"] = dataset.tools[id]

            self.env[env_idx].__init__(dataset=dataset_i)
        else:
            print(None)
            self.env[env_idx].__init__(dataset=self.env[env_idx].dataset)

    def step(self, env_idx, message: str):
        """
        Execute one step in the specified environment using the provided message.
        
        Args:
            env_idx (int): Index of the environment to step through.
            message (str): Action message in the format "Action: [action] with Action Input: [input]".
        
        Returns:
            tuple: (observation, reward, done, info) where:
                - observation (str): Environment observation after the step
                - reward (float): Reward received for the action
                - done (bool): Whether the episode has ended
                - info: Additional information (always None)
        
        Note:
            If the message format is incorrect, returns a format error observation
            with done=False and the current environment reward.
        """
        action, action_input = extract_action_name_and_action_input(message)
        if action_input == None:
            observation, done = (
                'Format error, please response in the format of  "Action: [your action] with Action Input: [your action input]',
                False,
            )
            reward = self.env[env_idx].reward
            return observation, reward, done, None
        else:
            action_with_action_input = action + " with Action Input: " + action_input
            observation, reward, done, _ = self.env[env_idx].step(
                action=action_with_action_input
            )
            observation = "Observation: " + observation + "\nGive me one action."
            return observation, reward, done, None

    def observation(self, env_idx):
        """
        Get the current observation from the specified environment.
        
        Args:
            env_idx (int): Index of the environment to get observation from.
        
        Returns:
            str: Formatted observation string. If a new trial is starting,
                 returns a message with the goal. Otherwise, returns the
                 current environment observation prefixed with "Observation: ".
        
        Note:
            Observations typically contain academic data such as:
            {'year': 2021, 'venue': 'AAAI Spring Symposium - MLPS', 
             'n_citation': 0, 'keywords': [], 'doc_type': 'Conference'}
        """
        if "New trial starts." in self.env[env_idx].get_obs():
            return (
                "Now new trial starts.\nYou should perform actions to accomplish the goal: "
                + self.env[env_idx].goal
                + "\nGive me one action."
            )
        return "Observation: " + self.env[env_idx].get_obs()


academia_env_server = AcademiaEnvServer()
