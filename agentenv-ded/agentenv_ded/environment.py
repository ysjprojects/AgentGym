"""
DEDEnvServer
"""

from typing import Optional
import threading
import json
import os
import re
import argparse
import datasets
from .utils import evaluate
file_path = os.path.dirname(os.path.abspath(__file__))

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NotInitializedError(Exception):
    pass

ITEM_RANGE = {
    "train": (0, 23302),
}

class DEDEnv:

    def __init__(self) -> None:

        self._max_id = 0
        self.env = {}
        self.ls = []
        self._lock = threading.Lock()

        train_dataset = datasets.load_dataset(
            "satpalsr/rl-python",
            keep_in_memory=False,
        )["train"]

        self.dataset = {
            "train": train_dataset,
        }

    def create(self, item_id: int = 0) -> int:
        with self._lock:
            env_idx = self._max_id
            self._max_id += 1

        self.env[env_idx] = self._fetch_data(
            item_id
        )  # redundancy fetch to prevent NoneType Error
        self.ls.append(env_idx)

        return env_idx

    def step(self, env_idx, response: str):
        """
        Perform a step in the environment with the given action.
        Input:
            env_idx: the index of the environment
            action: a string in the format "<search> query </search>" or "<answer> answer </answer>"
        Output:
            observation: the observation after taking the action
            reward: the reward received after taking the action
            done: whether the episode is done
            info: additional information (not used here)
        Raises:
            ValueError: if the action is not a valid string format
        """
        self._check_env_idx(env_idx)
        reward = 0
        done = False
        observation = ""

        if isinstance(response, str):  # for llm output
            evaluation = evaluate(self.env[env_idx], response)
        else:
            raise ValueError(f"Invalid action type: {type(response)}")
        
        observation = evaluation["observation"]
        reward = evaluation["reward"]

        if reward == 1:
            done = True

        return observation, reward, done, None

    def observation(self, env_idx):
        self._check_env_idx(env_idx)
        question = self.env[env_idx]["prompt"]
        user_prompt = f"""You must reason inside <think>...</think> first. Follow this process every time.\n\n Question: {question.strip()}"""
        return user_prompt

    def reset(self, env_idx, item_id: Optional[int] = None):
        self._check_env_idx(env_idx)
        self.env[env_idx] = self._fetch_data(item_id)

    def _check_env_idx(self, env_idx):
        if env_idx not in self.env:
            raise IndexError(f"Env {env_idx} not found")
        if self.env[env_idx] is None:
            raise NotInitializedError(f"Env {env_idx} not initialized")

    def _fetch_data(self, item_id: int):
        """
        Fetch data from the dataset based on the item_id.
        """
        _id = None
        for mode, r in ITEM_RANGE.items():
            if r[0] <= item_id < r[1]:
                _id = item_id - r[0]

                return self.dataset[mode][_id]
        if _id is None:
            raise ValueError(f"Item id {item_id} is out of range.")
        
    def __del__(self):
        for idx in self.ls:
            del self.env[idx]
            print(f"-------Env {idx} closed--------")
    def close(self,id):
        try:
            self._check_env_idx(id)
            self.ls.remove(id)
            del self.env[id]
            print(f"-------Env {id} closed--------")
            return True
        except Exception as e:
            print(f"Error closing env {id}: {e}")
            return False

server = DEDEnv()
