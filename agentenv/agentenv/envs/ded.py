from typing import Any, Mapping
import re

import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput

# SJ
# Compulsory: __init__, observe (get), step (post), reset (post)
class DEDEnvClient(BaseEnvClient):
    extra_hint = (
            "\n\n---\n"
            "⚠️ **Instructions** ⚠️\n"
            "Write a complete **Python 3** program that\n"
            "• reads *all* input from **STDIN** (using `input()` / `sys.stdin`),\n"
            "• writes *only* the required answer(s) to **STDOUT** using `print`,\n"
            "• contains no additional prompts or debug text, and\n"
            "• is returned as a single ```python … ``` fenced block.\n"
        )
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": "You are an expert deductive reasoning solver. Given a problem, you should solve it step by step and return the answer in the format of ```python … ``` fenced block." + extra_hint,
            }
        ),
        ConversationMessage(
            {
                "from": "gpt",
                "loss": False,
                "value": "OK. I'll follow your instructions and try my best to solve the task.",
            }
        ),
    )

    def __init__(
        self, env_server_base: str, data_len: int, *args,
        timeout: int = 300, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len
        self.id = 0
        data = dict()
        data['id'] = 0
        ok = requests.post(
            f"{self.env_server_base}/create",
            json=data,
            timeout=self.timeout,
        )
        if ok.status_code != 200:
                raise RequestException(f"Failed to create environment: {ok}")

            self.env_id = ok.json()
    
    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        data["env_idx"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()
    
    def _get(self, path: str) -> Dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> Dict[str, Any]:
        response = self._get("observation")
        return response

    def step(self, action: str) -> StepOutput:
        # action is the original output of llm
        response = self._post("step", {"action": action})
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, id: int) -> Dict[str, Any]:
        self.id = id
        response = self._post("reset", {"id": self.id})
        return response

# Typically no need to change
class DEDTask(BaseTask):
    env_client_cls = DEDEnvClient
    env_name = "DED"

    def __init__(
        self, client_args: Mapping[str, Any], n_clients: int, *args, **kwargs
    ):
        super().__init__(client_args, n_clients, *args, **kwargs)