# AgentGym Tool Environment Architecture

## Overview

This document outlines the common components and architecture patterns shared across all agent environments in the AgentGym tool ecosystem. The environments follow a standardized pattern that extends the Gymnasium framework and provides unified interfaces for agent interaction.

## Core Architecture Patterns

### 1. Server-Environment Separation
Each agentenv has two main components:
- **Environment Server** (`*_environment.py`): HTTP server wrapper handling concurrent requests
- **Core Environment** (`*_env.py`): Core environment implementation with business logic

### 2. Environment Interface
All environments implement standard methods compatible with [Gymnasium](https://gymnasium.farama.org/api/env/):
- `get_info()`, `get_obs()`, `get_goal()`, `get_history()`
- `get_action_space()`, `is_done()`, `reset()`, `step()`
- `save_log()`, `update()`

## Shared Base Components

### 1. Base Environment Interface
- **File**: `/agentenv-tool/Toolusage/toolusage/environment/base_env.py`
- **Purpose**: Abstract base class defining standard environment interface
- **Note**: Inheritance check is currently commented out in registry system

### 2. Registry System  
- **File**: `/agentenv-tool/Toolusage/toolusage/common/registry.py`
- **Purpose**: Central registration system for environments, agents, LLMs, and tasks
- **Usage**: All environments use `@registry.register_environment()` decorator

### 3. Data Management
- **File**: `/agentenv-tool/Toolusage/toolusage/utils/tool/data_utils.py`
- **Class**: `ToolDataset`
- **Purpose**: Loading and parsing `.jsonl` test data files with standardized structure

### 4. Action Processing
- **File**: `/agentenv-tool/Toolusage/toolusage/utils/tool/helpers.py`
- **Core Functions**:
  - `extract_action_name_and_action_input()` - Parses action strings
  - `parse_action()` - Extracts action type and parameters
  - `extract_sheet_number()` - Sheet-specific helper
  - `is_same_location()` - Location comparison for weather
  - `contains_network_error()` - Network error detection
  - `check_credentials()` - Environment validation
  - `save_log()` - Logging utilities

### 5. Logging Infrastructure
- `/agentenv-tool/Toolusage/toolusage/utils/logging/agent_logger.py` - Agent-specific logging
- `/agentenv-tool/Toolusage/toolusage/utils/logging/logger.py` - General logging utilities

### 6. Exception Handling
- `/agentenv-tool/Toolusage/toolusage/utils/common_exception.py` - Shared exception definitions

## Implementation Patterns

### Server Component Structure
All environment servers follow this pattern:
```python
class {Domain}EnvServer:
    def __init__(self) -> None:
        self._max_id = 0
        self.env = {}
        self.dataset = ToolDataset(test_file=dataset_path)
        self._lock = threading.Lock()
    
    def create(self, id: int = 0) -> int: # Creates new environment instance
    def reset(self, env_idx, id: Optional[int] = None): # Resets environment
    def step(self, env_idx, message: str): # Processes actions
    def observation(self, env_idx): # Gets current observation
```

### Dataset Structure
Each environment expects datasets with:
- `goal`: Task objective
- `ground_truth`: Expected final state  
- `ground_truth_subgoals`: Intermediate checkpoints
- `tool`: Available tool configurations
- `init_configs`: Environment-specific initialization data

### Action Format
Universal action pattern across all environments:
```
"Action: [action_name] with Action Input: [action_parameters]"
```

### State Management
Standard state tracking:
- `states[]`: Observation history
- `action_path[]`: Action sequence tracking  
- `done`: Completion status
- `reward`: Performance metric

## Available Environments

The `agentenv-tool` package includes these domains:
- **Todo** (`agentenv_todo/`) - Task management with date/location tracking
- **Academia** (`agentenv_academia/`) - Academic paper search and analysis
- **Sheet** (`agentenv_sheet/`) - Google Sheets manipulation with retry logic
- **Movie** (`agentenv_movie/`) - Movie database queries with connection handling
- **Weather** (`agentenv_weather/`) - Weather data retrieval with location services

## Error Handling Patterns

### Network Resilience
- **Sheet & Movie**: Implement retry mechanisms for API timeouts
- **All environments**: Standardized network error detection via `contains_network_error()`

### Credential Management
- Unified credential validation via `check_credentials()`
- Environment-specific API key requirements (MOVIE_KEY, TODO_KEY, etc.)

## Episode Structure by Environment Type

### Single-Step Environments (Episode ends after one action)

**SearchQA**:
- Episode terminates immediately after providing an answer
- Task is question-answering with document retrieval
- Agent gets context documents and answers in one step

**SqlGym**:
- Episodes end after executing one SQL query
- Agent writes SQL to answer a database question
- Success/failure determined by query execution result

### Multi-Step Environments (Require multiple actions per episode)

#### Tool-based Environments (End with explicit "finish" action)

**Academia**:
- Research academic networks (PaperNet/AuthorNet)
- Multiple tool calls needed: `loadPaperNet()`, `neighbourCheck()`, `paperNodeCheck()`, etc.
- Terminates when agent calls `finish(answer)` action
- Example: `done = "Finish" in action or "finish" in action`

**Weather**:
- Weather data analysis tasks
- Multiple API calls: `get_current_location()`, `get_historical_temp()`, etc.
- Terminates with `finish(answer)` action

**Todo**:
- Todoist task management
- Multiple operations: `get_projects()`, `get_tasks()`, `complete_task()`, etc.
- Terminates with `finish(answer)` action

**Movie**:
- Movie database queries and analysis
- Multiple tool calls for movie information
- Terminates with `finish(answer)` action

**Sheet**:
- Spreadsheet operations and analysis
- Multiple sheet manipulation actions
- Terminates with `finish(answer)` action

#### Simulation Environments (Goal-based termination)

**TextCraft**:
- Minecraft-style crafting environment
- Multiple actions: `get`, `craft`, `inventory`
- Terminates when target item is crafted: `if item_obj.item_id == self.goal: terminated = True`

**BabyAI**:
- Grid-world navigation and manipulation
- Multiple actions: `turn left`, `move forward`, `pick up`, etc.
- Episodes end when task goal is achieved

**AlfWorld**:
- Household task completion
- Multiple navigation and manipulation actions
- Terminates when household task is completed

**WebArena**:
- Web browser automation
- Multiple web interactions across pages
- Terminates when web task objective is met

**WebShop**:
- E-commerce shopping simulation
- Multiple browsing and purchasing actions
- Terminates when purchase is completed or budget exceeded

**SciWorld**:
- Scientific experiment simulation
- Multiple experimental actions and observations
- Terminates when scientific goal is achieved

**Maze**:
- Grid-world maze navigation
- Multiple movement actions
- Terminates when agent reaches the goal

**Wordle**:
- Word guessing game
- Multiple guess attempts (typically 6 maximum)
- Terminates when word is guessed or attempts exhausted

### Key Pattern Differences

**Single-Step**:
- Query → Process → Answer format
- No state accumulation across actions
- Typically information retrieval or analysis tasks

**Multi-Step Tool-based**:
- Require explicit `finish(answer)` to terminate
- Build up information through multiple tool calls
- Focus on research and data gathering tasks

**Multi-Step Simulation**:
- Goal-based termination (reach target state)
- State changes through environment interaction
- Focus on control, planning, and sequential decision-making

The tool-based environments are designed for **information gathering and analysis workflows**, while simulation environments focus on **control and planning tasks**.

### **Architectural Principle: gym.Env Inheritance**

**gym.Env is only used when you actually need episodic, multi-step reinforcement learning semantics.**

- **Single-Step Environments** (SearchQA, SqlGym): Do NOT inherit from gym.Env
  - Stateless query-response pattern
  - No episode management needed
  - Simple server-based architecture suffices
  
- **Multi-Step Environments** (TextCraft, BabyAI, Tool-based envs): DO inherit from gym.Env
  - Require episode lifecycle management (`reset()`, `step()`, `done` status)
  - Need state persistence across actions
  - Benefit from standard RL interaction patterns

## Development Guidelines

When creating new environments:
1. Extend the base patterns outlined above
2. Use the registry system for registration
3. Follow the standard action format
4. Implement proper error handling and logging
5. Include domain-specific toolkits in `/utils/{domain}/`
6. Provide corresponding dataset in `/Toolusage/data/{domain}.jsonl`
7. Consider episode structure: single-step vs multi-step, tool-based vs simulation-based
