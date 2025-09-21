# **TextCraft Environment Architecture**

## **1. Backend Components (Python)**

### **Core Environment Layer**
```
/agentenv-textcraft/agentenv_textcraft/
├── environment.py          # Main TextCraftEnv class (Gymnasium interface)
├── crafting_tree.py        # Recipe management and crafting logic
├── utils.py               # Data structures (ItemTag, Recipe, etc.)
└── __init__.py
```

### **Server Layer** 
```
/agentenv-textcraft/agentenv_textcraft/
├── server.py              # FastAPI REST server
├── env_wrapper.py         # TextCraft_Wrapper (environment management)
├── model.py               # Pydantic request/response models
└── launch.py              # Server entrypoint
```

### **Integration Layer**
```
/agentenv/agentenv/envs/
└── textcraft.py           # TextCraftEnvClient & TextCraftTask (AgentGym integration)
```

## **2. Frontend Components (JavaScript/Vue)**

### **Visualization Core**
```
/env-visualization/src/environments/textcraft/
├── index.js               # Environment module exports & configuration
├── components/
│   ├── TextcraftViewer.vue    # Main environment viewer
│   ├── InventoryViewer.vue    # Inventory display component  
│   └── ActionInput.vue        # Action input interface
├── client/
│   ├── textcraftClient.js     # API client for TextCraft server
│   └── textcraftAPI.js        # Lower-level API functions
└── utils/
    └── TextCraftUtils.js      # UI utilities and item icons
```

### **Legacy/Backup Components**
```
/env-visualization/src/environments/textcraft/backup/
├── TextCraftPage.vue          # Complete page interface
├── TextCraftCompleteSystem.vue # Full system component
├── TextCraftInventoryViewer.vue # Inventory viewer
├── TextCraftActionInput.vue    # Action input
└── TextcraftConfig.vue        # Configuration interface
```

## **3. Component Connections & Data Flow**

### **Backend Flow**
```
1. launch.py → server.py (FastAPI app)
2. server.py → env_wrapper.py (TextCraft_Wrapper)
3. env_wrapper.py → environment.py (TextCraftEnv)
4. environment.py → crafting_tree.py (recipe logic)
5. crafting_tree.py → utils.py (data structures)
```

### **Frontend Flow**
```
1. index.js → TextcraftViewer.vue (main interface)
2. TextcraftViewer.vue → textcraftClient.js (API calls)
3. textcraftClient.js → BaseEnvClient.js (HTTP layer)
4. TextCraftUtils.js → components (UI utilities)
```

### **Integration Flow**
```
Agent Training:
agentenv/envs/textcraft.py → REST API → agentenv-textcraft server

Visualization:
Frontend Components → textcraftClient.js → REST API → agentenv-textcraft server
```

## **4. Key Relationships**

### **Environment Management**
- **TextCraft_Wrapper**: Manages multiple environment instances with thread safety
- **TextCraftEnv**: Core Gymnasium environment with crafting logic
- **CraftingTree**: Handles recipe parsing, validation, and execution

### **API Layer**
- **FastAPI Server**: RESTful endpoints (/create, /step, /reset, /observation, etc.)
- **Pydantic Models**: Type-safe request/response validation
- **CORS Support**: Optional visual mode for frontend integration

### **Client Integration**
- **TextCraftEnvClient**: Python client for agent training (implements BaseEnvClient)
- **TextCraftClient.js**: JavaScript client for visualization (extends BaseEnvClient)
- **Shared Interface**: Both clients use same REST API endpoints

### **UI Architecture**
- **Modular Design**: Separate components for viewing, inventory, and actions
- **Asset Management**: Icon mapping and resource loading
- **Event System**: State change notifications and caching

## **5. File Connections Summary**

**Core Environment**: `environment.py` ↔ `crafting_tree.py` ↔ `utils.py`

**Server**: `server.py` → `env_wrapper.py` → `environment.py`

**Training Integration**: `agentenv/envs/textcraft.py` → REST API → Server

**Visualization**: `TextcraftViewer.vue` → `textcraftClient.js` → REST API → Server

**Utilities**: `TextCraftUtils.js` provides UI support for frontend components

This architecture provides a **clean separation** between the core environment logic, server infrastructure, training integration, and visualization interface, all connected through a RESTful API layer.