from fastapi import FastAPI
import os
from .model import *
from .environment import server

app = FastAPI()

VISUAL = os.environ.get("VISUAL", "false").lower() == "true"
if VISUAL:
    print("Running in VISUAL mode")
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.middleware("http")
async def log_request_response_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logging.info(
        f"{request.client.host} - {request.method} {request.url.path} - {response.status_code} - {process_time:.2f} seconds"
    )
    return response

@app.get("/", response_model=str)
def generate_ok():
    """Test connectivity"""
    return "ok"

@app.post("/create", response_model=int)
def create(create_query: CreateQuery):
    """Create a new environment"""
    env = server.create(create_query.id)
    return env

@app.post("/step", response_model=StepResponse)
def step(step_query: StepQuery):
    # print(f"Step query: {step_query.action}")
    observation, reward, done, info = server.step(
        step_query.env_idx, step_query.action
    )
    # print(f"Observation: {observation}")
    return StepResponse(observation=observation, reward=reward, done=done,info=info)

@app.get("/observation", response_model=str)
def observation(env_idx: int):
    return server.observation(env_idx)

@app.post("/reset", response_model=str)
def reset(reset_query: ResetQuery):
    server.reset(reset_query.env_idx, reset_query.id)
    return server.observation(reset_query.env_idx)


@app.post("/close")
def close(body: CloseRequestBody):
    # print(f"/close {body.env_idx}")
    return server.close(body.env_idx)
