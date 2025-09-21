"""
Entrypoint for the DED agent environment.
"""

import argparse
import uvicorn


def launch():

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run("agentenv_ded:app", host=args.host, port=args.port)
