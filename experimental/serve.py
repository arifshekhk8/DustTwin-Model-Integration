"""Run the experimental API alongside the unchanged v1 service."""
import argparse
import uvicorn
from service import create_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)
