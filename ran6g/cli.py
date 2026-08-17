"""Unified command-line interface for the 6G RAN optimization system.

Examples
--------
    ran6g generate            # write the synthetic dataset CSV
    ran6g train               # train all models, print metrics
    ran6g realtime --steps 15 # run the online inference demo
    ran6g serve --port 8000   # launch the FastAPI server
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from ran6g.config import get_paths
from ran6g.logging_utils import get_logger

logger = get_logger("ran6g.cli")


def _cmd_generate(args: argparse.Namespace) -> int:
    from data.generator import SyntheticRANDataGenerator

    paths = get_paths().ensure_dirs()
    generator = SyntheticRANDataGenerator()
    out = generator.to_csv(paths.data_path)
    logger.info("Synthetic dataset generated at: %s", out)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from pipeline.trainer import train_all

    metrics = train_all()
    print(json.dumps(metrics, indent=2))
    return 0


def _cmd_realtime(args: argparse.Namespace) -> int:
    from simulation.realtime_loop import run_realtime_demo

    run_realtime_demo(steps=args.steps, sleep_s=args.sleep)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ran6g", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate", help="Generate the synthetic dataset CSV").set_defaults(func=_cmd_generate)
    sub.add_parser("train", help="Train all models and print metrics").set_defaults(func=_cmd_train)

    rt = sub.add_parser("realtime", help="Run the real-time inference demo")
    rt.add_argument("--steps", type=int, default=10, help="Number of simulation steps")
    rt.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep between steps")
    rt.set_defaults(func=_cmd_realtime)

    serve = sub.add_parser("serve", help="Launch the FastAPI server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true", help="Enable auto-reload (development)")
    serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
