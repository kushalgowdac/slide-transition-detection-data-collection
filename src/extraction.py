"""Extraction module: thin wrapper to call the project's main extraction script or provide reusable functions."""
import argparse
import subprocess
import sys
from pathlib import Path


def run_via_main(args):
    root_main = Path(__file__).resolve().parents[1] / 'main.py'
    if not root_main.exists():
        raise FileNotFoundError(f"main.py not found at {root_main}")
    cmd = [sys.executable, str(root_main)] + args
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description='Extraction wrapper (delegates to main.py)')
    parser.add_argument('args', nargs=argparse.REMAINDER)
    ns = parser.parse_args()
    run_via_main(ns.args)


if __name__ == '__main__':
    main()
