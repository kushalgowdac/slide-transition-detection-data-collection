"""Top-level CLI for the project. Provides entrypoints for extraction, features, classifier."""
import argparse
import subprocess
import sys
from pathlib import Path


def run_extraction(args):
    # Delegate to project root main.py for extraction
    root_main = Path(__file__).resolve().parents[1] / 'main.py'
    if not root_main.exists():
        print(f"main.py not found at {root_main}")
        sys.exit(1)
    cmd = [sys.executable, str(root_main)] + args
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(prog='src', description='Project CLI')
    sub = parser.add_subparsers(dest='cmd')

    p_ext = sub.add_parser('extraction', help='Run extraction pipeline')
    p_ext.add_argument('args', nargs=argparse.REMAINDER)

    p_feat = sub.add_parser('features', help='Compute features (stub)')
    p_feat.add_argument('args', nargs=argparse.REMAINDER)

    p_clf = sub.add_parser('classifier', help='Train / eval classifier (stub)')
    p_clf.add_argument('args', nargs=argparse.REMAINDER)

    ns = parser.parse_args()
    if ns.cmd == 'extraction':
        run_extraction(ns.args)
    elif ns.cmd == 'features':
        print('Features entrypoint — implement in src/features.py')
    elif ns.cmd == 'classifier':
        print('Classifier entrypoint — implement in src/classifier.py')
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
