"""Classifier training and evaluation utilities (minimal stub).

Implements a CLI for training/evaluation. Actual model code should be added here.
"""
import argparse
import json
from pathlib import Path


def train(args):
    print('Training stub — load dataset and implement model training in src/classifier.py')


def evaluate(args):
    print('Evaluation stub — implement evaluation here')


def main():
    parser = argparse.ArgumentParser(description='Train/evaluate classifier')
    sub = parser.add_subparsers(dest='cmd')
    p_train = sub.add_parser('train')
    p_train.add_argument('--config', help='Path to config json', required=False)
    p_eval = sub.add_parser('eval')
    p_eval.add_argument('--model', help='Path to model', required=False)
    ns = parser.parse_args()
    if ns.cmd == 'train':
        train(ns)
    elif ns.cmd == 'eval':
        evaluate(ns)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
