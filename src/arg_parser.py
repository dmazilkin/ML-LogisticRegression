from argparse import ArgumentParser
from typing import Dict, Union

class ArgParser:
    def __init__(self) -> None:
        self._arg_parser = ArgumentParser()
        self._arg_parser.add_argument('-e', '--example', required=True, type=str, choices=['linear', 'regularization'])
        self._arg_parser.add_argument('-i', '--iter', required=True, type=int)
        self._arg_parser.add_argument('-m', '--metric', required=True, type=str, choices=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        self._arg_parser.add_argument('-r', '--reg', required=False, type=str, choices=['l1', 'l2', 'elastic'])
        self._arg_parser.add_argument('--l1', required=False, type=float)
        self._arg_parser.add_argument('--l2', required=False, type=float)

    def parse(self) -> Dict[str, Union[str, float, int]]:
        self._arg_parser.parse_args()