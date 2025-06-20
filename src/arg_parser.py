from argparse import ArgumentParser

class ArgParser:
    def __init__(self) -> None:
        self._arg_parser = ArgumentParser()
        self._arg_parser.add_argument('-e', '--example', required=True, type=str, choices=['linear', 'regularization'])
        self._arg_parser.add_argument('-i', '--iter', required=True, type=int)
        self._arg_parser.add_argument('-l', '--lr', required=True, type=float)
        self._arg_parser.add_argument('-m', '--metric', required=True, type=str, choices=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        self._arg_parser.add_argument('-r', '--reg', required=False, type=str, choices=['l1', 'l2', 'elastic'])
        self._arg_parser.add_argument('--l1', required=False, type=float)
        self._arg_parser.add_argument('--l2', required=False, type=float)

    def parse(self) -> dict:
        args = vars(self._arg_parser.parse_args())
        if args['example'] == 'regularization':
            if 'reg' not in args:
                raise Exception('Regularization example is set, but regularization type is not specified.')
            else:
                if args['reg'] == 'l1' and 'l1' not in args:
                    raise Exception('Lasso regularization is set, but L1 coefficient is not specified.')
                if args['reg'] == 'l2' and 'l2' not in args:
                    raise Exception('Ridge regularization is set, but L2 coefficient is not specified.')
                if args['reg'] == 'elastic':
                    if 'l1' not in args:
                        raise Exception('ElasticNet regularization is set, but L1 coefficient is not specified.')
                    if 'l2' not in args:
                        raise Exception('ElasticNet regularization is set, but L2 coefficient is not specified.')

        return args