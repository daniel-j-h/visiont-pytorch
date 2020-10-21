import argparse
from pathlib import Path

import visiont.scripts.train


def main():
    parser = argparse.ArgumentParser(prog="vision-transformer")

    subcmd = parser.add_subparsers(dest="command")
    subcmd.required = True

    Formatter = argparse.ArgumentDefaultsHelpFormatter

    train = subcmd.add_parser("train", help="trains the vision transformer", formatter_class=Formatter)
    train.add_argument("dataset", type=Path, help="path to ImageNet-like dataset directory")
    train.set_defaults(main=visiont.scripts.train.main)

    args = parser.parse_args()
    args.main(args)


if __name__ == "__main__":
    main()
