import argparse
from pathlib import Path

import visiont.scripts.train
import visiont.scripts.val
import visiont.scripts.data_gen


def main():
    parser = argparse.ArgumentParser(prog="vision-transformer")

    subcmd = parser.add_subparsers(dest="command")
    subcmd.required = True

    Formatter = argparse.ArgumentDefaultsHelpFormatter

    train = subcmd.add_parser(
        "train", help="trains the vision transformer", formatter_class=Formatter
    )
    val = subcmd.add_parser(
        "val",
        help="validates the trained vision transformer",
        formatter_class=Formatter,
    )
    data_gen = subcmd.add_parser(
        "generate",
        help="generates data to see training sample diversity",
        formatter_class=Formatter,
    )
    train.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        type=Path,
        help="path to ImageNet-like dataset directory",
    )
    val.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        type=Path,
        help="path to ImageNet-like dataset directory",
    )
    val.add_argument(
        "-w",
        "--weights",
        dest="weights",
        type=Path,
        help="pre-trained model weights",
    )
    data_gen.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        type=Path,
        help="path to ImageNet-like dataset directory",
    )
    data_gen.add_argument(
        "-l",
        "--location",
        dest="location",
        type=Path,
        help="path where to store samples",
    )
    data_gen.add_argument(
        "-n",
        "--num",
        dest="num",
        type=int,
        help="number of samples to generate",
    )

    train.set_defaults(main=visiont.scripts.train.main)
    val.set_defaults(main=visiont.scripts.val.main)
    data_gen.set_defaults(main=visiont.scripts.data_gen.main)

    args = parser.parse_args()
    args.main(args)


if __name__ == "__main__":
    main()
