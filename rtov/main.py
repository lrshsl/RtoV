
import argparse


def main() -> None:
    argparser: argparse.ArgumentParser = argparse.ArgumentParser(
            prog='rtov',
            description='Raster to Vector converter',
            epilog='Find more info at https://github.com/lrshsl/rtov')
    argparser.add_argument(
            '-h', '--help',
            help='Show this help message and exit',
            action='help')

    mode_parsers = argparser.add_subparsers()
    train_mode_parser = mode_parsers.add_parser(
            'train',
            help='Train the model',
            description='Train a new or loaded model',
            epilog='Find more info at https://github.com/lrshsl/rtov')
    train_mode_parser.add_argument(
            '-h', '--help',
            help='Show this help message and exit',
            action='help')
    train_mode_parser.add_argument(
            '-m', '--model',
            help='Load a pretrained model',
            type=str,
            required=True)

    convert_mode_parser: argparse.ArgumentParser = mode_parsers.add_parser(
            'convert',
            help='Convert a raster image to a vector',
            description='Convert a raster image to a vector',
            epilog='Find more info at https://github.com/lrshsl/rtov')
    convert_mode_parser.add_argument(
            '-h', '--help',
            help='Show this help message and exit',
            action='help')
    convert_mode_parser.add_argument(
            'image',
            help='Input image',
            type=argparse.FileType('rb'),
            required=True)
    convert_mode_parser.add_argument(
        '-o', '--output',
        help='Output image',
        type=argparse.FileType('wb'),
        required=True)

    args: argparse.Namespace = argparser.parse_args()
    print(args)


if __name__ == '__main__':
    main()
