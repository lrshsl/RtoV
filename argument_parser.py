
import argparse


def get_parser() -> argparse.ArgumentParser:
    """Creates an ArgumentParser for rtov.
    The returned object has the following attributes:
        mode: 'train' or 'convert'

    mode == 'test'
        load_model: str, model that should be tested
        show_examples: bool, show an example batch
        hide_plot: bool, hide plotting window
        result_save_path: str, path to save the results to (if not given, no saving)
        batch_size: int, batch size
        total_num_samples: int, total number of samples
        num_workers: int, number of workers

    mode == 'train'
        load_model: str, model that should be trained

    mode == 'convert'
        input_image: file, input image
        output_image: file, output image
    """
    argparser: argparse.ArgumentParser = argparse.ArgumentParser(
            prog='rtov',
            description='Raster to Vector converter',
            epilog='Find more info at https://github.com/lrshsl/RtoV')

    mode_parsers = argparser.add_subparsers(dest='mode')

    # Test mode {{{
    test_mode_parser = mode_parsers.add_parser(
            'test',
            help='Train the model',
            description='Train a loaded model from the saved_models folder (by default "default_model" is used',
            epilog='Find more info at https://github.com/lrshsl/RtoV')
    test_mode_parser.add_argument(
            '-m', '--model',
            dest='load_model',
            help='Which model to test (from the saved_models folder, without extension)',
            type=str,
            default=None,
            required=False)
    test_mode_parser.add_argument(
            '-show', '--show-examples',
            dest='show_examples',
            help='Show example batch',
            action='store_true',
            default=True,
            required=False)
    test_mode_parser.add_argument(
            '--hide-plot',
            dest='hide_plot',
            help='Hide plotting window',
            action='store_true',
            default=False,
            required=False)
    test_mode_parser.add_argument(
            '-o', '--output',
            dest='result_save_path',
            help='Which file to save the results to',
            type=argparse.FileType('wb'),
            default=None,
            required=False)
    test_mode_parser.add_argument(
            '--batch-size',
            dest='batch_size',
            help='Batch size',
            type=int,
            default=4,
            required=False)
    test_mode_parser.add_argument(
            '-n', '--num-samples',
            dest='total_num_samples',
            help='Number of samples to use for the testing',
            type=int,
            default=2000,
            required=False)
    test_mode_parser.add_argument(
            '-j', '--jobs',
            dest='num_workers',
            help='Number of workers to use for the dataloading',
            type=int,
            default=8,
            required=False)
    # }}}

    # Train mode {{{
    train_mode_parser = mode_parsers.add_parser(
            'train',
            help='Train the model',
            description='Train a new or loaded model',
            epilog='Find more info at https://github.com/lrshsl/RtoV')
    train_mode_parser.add_argument(
            '-m', '--model',
            dest='load_model',
            help='Which pretrained model to test',
            type=str,
            default=None,
            required=False)
    train_mode_parser.add_argument(
            '-o', '--output',
            dest='model_save_path',
            help='Where to save the trained model',
            type=argparse.FileType('wb'),
            default=None,
            required=False)

    train_mode_parser.add_argument(
            '-e', '--epochs',
            dest='num_epochs',
            help='Number of epochs',
            type=int,
            default=10,
            required=False)
    train_mode_parser.add_argument(
            '-lr', '--learning-rate',
            dest='learning_rate',
            help='Learning rate',
            type=float,
            default=0.0001,
            required=False)
    # }}}

    # Convert mode {{{
    convert_mode_parser: argparse.ArgumentParser = mode_parsers.add_parser(
            'convert',
            help='Convert a raster image to a vector',
            description='Convert a raster image to a vector',
            epilog='Find more info at https://github.com/lrshsl/RtoV')
    convert_mode_parser.add_argument(
            'input_image',
            help='Input image',
            type=argparse.FileType('rb'))
    convert_mode_parser.add_argument(
        '-o', '--output',
        dest='output_image',
        help='Output image',
        type=argparse.FileType('wb'),
        required=True)
    # }}}

    return argparser
