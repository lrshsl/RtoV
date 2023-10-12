
import argparse


def get_parser() -> argparse.ArgumentParser:
    """Creates an ArgumentParser for rtov.
    The returned object has the following attributes:
        mode: 'train' or 'convert'

    mode == 'test'
        load_model: str, model that should be tested
        show_examples: bool, show an example batch
        hide_plot: bool, hide plotting window
        result_save_path: str, path to save the results to (if not given, don't save)
        batch_size: int, batch size
        total_num_samples: int, total number of samples
        num_workers: int, number of workers

    mode == 'train'
        base_model: str, model that should be trained
        model_save_path: str, path to save the model to (if not given, don't save)
        num_workers: int, number of workers
        hide_plot: bool, don't show the plotting window at the end
        num_epochs: int, number of epochs
        batch_size: int, batch size
        learning_rate: float, learning rate
        learning_momentum: float, learning momentum
        # weight_decay: float, weight decay
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

    # Model
    test_mode_parser.add_argument(
            '-m', '--model',
            dest='load_model',
            help='Which model to test (from the saved_models folder, without extension)',
            type=str,
            default=None,
            required=False)

    # Hide plot
    test_mode_parser.add_argument(
            '--hide-plot',
            dest='hide_plot',
            help='Hide plotting window',
            action='store_true',
            default=False,
            required=False)

    # Plot save path
    test_mode_parser.add_argument(
            '-o', '--output',
            dest='result_save_path',
            help='Which file to save the results to',
            type=argparse.FileType('wb'),
            default=None,
            required=False)

    # Batch size
    test_mode_parser.add_argument(
            '--batch-size',
            dest='batch_size',
            help='Batch size',
            type=int,
            default=4,
            required=False)

    # Number samples
    test_mode_parser.add_argument(
            '-n', '--num-samples',
            dest='total_num_samples',
            help='Number of samples to use for the testing',
            type=int,
            default=2000,
            required=False)

    # Number of workers
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

    # Model
    train_mode_parser.add_argument(
            '-m', '--model',
            dest='base_model',
            help='Which pretrained model to test (Empty: train a new model from scratch)',
            type=str,
            default=None,
            required=False)

    # Model save path
    train_mode_parser.add_argument(
            '-o', '--output',
            dest='model_save_path',
            help='Where to save the trained model',
            type=argparse.FileType('wb'),
            default=None,
            required=False)

    # Number of workers
    train_mode_parser.add_argument(
            '-j', '--jobs',
            dest='num_workers',
            help='Number of workers to use for the dataloading',
            type=int,
            default=8,
            required=False)

    # Hide plot
    train_mode_parser.add_argument(
            '--hide-plot',
            dest='hide_plot',
            help='Do not show plotting window after training',
            action='store_true',
            default=False,
            required=False)

    # Number of epochs
    train_mode_parser.add_argument(
            '-e', '--epochs',
            dest='num_epochs',
            help='Number of epochs',
            type=int,
            default=10,
            required=False)

    # Batch size
    train_mode_parser.add_argument(
            '-b', '--batch-size',
            dest='batch_size',
            help='Batch size (default 4)',
            type=int,
            default=4,
            required=False)


    # Learning rate
    train_mode_parser.add_argument(
            '-lr', '--learning-rate',
            dest='learning_rate',
            help='Learning rate (default 0.00005)',
            type=float,
            default=0.00005,
            required=False)

    # Learning momentum
    train_mode_parser.add_argument(
            '-lm', '--learning-momentum',
            dest='learning_momentum',
            help='Learning momentum (default 0.9)',
            type=float,
            default=0.9,
            required=False)

    # # Weight decay
    # train_mode_parser.add_argument(
    #         '-wd', '--weight-decay',
    #         dest='weight_decay',
    #         help='Weight decay (default 0.0)',
    #         type=float,
    #         default=0.0,
    #         required=False)
    # }}}

    # Convert mode {{{

    # FIX: Not yet implemented
    # convert_mode_parser: argparse.ArgumentParser = mode_parsers.add_parser(
    #         'convert',
    #         help='Convert a raster image to a vector',
    #         description='Convert a raster image to a vector',
    #         epilog='Find more info at https://github.com/lrshsl/RtoV')
    # convert_mode_parser.add_argument(
    #         'input_image',
    #         help='Input image',
    #         type=argparse.FileType('rb'))
    # convert_mode_parser.add_argument(
    #     '-o', '--output',
    #     dest='output_image',
    #     help='Output image',
    #     type=argparse.FileType('wb'),
    #     required=True)
    # }}}

    return argparser
