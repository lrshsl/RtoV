
import argument_parser


def main() -> None:
    args = argument_parser.get_parser().parse_args()

    # Train mode
    if args.mode == 'train':
        from rtov.train import train_model, TrainParameters
        train_model(
            base_model = args.base_model,
            model_type_str = args.model_type,
            train_parameters = TrainParameters(
                num_workers = args.num_workers,
                num_samples = args.num_samples,
                batch_size = args.batch_size,
                epochs = args.num_epochs,
                learning_rate = args.learning_rate,
                learning_momentum = args.learning_momentum),
            hide_plot = args.hide_plot,
            model_save_name = args.model_save_path)

    # Test / evaluation mode
    elif args.mode == 'test':
        from rtov.test import test_model, TestParameters
        test_model(
            base_model = args.load_model,
            model_type_str = args.model_type,
            test_parameters = TestParameters(
                num_workers = args.num_workers,
                num_samples = args.samples_per_epoch,
                batch_size = args.batch_size),
            hide_plot=args.hide_plot,
            demonstration_save_path = args.result_save_path)

    # Conversion / execution mode
    elif args.mode == 'convert':
        from rtov.convert import convert_image
        convert_image(
            input_image_path = args.input_image,
            output_path = args.output_image,
            output_format = args.output_format,
            model = args.model,
            model_type_str = args.model_type,
        )


if __name__ == '__main__':
    main()
