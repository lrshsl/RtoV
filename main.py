
import argument_parser


def main() -> None:
    args = argument_parser.get_parser().parse_args()

    # Train mode
    if args.mode == 'train':
        from model.train import train_model, TrainParameters
        train_model(
            base_model = args.base_model,
            train_parameters = TrainParameters(
                num_workers = args.num_workers,
                batch_size = args.batch_size,
                epochs = args.num_epochs,
                learning_rate = args.learning_rate,
                learning_momentum = args.learning_momentum,
        #         weight_decay: args.weight_decay,
            ),
            hide_plot = args.hide_plot,
            model_save_name = args.model_save_path)

    # Test / evaluation mode
    elif args.mode == 'test':
        from model.test import test_model, TestParameters
        test_model(
            base_model = args.load_model,
            test_parameters = TestParameters(
                num_workers = args.num_workers,
                batch_size = args.batch_size,
                total_num_samples = args.total_num_samples),
            hide_plot=args.hide_plot,
            demonstration_save_path = args.result_save_path
        )

    # Conversion / execution mode
    # elif args.mode == 'convert':
        # from convert import convert
        # convert(args)


if __name__ == '__main__':
    main()
