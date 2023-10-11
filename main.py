
import argument_parser


def main() -> None:
    args = argument_parser.get_parser().parse_args()
    if args.mode == 'train':
        import train
        # train(
        #     base_model = args.load_model,
        #     train_parameters = {
        #         'batch_size': args.batch_size,
        #         'epochs': args.epochs,
        #         'learning_rate': args.learning_rate,
        #         'weight_decay': args.weight_decay,
        #     },
        #     save_path = args.save_path)
    elif args.mode == 'test':
        from test_model import test_model, TestParameters
        test_model(
            load_model = args.load_model,
            test_parameters = TestParameters(
                num_workers = args.num_workers,
                batch_size = args.batch_size,
                total_num_samples = args.total_num_samples),
            show_examples=args.show_examples,
            hide_plot=args.hide_plot,
            result_save_path = args.result_save_path
        )
    # elif args.mode == 'convert':
        # from convert import convert
        # convert(args)


if __name__ == '__main__':
    main()
