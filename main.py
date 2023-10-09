
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
            _load_model = args.load_model,
            num_images=args.num_images,
            test_parameters = TestParameters(
                num_workers = args.num_workers),
            show_examples=args.show_examples,
            result_save_path = args.result_save_path
        )
    # elif args.mode == 'convert':
        # from convert import convert
        # convert(args)


if __name__ == '__main__':
    main()
