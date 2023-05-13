import stage1.main as stage1
from argparse import ArgumentParser


STAGES = [stage1,];

def main() -> None:
    parser: ArgumentParser = ArgumentParser(
        prog='Ras2Vec',
        description='Convert PNG to SVG Images',
        epilog='This tool is not yet completed');
    parser.add_argument(
        '-s', '--stage', type=int, default=-1,
    );
    stage: int = parser.parse_args().stage;

    # Execute the main function of the selected stage
    STAGES[stage].main();


if __name__ == '__main__':
    main();

