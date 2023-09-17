import stage0.main as stage0
import stage1.main as stage1
import stage2.main as stage2
from argparse import ArgumentParser

from std.io import cout, endl


STAGES = [stage0, stage1, stage2];

def main() -> None:
    # Create the argument parser
    parser: ArgumentParser = ArgumentParser(
        prog='Ras2Vec',
        description='Convert PNG to SVG Images',
        epilog='This tool is not yet completed');
    # Add arguments
    parser.add_argument(
        '-s', '--stage', type=int, default=-1,
    );
    stage: int = parser.parse_args().stage;

    if not 0 <= stage < len(STAGES):
        cout << 'Invalid stage: ' << stage << endl;
        return;

    # Execute the main function of the selected stage
    cout << 'Executing stage ' << stage << endl;
    STAGES[stage].main();


if __name__ == '__main__':
    main();

