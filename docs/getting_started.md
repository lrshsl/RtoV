# Getting started


## Dependencies

In order to use RtoV, a recent Python3 installation is required. It was tested and developed in a [Python 3.11.6](https://www.python.org/downloads/release/python-3116) virtual environment on Linux.
To check if it is installed and on which version, type into a terminal `python3 --version`

It is recommended to use a virtual environment with Python. One can be created using e. g. anaconda or with the following command, which should work on most platforms:
```bash
# Create a virtual environment called `.venv` in the current working directory
python3 -m venv .venv
```

For the venv to be activated, the correct file under `.venv/bin/` has to be sources. In a bash shell, the following command should work:
```
source .venv/bin/activate
```

The dependencies can directly be installed from the requirements file in the RtoV root directory:
```
pip3 install -r requirements.txt
```

## Using the program

After installing the dependencies and tools, the program should be able to be run:
```
# Display the help menu to get an overview
python3 src/main.py --help
```

As one can see, there are three different modes for RtoV:
- train: Create or load a model and train it
- test: Load a model and evaluate its performance
- convert: Run the program on a specified raster image to convert it into a vector representation

A help menu for each mode can be obtained through running `python3 src/main.py [mode] --help`, where [mode] is the given mode (without brackets).


### Training

For training an existing model (v2 exists by default):
```
python3 src/main.py train -m v2
```

The number of epochs, images per epoch, learning rate and more can be specified as well:
```
# Train v2 for 20 epochs with 200 images each, with a learning rate = 0.002
python3 src/main.py train -m v2 -e 20 -n 200 -lr 0.002
```

For experimentation with different models, a model type can be specified. Currently, there's only 'main' and 'large':
```
# Load the model 'l1' of type 'large' and save it after the training as 'l2'
python3 src/main.py train -t large -m l1  -o l1
```


### Testing

Many options of the training can be used here as well. Interesting are the different levels of verbosity through the verbose flag.
```
python3 src/main.py  --verbose 0  test  -t large  -m l1
```

It accepts a integer from 0 to 3 (0 -> quiet, 3 -> very verbose) and has to be positioned befor the mode subcommand, since it applies to all modes.





