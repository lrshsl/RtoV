import torch
from torch import nn

from utils.shapes import Shapes
import constants

class RtoVMainModel(torch.nn.Module):

    # Layers {{{
    def __init__(self):
        super().__init__()

        # Convolution layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Fully connected layers for shape prediction
        self.shape_fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
            nn.ReLU(),
            nn.Linear(10, len(Shapes)),
        )

        # Output sizes of the shapes layers
        self.circle_out = 3        # Center coordinates + radius -> 3 numbers
        self.line_out = 5          # Two endpoints for line + width -> 5 coordinates
        self.triangle_out = 6      # Three points for triangle -> 6 coordinates
        self.rectangle_out = 5     # One point + width + height + orientation -> 5 numbers

        # Separate layers for each shape
        mk_pt_layer = lambda out: (
                nn.Sequential(
                    nn.Linear(16 * 5 * 5, 120),
                    nn.Linear(120, 84),
                    nn.Linear(84, 10),
                    nn.Linear(10, out))
        )
        self.fc_circle = mk_pt_layer(self.circle_out)
        self.fc_line = mk_pt_layer(self.line_out)
        self.fc_triangle = mk_pt_layer(self.triangle_out)
        self.fc_rectangle = mk_pt_layer(self.rectangle_out)
    # }}}

    # Forward {{{
    def forward(self, x):

        # Apply convolutions
        x = self.conv1(x)
        x = self.conv2(x)

        # Flatten
        x = torch.flatten(x, 1)     # Flatten all dimensions except batch

        # Predict shape
        shapes = self.shape_fc(x)

        # Predict color
        # TODO

        # Predict points
        out = torch.zeros((x.shape[0], constants.SHAPE_SPEC_MAX_SIZE), dtype=torch.float32)

        # Predict points, per sample
        for shape_pred in shapes:

            # Output size depends on shape
            match shape_pred.argmax():
                case Shapes.Circle:
                    out[:, :self.circle_out] = self.fc_circle(x)
                case Shapes.Line:
                    out[:, :self.line_out] = self.fc_line(x)
                case Shapes.Triangle:
                    out[:, :self.triangle_out] = self.fc_triangle(x)
                case Shapes.Rectangle:
                    out[:, :self.rectangle_out] = self.fc_rectangle(x)
                case _:
                    raise ValueError("Unknown or unimplemented shape: {}".format(shape_pred))

        return shapes, out
    # }}}




class RtoVLargeModel(RtoVMainModel):

    # Layers {{{
    def __init__(self):
        super().__init__()

        # Convolution layers
        self.conv3 = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # -> 16 x 16 x 6
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # -> 8 x 8 x 16
        )

        # Separate layers for each shape
        mk_pt_layer = lambda out: (
                nn.Sequential(
                    nn.Linear(16 * 5 * 5, 120),
                    nn.Linear(120, 84),
                    nn.Linear(84, 10),
                    nn.Linear(10, out))
        )
        self.fc_circle = mk_pt_layer(self.circle_out)
        self.fc_line = mk_pt_layer(self.line_out)
        self.fc_triangle = mk_pt_layer(self.triangle_out)
        self.fc_rectangle = mk_pt_layer(self.rectangle_out)
    # }}}

    # Forward {{{
    def forward(self, x_inp):

        # Apply convolutions
        x = self.conv1(x_inp)
        x = self.conv2(x)

        # Flatten
        x = torch.flatten(x, 1)     # Flatten all dimensions except batch

        # Predict shape
        shapes = self.shape_fc(x)

        # Predict points
        x = self.conv3(x_inp)
        x = self.conv4(x)
        x = torch.flatten(x, 1)
        out = torch.zeros((x.shape[0], constants.SHAPE_SPEC_MAX_SIZE), dtype=torch.float32)

        # Predict points, per sample
        for shape_pred in shapes:

            # Output size depends on shape
            match shape_pred.argmax():
                case Shapes.Circle:
                    out[:, :self.circle_out] = self.fc_circle(x)
                case Shapes.Line:
                    out[:, :self.line_out] = self.fc_line(x)
                case Shapes.Triangle:
                    out[:, :self.triangle_out] = self.fc_triangle(x)
                case Shapes.Rectangle:
                    out[:, :self.rectangle_out] = self.fc_rectangle(x)
                case _:
                    raise ValueError("Unknown or unimplemented shape: {}".format(shape_pred))

        return shapes, out
    # }}}


