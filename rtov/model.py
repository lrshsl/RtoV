import torch
from torch import nn

from rtov.utils.shapes import Shapes
import constants

class RtoVMainModel(torch.nn.Module):
    # Layers {{{
    def __init__(self):
        super().__init__()
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
        self.shape_fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
            nn.ReLU(),
            nn.Linear(10, len(Shapes)),
        )
        self.circle_out = 3
        self.line_out = 5
        self.triangle_out = 6
        self.rectangle_out = 5
        self.fc_circle = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
            nn.Linear(10, self.circle_out),        # Center coordinates + radius -> 3 numbers
        )
        self.fc_line = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
            nn.Linear(10, self.line_out),          # Two endpoints for line + width -> 5 coordinates
        )
        self.fc_triangle = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
            nn.Linear(10, self.triangle_out),      # Three points for triangle -> 6 coordinates
        )
        self.fc_rectangle = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
            nn.Linear(10, self.rectangle_out),     # One point + width + height + orientation -> 5 numbers
        )
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

