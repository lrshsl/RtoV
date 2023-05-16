import numpy as np
import io
from cairosvg import svg2png
from PIL import Image

from typing import Iterable, Sequence
from std.io import cout, endl

svg_code: str = """
<svg width="100" height="100">
<rect x="0" y="0" width="100" height="100" fill="rgb({r}, {g}, {b})"></rect>
</svg>
"""

class ImageGenerator:
    def __init__(self, redundancy: int = 1):
        self.redundancy = redundancy;

    def default(redundancy: int):
        return ImageGenerator(redundancy=redundancy);

    def generate(self) -> Iterable[np.ndarray]:
        for _ in range(self.redundancy):
            for r in range(0, 255, 255//6):
                arr = self.get_image_array(r, 0, 0);
                # cout << "Shape: " << arr.shape << endl;
                # cout << "R: " << r << endl;
                cout << "Arr: " << arr // 255. << endl;
                yield arr // 255., r;

    def images(self) -> Sequence[np.ndarray]:
        return [self.get_image_array(r, 0, 0) // 255.
                for r in range(0, 255, 255//6)];

    def labels(self) -> np.ndarray:
        return np.ndarray(range(0, 255, 255//6));

    @staticmethod
    def get_image_array(r, g, b):
        png_data = svg2png(
            svg_code.format(r=r, g=g, b=b)
        );
        rgb_data = Image.open(io.BytesIO(png_data));
        return np.array(rgb_data);


