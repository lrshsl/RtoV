import numpy as np
from typing import Self, Union, Generator


Number = Union[np.number, int, float]

# Vec2 {{{
class Vec2:
    x: Number
    y: Number

    # Constructors {{{
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
    # }}}

    # As .. {{{
    def as_tuple(self) -> tuple[Number, Number]:
        return self.x, self.y

    def as_int_tuple(self) -> tuple[int, int]:
        return int(self.x), int(self.y)

    def as_float_tuple(self) -> tuple[float, float]:
        return float(self.x), float(self.y)

    def __iter__(self) -> Generator[Number, None, None]:
        yield self.x
        yield self.y

    def __repr__(self) -> str:
        return f'Vec2({self.x}, {self.y})'

    def __getitem__(self, index: int) -> Number:
        return self.as_tuple()[index]

    def __setitem__(self, index: int, value: Number):
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        else:
            raise IndexError
    # }}}

    # Operations {{{
    def __add__(self, other: Self) -> Self:
        """Element-wise addition"""
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Self) -> Self:
        """Element-wise subtraction"""
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, other: Self) -> Self:
        """Element-wise multiplication"""
        return Vec2(self.x * other.x, self.y * other.y)

    def __truediv__(self, other: Self) -> Self:
        """Element-wise division"""
        return Vec2(self.x / other.x, self.y / other.y)

    def __floordiv__(self, other: Self) -> Self:
        """Element-wise floor division"""
        return Vec2(self.x // other.x, self.y // other.y)

    def __pow__(self, other: Self) -> Self:
        """Element-wise power"""
        return Vec2(self.x ** other.x, self.y ** other.y)

    def __abs__(self) -> Self:
        """Absolute value"""
        return Vec2(abs(self.x), abs(self.y))

    def __neg__(self) -> Self:
        """Negation"""
        return Vec2(-self.x, -self.y)

    def __eq__(self, other: Self) -> bool:      # Return Number?
        return self.x == other.x and self.y == other.y

    def __ne__(self, other: Self) -> bool:
        return self.x != other.x or self.y != other.y

    def __lt__(self, other: Self) -> bool:
        return abs(self) < abs(other)

    def __gt__(self, other: Self) -> bool:
        return abs(self) > abs(other)

    def __le__(self, other: Self) -> bool:
        return abs(self) <= abs(other)

    def __ge__(self, other: Self) -> bool:
        return abs(self) >= abs(other)
    # }}}

# }}}

# Vec3 {{{
class Vec3:
    x: Number
    y: Number
    z: Number

    # Constructors {{{
    def __init__(self, x, y, z) -> None:
        self.x = x
        self.y = y
        self.z = z
    # }}}

    # As .. {{{
    def as_tuple(self) -> tuple[Number, Number, Number]:
        return self.x, self.y, self.z

    def as_int_tuple(self) -> tuple[int, int, int]:
        return int(self.x), int(self.y), int(self.z)

    def as_float_tuple(self) -> tuple[float, float, float]:
        return float(self.x), float(self.y), float(self.z)

    def __iter__(self) -> Generator[Number, None, None]:
        yield self.x
        yield self.y
        yield self.z

    def __repr__(self) -> str:
        return f'Vec3({self.x}, {self.y}, {self.z})'

    def __getitem__(self, item: int) -> Number:
        return self.as_tuple()[item]

    def __setitem__(self, key: int, value: Number) -> None:
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value
        elif key == 2:
            self.z = value
        else:
            raise IndexError
    # }}}

    # Operations {{{
    def __add__(self, other: Self) -> Self:
        """Element-wise addition"""
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Self) -> Self:
        """Element-wise subtraction"""
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: Self) -> Self:
        """Element-wise multiplication"""
        return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)

    def __truediv__(self, other: Self) -> Self:
        """Element-wise division"""
        return Vec3(self.x / other.x, self.y / other.y, self.z / other.z)

    def __floordiv__(self, other: Self) -> Self:
        """Element-wise floor division"""
        return Vec3(self.x // other.x, self.y // other.y, self.z // other.z)

    def __pow__(self, other: Self) -> Self:
        """Element-wise power"""
        return Vec3(self.x ** other.x, self.y ** other.y, self.z ** other.z)

    def __abs__(self) -> Self:
        """Absolute value"""
        return Vec3(abs(self.x), abs(self.y), abs(self.z))

    def __neg__(self) -> Self:
        """Negation"""
        return Vec3(-self.x, -self.y, -self.z)

    def __eq__(self, other: Self) -> bool:
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __ne__(self, other: Self) -> bool:
        return self.x != other.x or self.y != other.y or self.z != other.z

    def __lt__(self, other: Self) -> bool:
        return abs(self) < abs(other)

    def __gt__(self, other: Self) -> bool:
        return abs(self) > abs(other)

    def __le__(self, other: Self) -> bool:
        return abs(self) <= abs(other)

    def __ge__(self, other: Self) -> bool:
        return abs(self) >= abs(other)
    # }}}

# }}}

# Rect {{{
class Rect:
    pos: Vec2
    size: Vec2

    # Constructors {{{
    def __init__(self, x, y, w, h) -> None:
        self.pos = Vec2(x, y)
        self.size = Vec2(w, h)
    # }}}

    # As .. {{{
    def as_tuple(self) -> tuple[Number, Number, Number, Number]:
        return (self.pos.x, self.pos.y, self.size.x, self.size.y)

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return int(self.pos.x), int(self.pos.y), int(self.size.x), int(self.size.y)

    def as_float_tuple(self) -> tuple[float, float, float, float]:
        return float(self.pos.x), float(self.pos.y), float(self.size.x), float(self.size.y)

    def __iter__(self) -> Generator[Number, None, None]:
        yield self.pos.x
        yield self.pos.y
        yield self.size.x

    def __repr__(self) -> str:
        return f'Rect({self.pos.x}, {self.pos.y}, {self.size.x}, {self.size.y})'

    def __getitem__(self, key: int) -> Number:
        return self.as_tuple()[key]

    def __setitem__(self, key: int, value: Number) -> None:
        if key == 0:
            self.pos.x = value
        elif key == 1:
            self.pos.y = value
        elif key == 2:
            self.size.x = value
        elif key == 3:
            self.size.y = value
        else:
            raise IndexError
    # }}}

    # x, y, w, h properties {{{
    @property
    def x(self) -> Number:
        return self.pos.x

    @x.setter
    def x(self, value: Number) -> None:
        self.pos.x = value


    @property
    def y(self) -> Number:
        return self.pos.y
    
    @y.setter
    def y(self, value: Number) -> None:
        self.pos.y = value


    @property
    def w(self) -> Number:
        return self.size.x

    @w.setter
    def w(self, value: Number) -> None:
        self.size.x = value


    @property
    def h(self) -> Number:
        return self.size.y

    @h.setter
    def h(self, value: Number) -> None:
        self.size.y = value
    # }}}

    # Combinations {{{
    @property
    def xw(self) -> Vec2:
        return Vec2(self.pos.x, self.size.x)

    @xw.setter
    def xw(self, value: Vec2):
        self.pos.x = value.x
        self.size.x = value.y

    @property
    def yh(self) -> Vec2:
        return Vec2(self.pos.y, self.size.y)

    @yh.setter
    def yh(self, value: Vec2):
        self.pos.y = value.x
        self.size.y = value.y
    # }}}

    # Special Rect Operations {{{
    def intersection(self, other: Self) -> Self:
        return Rect(max(self.x, other.x), max(self.y, other.y),
                    min(self.x + self.w, other.x + other.w) - max(self.x, other.x),
                    min(self.y + self.h, other.y + other.h) - max(self.y, other.y))
    
    def is_intersecting(self, other: Self) -> bool:
        return self.intersection(other).w > 0 and self.intersection(other).h > 0

    def is_inside(self, other: Self) -> bool:
        return all((self.x <= other.x, self.y <= other.y,
                    self.x + self.w >= other.x + other.w,
                    self.y + self.h >= other.y + other.h)) or all((
            other.x <= self.x, other.y <= self.y,
            other.x + other.w >= self.x + self.w,
            other.y + other.h >= self.y + self.h))
    # }}}

    # Operations {{{
    def __add__(self, other: Self) -> Self:
        """Element-wise addition"""
        return Rect(self.x + other.x, self.y + other.y,
                    self.w + other.w, self.h + other.h)

    def __sub__(self, other: Self) -> Self:
        """Element-wise subtraction"""
        return Rect(self.x - other.x, self.y - other.y,
                    self.w - other.w, self.h - other.h)

    def __mul__(self, other: Self) -> Self:
        """Element-wise multiplication"""
        return Rect(self.x * other.x, self.y * other.y,
                    self.w * other.w, self.h * other.h)

    def __truediv__(self, other: Self) -> Self:
        """Element-wise division"""
        return Rect(self.x / other.x, self.y / other.y,
                    self.w / other.w, self.h / other.h)

    def __floordiv__(self, other: Self) -> Self:
        """Element-wise floor division"""
        return Rect(self.x // other.x, self.y // other.y,
                    self.w // other.w, self.h // other.h)

    def __pow__(self, other: Self) -> Self:
        """Element-wise power"""
        return Rect(self.x ** other.x, self.y ** other.y,
                    self.w ** other.w, self.h ** other.h)

    def __abs__(self) -> Self:
        """Element-wise absolute value"""
        return Rect(abs(self.x), abs(self.y), abs(self.w), abs(self.h))

    def __neg__(self) -> Self:
        """Element-wise negation"""
        return Rect(-self.x, -self.y, -self.w, -self.h)

    def __eq__(self, other: Self) -> bool:
        return all((self.x == other.x, self.y == other.y,
                    self.w == other.w, self.h == other.h))

    def __ne__(self, other: Self) -> bool:
        return all((self.x != other.x, self.y != other.y,
                    self.w != other.w, self.h != other.h))

    def __lt__(self, other: Self) -> bool:
        return all((self.x < other.x, self.y < other.y,
                    self.w < other.w, self.h < other.h))

    def __gt__(self, other: Self) -> bool:
        return all((self.x > other.x, self.y > other.y,
                    self.w > other.w, self.h > other.h))

    def __le__(self, other: Self) -> bool:
        return all((self.x <= other.x, self.y <= other.y,
                    self.w <= other.w, self.h <= other.h))

    def __ge__(self, other: Self) -> bool:
        return all((self.x >= other.x, self.y >= other.y,
                    self.w >= other.w, self.h >= other.h))
    # }}}

# }}}


VecXd = Union[Vec2, Vec3]
Vec4 = Rect

