import sys
from typing import Optional

class IO:
    class OStream:
        def __init__(self,
                     fhandle: Optional[any] = sys.stdout):
            self.fhandle = fhandle

        def __lshift__(self, other):
            print(other, end="", file=self.fhandle)
            return self;

cout: IO.OStream = IO.OStream();
endl = "\n";


