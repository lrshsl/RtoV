import timeit
import matplotlib.pyplot as plt
from std.io import cout, endl

# Define the template SVG code with placeholders for RGB values
svg_template = "<svg><rect x='10' y='10' width='50' height='50' fill='rgb(%d, %d, %d)'/></svg>";
svg_template2 = "<svg><rect x='10' y='10' width='50' height='50' fill='rgb({}, {}, {})'/></svg>";
svg_template2_keywords = "<svg><rect x='10' y='10' width='50' height='50' fill='rgb({r}, {g}, {b})'/></svg>";
svg_template3_start = "<svg><rect x='10' y='10' width='50' height='50' fill='rgb(";
svg_template3_end = ")'/></svg>";

# Define a list of RGB values to iterate over
rgb_values = [(r, g, b)
              for r in range(0, 255, 255//6)
              for g in range(0, 255, 255//6)
              for b in range(0, 255, 255//6)];

# Define a function to create SVG strings using % operator
def create_svg_with_percent(r, g, b):
    return svg_template % (r, g, b);

# Define a function to create SVG strings using .format()
def create_svg_with_format(r, g, b):
    return svg_template2.format(r, g, b);

def create_svg_with_format_keywords(r, g, b):
    return svg_template2_keywords.format(r=r, g=g, b=b);

def create_svg_with_str_addition(r, g, b):
    return svg_template3_start + str(r) + "," + str(g) + "," + str(b) + svg_template3_end;

def create_svg_with_join1(r, g, b):
    return "".join((svg_template3_start, str(r), ",", str(g), ",", str(b), svg_template3_end));

def create_svg_with_join2(r, g, b):
    return "".join((svg_template3_start, ",".join((str(r), str(g), str(b))), svg_template3_end));

def create_svg_with_join3(r, g, b):
    return svg_template3_start + ",".join((str(r), str(g), str(b))) + svg_template3_end;

def create_svg_with_join4(r, g, b):
    return "".join((svg_template3_start, str(r) + "," + str(g) + "," + str(b), svg_template3_end));

def exec_benchmark(fn, count=100):
    time = timeit.timeit(lambda: [fn(*rgb) for rgb in rgb_values], number=count);
    cout << "Benchmarking function: " << fn.__name__ << ", Count: " << count << endl;
    cout << "Execution time: " << time << endl;
    return time;

def benchmark_fn(fn):
   return [
       exec_benchmark(fn, count=n)
       for n in [100, 1000, 10000, 100000]
   ];


functions = {
    "%": create_svg_with_percent,
    ".format(r)": create_svg_with_format,
    ".format(r=r)": create_svg_with_format_keywords,
    "+": create_svg_with_str_addition,
    "join for everything": create_svg_with_join1,
    "join twice separately": create_svg_with_join2,
    "join only for the comma": create_svg_with_join3,
    "join only for the string overall": create_svg_with_join4
};

for label, fn in functions.items():
    time = benchmark_fn(fn);
    plt.plot(time, label=label);

plt.legend();
plt.savefig("images/benchmark.png");

# Formatting with .format(r=r) using keyword arguments is the fastest


