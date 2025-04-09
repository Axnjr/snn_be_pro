<h1 align="center">A lightning fast, header-only machine learning library
</h1>

**snn-core** is an intuitive, fast, and flexible header-only C++ machine learning
library with bindings to other languages.  It is meant to be a machine learning
analog to LAPACK, and aims to implement a wide array of machine learning methods
and functions as a "swiss army knife" for machine learning researchers.

snn-core's lightweight C++ implementation makes it ideal for deployment, and it
can also be used for interactive prototyping via C++ notebooks (these can be
seen in action on mlpack's [homepage](https://www.mlpack.org/)).

In addition to its powerful C++ interface, mlpack also provides command-line
programs, Python bindings, Julia bindings, Go bindings and R bindings.

***Quick links:***

 - Quickstart guides: [C++](doc/quickstart/cpp.md),
   [CLI](doc/quickstart/cli.md), [Python](doc/quickstart/python.md),
   [R](doc/quickstart/r.md), [Julia](doc/quickstart/julia.md),
   [Go](doc/quickstart/go.md)
 - [Examples repository](https://github.com/mlpack/examples/)
 - [Tutorials](doc/user/tutorials.md)


## 0. Contents

 1. [Dependencies](#2-dependencies)
 2. [Installation](#3-installation)
 3. [Usage from C++](#4-usage-from-c)
     1. [Reducing compile time](#41-reducing-compile-time)
 4. [Building mlpack's test suite](#5-building-mlpacks-test-suite)
 5. [Further resources](#6-further-resources)

## 1. Dependencies

**mlpack** requires the following additional dependencies:

 - C++17 compiler
 - [Armadillo](https://arma.sourceforge.net)      &nbsp;&emsp;>= 10.8
 - [ensmallen](https://ensmallen.org)      &emsp;>= 2.10.0
 - [cereal](http://uscilab.github.io/cereal/)         &ensp;&nbsp;&emsp;&emsp;>= 1.1.2

If the STB library headers are available, image loading support will be
available.

If you are compiling Armadillo by hand, ensure that LAPACK and BLAS are enabled.

## 3. Installation

Detailed installation instructions can be found on the
[Installing mlpack](doc/user/install.md) page.

## 4. Usage from C++

Once headers are installed with `make install`, using mlpack in an application
consists only of including it.  So, your program should include mlpack:

```c++
#include <mlpack.hpp>
```

and when you link, be sure to link against Armadillo.  If your example program
is `my_program.cpp`, your compiler is GCC, and you would like to compile with
OpenMP support (recommended) and optimizations, compile like this:

```sh
g++ -O3 -std=c++17 -o my_program my_program.cpp -larmadillo -fopenmp
```

Note that if you want to serialize (save or load) neural networks, you should
add `#define MLPACK_ENABLE_ANN_SERIALIZATION` before including `<mlpack.hpp>`.
If you don't define `MLPACK_ENABLE_ANN_SERIALIZATION` and your code serializes a
neural network, a compilation error will occur.

***Warning:*** older versions of OpenBLAS (0.3.26 and older) compiled to use
pthreads may use too many threads for computation, causing significant slowdown.
OpenBLAS versions compiled with OpenMP do not suffer from this issue.  See the
[test build guide](doc/user/install.md#build-tests) for more details and simple
workarounds.

See also:

 * the [test program compilation section](doc/user/install.md#compiling-a-test-program)
   of the installation documentation,
 * the [C++ quickstart](doc/quickstart/cpp.md), and
 * the [examples repository](https://github.com/mlpack/examples) repository for
   some examples of mlpack applications in C++, with corresponding `Makefile`s.

### 4.1. Reducing compile time

mlpack is a template-heavy library, and if care is not used, compilation time of
a project can be very high.  Fortunately, there are a number of ways to reduce
compilation time:

 * Include individual headers, like `<mlpack/methods/decision_tree.hpp>`, if you
   are only using one component, instead of `<mlpack.hpp>`.  This reduces the
   amount of work the compiler has to do.

 * Only use the `MLPACK_ENABLE_ANN_SERIALIZATION` definition if you are
   serializing neural networks in your code.  When this define is enabled,
   compilation time will increase significantly, as the compiler must generate
   code for every possible type of layer.  (The large amount of extra
   compilation overhead is why this is not enabled by default.)

 * If you are using mlpack in multiple .cpp files, consider using [`extern
   templates`](https://isocpp.org/wiki/faq/cpp11-language-templates) so that the
   compiler only instantiates each template once; add an explicit template
   instantiation for each mlpack template type you want to use in a .cpp file,
   and then use `extern` definitions elsewhere to let the compiler know it
   exists in a different file.

Other strategies exist too, such as precompiled headers, compiler options,
[`ccache`](https://ccache.dev), and others.

## 5. Building mlpack's test suite

See the [installation instruction section](doc/user/install.md#build-tests).

## 6. Further Resources

More documentation is available for both users and developers.

 * [Documentation homepage](https://www.mlpack.org/doc/index.html)

To learn about the development goals of mlpack in the short- and medium-term
future, see the [vision document](https://www.mlpack.org/papers/vision.pdf).

If you have problems, find a bug, or need help, you can try visiting
the [mlpack help](https://www.mlpack.org/questions.html) page, or [mlpack on
Github](https://github.com/mlpack/mlpack/).  Alternately, mlpack help can be
found on Matrix at `#mlpack`; see also the
[community](https://www.mlpack.org/doc/developer/community.html) page.
