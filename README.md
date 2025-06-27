# C CMA-ES Optimizer with IPOP Restarts

This is a very basic implementation of the CMA-ES algorithm with IPOP using GSL for math operations.

## Prerequisites

* C Compiler
* GNU Scientific Library (GSL): you need the GSL development files installed (apt-get install libgsl-dev on Debian/Ubuntu, sudo dnf install gsl-devel Fedora/CentOS/RHEL, brew install gsl for macOS, or install from source - [GSL website](https://www.gnu.org/software/gsl/)).

## Building

This project uses a `Makefile` for easy compilation on Linux and macOS.
Jsut clone the repository and compile using Make.

## Usage

### 1. Including in your project:

* Copy `cmaes.h`, `cmaes.c`, `optimizer.h`, `optimizer.c` into your project source directory.
* Include the headers in your C code:
    ```c
    #include "cmaes.h"
    #include "optimizer.h"
    ```
* Compile your project along with `cmaes.c` and `optimizer.c`, linking against GSL and the math library.
    Example:
    ```bash
    gcc your_file.c cmaes.c optimizer.c -o your_app -lm -lgsl -lgslcblas -Wall -O2
    ```

### 2. Using the Optimizer (`optimizer.h`):

This might be the easiest way to use the library - via the IPOP-CMA-ES interface provided by `optimizer.h`:

```c
#include "optimizer.h"
#include <stdio.h>
#include <stdlib.h>

double my_objective_function(const double* x, int N) {
    double sum_sq = 0.0;
    for (int i = 0; i < N; ++i) { sum_sq += x[i] * x[i]; }
    return sum_sq;
}

int main() {
    OptimizationParams params;
    params.N = 10;                             // Dimension
    params.max_evaluations = 100000;           // Evaluation budget
    params.target_fitness = 1e-8;              // Target fitness value
    params.search_range_min = -5.0;            // Lower bound for initial mean
    params.search_range_max = 5.0;             // Upper bound for initial mean

    OptimizationResult result = run_optimization_with_restarts(params, my_objective_function);

    if (!result.internal_error && result.best_solution != NULL) {
        printf("Optimization successful.\n");
        printf("Best fitness: %e\n", result.best_fitness);
        printf("Evaluations: %lld\n", result.total_evaluations);
        printf("Restarts: %d\n", result.restarts);
        // Use result.best_solution...

        free(result.best_solution);
    } else {
        printf("Optimization failed or encountered an internal error.\n");
    }

    return 0;
}
```

### 3. Using CMA-ES Directly (`cmaes.h`):

You can also use the core CMA-ES functions directly if you don't need the IPOP restart strategy but the results will be worse. See `cmaes.h` for function documentation and `optimizer.c` for example usage patterns.

## Example Program

The file `main_benchmark.c` provides an example of how to use the `run_optimization_with_restarts` function to optimize several standard benchmark functions (Sphere, Rosenbrock, Rastrigin, etc.).

Build it using `make` and run it like this:

```bash
./benchmark_test [function_index] [dimension]
```

Example: `./benchmark_test 2 10` (Optimize Rastrigin in 10 dimensions)

Run `./benchmark_test -h` to see the list of available functions and their indices.
