#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "optimizer.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Global optimum: f(0, ..., 0) = 0
double sphere_function(const double* x, int N) {
    double sum_sq = 0.0;
    for (int i = 0; i < N; ++i) { sum_sq += x[i] * x[i]; }
    return sum_sq;
}

// Global optimum: f(1, ..., 1) = 0
double rosenbrock_function(const double* x, int N) {
    double sum = 0.0;
    for (int i = 0; i < N - 1; ++i) {
        double term1 = x[i+1] - x[i]*x[i];
        double term2 = 1.0 - x[i];
        sum += 100.0 * term1 * term1 + term2 * term2;
    }
    return sum;
}

// Global optimum: f(0, ..., 0) = 0
double rastrigin_function(const double* x, int N) {
    double sum = 10.0 * N;
    for (int i = 0; i < N; ++i) {
        sum += x[i] * x[i] - 10.0 * cos(2.0 * M_PI * x[i]);
    }
    return sum;
}

// Global optimum: f(0, ..., 0) = 0
double ackley_function(const double* x, int N) {
    double sum_sq_term = 0.0;
    double sum_cos_term = 0.0;
    const double a = 20.0;
    const double b = 0.2;
    const double c = 2.0 * M_PI;
    const double invN = 1.0 / N;

    for (int i = 0; i < N; ++i) {
        sum_sq_term += x[i] * x[i];
        sum_cos_term += cos(c * x[i]);
    }

    double term1 = -a * exp(-b * sqrt(sum_sq_term * invN));
    double term2 = -exp(sum_cos_term * invN);

    return term1 + term2 + a + exp(1.0);
}

// Global optimum: f(0, ..., 0) = 0
double griewank_function(const double* x, int N) {
    double sum_term = 0.0;
    double prod_term = 1.0;

    for (int i = 0; i < N; ++i) {
        sum_term += (x[i] * x[i]) / 4000.0;
        prod_term *= cos(x[i] / sqrt((double)(i + 1)));
    }

    return sum_term - prod_term + 1.0;
}

objective_func_t func_list[] = {
    sphere_function, rosenbrock_function, rastrigin_function,
    ackley_function, griewank_function
};
const char* func_names[] = {
    "Sphere", "Rosenbrock", "Rastrigin", "Ackley", "Griewank"
};

double func_min_bounds[] = {-5.12, -5.0, -5.12, -32.768, -600.0};
double func_max_bounds[] = { 5.12, 10.0,  5.12,  32.768,  600.0};
const int NUM_FUNCTIONS = sizeof(func_list) / sizeof(func_list[0]);


void print_usage(char *prog_name) {
    printf("Usage: %s [function_index] [dimension]\n", prog_name);
    printf("  function_index (optional): Index of the function to optimize (0-%d, default: 2).\n", NUM_FUNCTIONS - 1);
    printf("  dimension (optional): Problem dimension N (e.g., 10, default: 10).\n");
    printf("\nAvailable functions:\n");
    for (int i = 0; i < NUM_FUNCTIONS; ++i) {
        printf("  %d: %s (Range: [%.1f, %.1f])\n", i, func_names[i], func_min_bounds[i], func_max_bounds[i]);
    }
}


int main(int argc, char *argv[]) {
    // Default: Rastrigin
    int selected_function_index = 2;
    int dimension_N = 10;

    if (argc > 1) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        char *endptr;
        long val = strtol(argv[1], &endptr, 10);
        if (*endptr == '\0' && val >= 0 && val < NUM_FUNCTIONS) {
            selected_function_index = (int)val;
        } else {
            fprintf(stderr, "Warning: Invalid function index '%s'. Using default %d (%s).\n",
                    argv[1], selected_function_index, func_names[selected_function_index]);
        }
    }
    if (argc > 2) {
         char *endptr;
         long val = strtol(argv[2], &endptr, 10);
         if (*endptr == '\0' && val > 0) {
             dimension_N = (int)val;
         } else {
             fprintf(stderr, "Warning: Invalid dimension '%s'. Using default %d.\n", argv[2], dimension_N);
         }
    }

    if (argc <= 1) {
        printf("No function index provided. Using default: %d (%s)\n",
               selected_function_index, func_names[selected_function_index]);
        printf("No dimension provided. Using default: %d\n", dimension_N);
        printf("Run with '-h' or '--help' for options.\n");
    }


    
    OptimizationParams params;
    params.N = dimension_N;
    params.max_evaluations = (long long)params.N * 1e5;
    params.target_fitness = 1e-8;
    params.search_range_min = func_min_bounds[selected_function_index];
    params.search_range_max = func_max_bounds[selected_function_index];

    if (params.search_range_max <= params.search_range_min) {
         fprintf(stderr, "Error: Invalid search range [%f, %f] for function %s. Check bounds.\n",
                 params.search_range_min, params.search_range_max, func_names[selected_function_index]);
         return 1;
    }

    objective_func_t selected_func = func_list[selected_function_index];
    const char* selected_func_name = func_names[selected_function_index];

    printf("\nSelected function: %s (N=%d, Index=%d)\n",
           selected_func_name, params.N, selected_function_index);
    printf("Budget: %lld evaluations, Target Fitness: %.1e\n",
           params.max_evaluations, params.target_fitness);


    printf("Starting optimization...\n");
    clock_t start_time = clock();
    OptimizationResult result = run_optimization_with_restarts(params, selected_func);
    clock_t end_time = clock();
    double elapsed_time_sec = (double)(end_time - start_time) / CLOCKS_PER_SEC;


    
    printf("\n=== Final Result for %s ===\n", selected_func_name);

    if (!result.internal_error && result.best_solution != NULL) {
        printf("Optimization completed successfully in %.2f seconds.\n", elapsed_time_sec);
        printf("Global best fitness: %.8e\n", result.best_fitness);
        printf("Target reached: %s\n", result.target_reached ? "Yes" : "No");
        printf("Global best solution (first %f dims): [", fmin(params.N, 5));
        for (int i = 0; i < fmin(params.N, 5); ++i) {
            printf("%.4f%s", result.best_solution[i], (i == fmin(params.N, 5) - 1) ? "" : ", ");
        }
        if (params.N > 5) printf(", ...");
        printf("]\n");
        printf("Total evaluations consumed: %lld / %lld\n", result.total_evaluations, params.max_evaluations);
        printf("Number of restarts performed: %d\n", result.restarts);

        free(result.best_solution);
        printf("Optimizer solution resources freed.\n");

    } else if (result.internal_error) {
        printf("Optimization failed after %.2f seconds due to an internal error.\n", elapsed_time_sec);
        printf("Check previous error messages from the optimizer library.\n");
    } else {
        printf("Optimization finished after %.2f seconds, but no solution was returned.\n", elapsed_time_sec);
        printf("Total evaluations consumed: %lld / %lld\n", result.total_evaluations, params.max_evaluations);
        printf("Number of restarts performed: %d\n", result.restarts);
        printf("Best fitness found: %.8e\n", result.best_fitness);
    }

    return result.internal_error ? 1 : 0;
}