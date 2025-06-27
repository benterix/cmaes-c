#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <stdbool.h>

typedef double (*objective_func_t)(const double* x, int N);

typedef struct {
    int N;
    long long max_evaluations;
    double target_fitness;
    double search_range_min;
    double search_range_max; 
} OptimizationParams;

typedef struct {
    double best_fitness;
    double* best_solution;
    long long total_evaluations;
    int restarts;
    bool target_reached;
    bool internal_error;
} OptimizationResult;


/**
 * @brief Runs the CMA-ES algorithm with an IPOP restart strategy.
 *
 * Performs Covariance Matrix Adaptation Evolution Strategy optimization,
 * automatically restarting the algorithm with increasing population sizes
 * (IPOP-CMA-ES) based on stagnation or other termination criteria, until
 * the evaluation budget is exhausted or the target fitness is reached.
 *
 * @param params Structure containing optimization parameters (dimension, budget, target, etc.).
 * @param obj_func Pointer to the objective function to be minimized.
 * @return An OptimizationResult structure containing the best solution found,
 * its fitness, evaluation count, and status flags.
 *
 * @note The 'best_solution' field in the returned OptimizationResult structure
 * is allocated dynamically. The caller is responsible for freeing this
 * memory using free() when it's no longer needed, unless an internal
 * error occurred during optimization (in which case 'best_solution'
 * will be NULL and 'internal_error' will be true).
 * Returns { .best_solution = NULL, .internal_error = true } on initial allocation failure.
 */
OptimizationResult run_optimization_with_restarts(OptimizationParams params,
                                                  objective_func_t obj_func);

#endif // OPTIMIZER_H