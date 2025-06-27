#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>

#include "cmaes.h"
#include "optimizer.h"



OptimizationResult run_optimization_with_restarts(OptimizationParams params,
                                                      objective_func_t obj_func)
{
    OptimizationResult result = {
        .best_fitness = INFINITY,
        .best_solution = NULL,
        .total_evaluations = 0,
        .restarts = 0,
        .target_reached = false,
        .internal_error = false
    };

    cmaes_t* cma_state = NULL;
    double* initial_mean = NULL;

    result.best_solution = (double*)malloc(params.N * sizeof(double));
    if (!result.best_solution) {
        fprintf(stderr, "Optimizer Error: Memory allocation failed for global_best_solution in optimizer.c\n");
        result.internal_error = true;
        return result;
    }
    memset(result.best_solution, 0, params.N * sizeof(double));


    if (params.search_range_max <= params.search_range_min) {
         fprintf(stderr, "Optimizer Error: search_range_max must be greater than search_range_min.\n");
         result.internal_error = true;
         goto cleanup_and_exit;
    }
    double search_range_width = params.search_range_max - params.search_range_min;
    double initial_sigma_base = search_range_width / 3.0;
    if (initial_sigma_base <= 0) initial_sigma_base = 1.0;

    int stagnation_limit_base = 200;
    double stagnation_limit_factor = 7.0;
    double sigma_termination_threshold = 1e-12 * initial_sigma_base;
    double fitness_improvement_threshold = 1e-11;

    double ipop_factor = 2.5;
    int default_lambda = 0;
    int max_lambda_factor = 100;
    int max_lambda_absolute = 2000;


    int current_run_lambda = 0;
    double current_initial_sigma = initial_sigma_base;
    int calculated_default_lambda = -1;


    static bool srand_called = false;
    if (!srand_called) {
        srand(time(NULL));
        srand_called = true;
    }

    printf("-----------------------------------------------------------\n");
    printf("Starting Optimization: N=%d, Budget=%lld, Target=%.3e\n",
           params.N, params.max_evaluations, params.target_fitness);
    printf("Initial Mean Range: [%.2f, %.2f], Initial Sigma ~ %.3e\n",
           params.search_range_min, params.search_range_max, initial_sigma_base);
    printf("-----------------------------------------------------------\n");


    while (result.total_evaluations < params.max_evaluations && !result.target_reached) {

        printf("\n--- Starting %s %d ---\n", (result.restarts == 0) ? "Initial Run" : "Restart", result.restarts);

        int run_lambda = (current_run_lambda <= 0) ? default_lambda : current_run_lambda;

        initial_mean = (double*)malloc(params.N * sizeof(double));
        if (!initial_mean) {
            fprintf(stderr, "Optimizer Error: Memory allocation failed for initial_mean in run %d.\n", result.restarts);
            result.internal_error = true;
            goto cleanup_and_exit;
        }
        for(int i=0; i<params.N; ++i) {
            initial_mean[i] = params.search_range_min + ((double)rand() / RAND_MAX) * search_range_width;
        }

        cma_state = cmaes_init(params.N, initial_mean, current_initial_sigma,
                                 run_lambda, 0, 0.0, 0,
                                 obj_func);
        free(initial_mean);
        initial_mean = NULL;

        if (!cma_state) {
            fprintf(stderr, "Optimizer Error: cmaes_init failed in run %d.\n", result.restarts);
            result.internal_error = true;
            goto cleanup_and_exit;
        }

        int actual_lambda_this_run = cmaes_get_lambda(cma_state);
        if (actual_lambda_this_run <= 0) {
             fprintf(stderr, "Optimizer Error: Invalid lambda (%d) reported by cmaes_state after init.\n", actual_lambda_this_run);
             result.internal_error = true;
             goto cleanup_and_exit;
        }
         if (result.restarts == 0) {
             calculated_default_lambda = actual_lambda_this_run;
         }

        int current_max_lambda = max_lambda_absolute;
        if (calculated_default_lambda > 0) {
             current_max_lambda = fmin(max_lambda_absolute, calculated_default_lambda * max_lambda_factor);
        }
        if (actual_lambda_this_run > current_max_lambda && result.restarts > 0) {
             printf("Optimizer Info: Run %d using lambda %d (limit %d).\n", result.restarts, actual_lambda_this_run, current_max_lambda);
        }


        printf("Run Parameters: Lambda=%d, Initial Sigma=%.3e\n", actual_lambda_this_run, current_initial_sigma);

        int stagnation_limit = stagnation_limit_base + (int)floor(stagnation_limit_factor * pow(params.N, 1.5) / sqrt(actual_lambda_this_run));
        stagnation_limit = fmax(50, stagnation_limit);
        printf("Stagnation limit for this run: %d generations\n", stagnation_limit);

        int stagnation_counter = 0;
        double last_checked_fitness = INFINITY;
        int generations_this_run = 0;
        int generations_since_last_check = 0;
        int check_interval = fmax(10, (int)floor((double)stagnation_limit / 10.0));


        while (result.total_evaluations < params.max_evaluations) {

            int gen_status = cmaes_run_generation(cma_state);
            if (gen_status != 0 || cmaes_get_error_flag(cma_state)) {
                 fprintf(stderr, "Optimizer Error: cmaes_run_generation failed or error flag set in run %d, generation %d.\n",
                         result.restarts, cmaes_get_generation(cma_state));
                 result.internal_error = true;
                 goto cleanup_and_exit;
            }
            result.total_evaluations += actual_lambda_this_run;
            generations_this_run++;

            double current_run_best_fitness = cmaes_get_best_fitness(cma_state);
            double current_sigma = cmaes_get_sigma(cma_state);
            int current_gen_total = cmaes_get_generation(cma_state);


            if (current_run_best_fitness < result.best_fitness) {
                result.best_fitness = current_run_best_fitness;
                const double* current_best_sol_ptr = cmaes_get_best_solution(cma_state);
                if (current_best_sol_ptr) {
                    memcpy(result.best_solution, current_best_sol_ptr, params.N * sizeof(double));
                } else {
                    fprintf(stderr, "Optimizer Warning: cmaes_get_best_solution returned NULL despite fitness update.\n");
                }
                stagnation_counter = 0;
                last_checked_fitness = current_run_best_fitness;
                generations_since_last_check = generations_this_run;

            }


            int log_interval = fmax(10, (int)floor(1000.0 / actual_lambda_this_run));
            if (generations_this_run % log_interval == 0 || generations_this_run == 1) {
                printf("Gen: %5d (Run %d) | BestFit Run: %.4e | Sigma: %.2e | TotEvals: %lld | Stagn: %d/%d\n",
                       current_gen_total, result.restarts, current_run_best_fitness,
                       current_sigma, result.total_evaluations, stagnation_counter, stagnation_limit);
            }


            if (result.best_fitness <= params.target_fitness) {
                printf("\nTarget fitness %.3e reached.\n", params.target_fitness);
                result.target_reached = true;
                goto cleanup_and_exit;
            }

            if (result.total_evaluations >= params.max_evaluations) {
                break;
            }

            bool restart_triggered = false;
            if (generations_this_run >= generations_since_last_check + check_interval) {
                int gens_passed = generations_this_run - generations_since_last_check;
                if (last_checked_fitness - current_run_best_fitness < fitness_improvement_threshold * fabs(last_checked_fitness) * gens_passed ) {
                     stagnation_counter += gens_passed;
                } else {
                     stagnation_counter = 0;
                }
                last_checked_fitness = current_run_best_fitness;
                generations_since_last_check = generations_this_run;

                if (stagnation_counter >= stagnation_limit) {
                    printf("--- Run %d restarting: Stagnation limit (%d) reached after %d generations. ---\n",
                           result.restarts, stagnation_limit, generations_this_run);
                    restart_triggered = true;
                }
            }

            if (!restart_triggered && current_sigma < sigma_termination_threshold) {
                 printf("--- Run %d restarting: Sigma (%.2e) below threshold (%.2e) after %d generations. ---\n",
                        result.restarts, current_sigma, sigma_termination_threshold, generations_this_run);
                 restart_triggered = true;
            }

            if (restart_triggered) {
                break;
            }

        } 

        cmaes_free(&cma_state);


        if (result.total_evaluations < params.max_evaluations && !result.target_reached) {
            result.restarts++;

            int next_lambda = (int)round(actual_lambda_this_run * ipop_factor);

             if (calculated_default_lambda > 0) {
                   current_max_lambda = fmin(max_lambda_absolute, calculated_default_lambda * max_lambda_factor);
                   if (next_lambda > current_max_lambda) {
                       printf("Optimizer Info: Capping next lambda at %d (was %d).\n", current_max_lambda, next_lambda);
                       next_lambda = current_max_lambda;
                   }
             } else {
                   next_lambda = fmin(next_lambda, max_lambda_absolute);
             }


            current_run_lambda = next_lambda;
            current_initial_sigma = initial_sigma_base * (0.5 + 0.5 * ((double)rand() / RAND_MAX));


            printf("Preparing for restart %d: Next Lambda = %d, Next Initial Sigma ~ %.3e\n",
                   result.restarts, current_run_lambda, current_initial_sigma);
        } else {
            break;
        }

    } 


cleanup_and_exit:
    free(initial_mean);
    cmaes_free(&cma_state);

    if (result.internal_error) {
        fprintf(stderr, "Optimizer finishing due to an internal error. Results may be incomplete.\n");
        free(result.best_solution);
        result.best_solution = NULL;
        result.best_fitness = INFINITY;
        result.target_reached = false;
    }

    printf("\n------------------- Optimization Finished -------------------\n");
    if (result.internal_error) {
         printf("Status: Internal error occurred.\n");
    } else if (result.target_reached) {
        printf("Status: Target fitness %.3e reached.\n", params.target_fitness);
    } else if (result.total_evaluations >= params.max_evaluations) {
        printf("Status: Maximum evaluation budget (%lld) reached.\n", params.max_evaluations);
    } else {
         printf("Status: Optimization loop exited unexpectedly.\n");
    }
    printf("Total evaluations: %lld\n", result.total_evaluations);
    printf("Number of restarts: %d\n", result.restarts);
    if (!result.internal_error && result.best_solution) {
        printf("Best fitness found: %.6e\n", result.best_fitness);
    } else {
         printf("Best fitness found: N/A (Error or no solution found)\n");
    }
    printf("-----------------------------------------------------------\n");


    return result;
}