#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include "cmaes.h"

typedef double (*objective_func_t)(const double*, int);

double sphere_function(const double* x, int N) {
    double sum_sq = 0.0;
    for (int i = 0; i < N; ++i) { sum_sq += x[i] * x[i]; }
    return sum_sq;
}

double rosenbrock_function(const double* x, int N) {
    double sum = 0.0;
    for (int i = 0; i < N - 1; ++i) {
        double term1 = x[i+1] - x[i]*x[i];
        double term2 = 1.0 - x[i];
        sum += 100.0 * term1 * term1 + term2 * term2;
    }
    return sum;
}

double rastrigin_function(const double* x, int N) {
    double sum = 10.0 * N;
    for (int i = 0; i < N; ++i) {
        sum += x[i] * x[i] - 10.0 * cos(2.0 * M_PI * x[i]);
    }
    return sum;
}

double ackley_function(const double* x, int N) {
    double sum_sq_term = 0.0;
    double sum_cos_term = 0.0;
    double a = 20.0;
    double b = 0.2;
    double c = 2.0 * M_PI;

    for (int i = 0; i < N; ++i) {
        sum_sq_term += x[i] * x[i];
        sum_cos_term += cos(c * x[i]);
    }

    double term1 = -a * exp(-b * sqrt(sum_sq_term / N));
    double term2 = -exp(sum_cos_term / N);

    return term1 + term2 + a + exp(1.0);
}

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
    sphere_function,
    rosenbrock_function,
    rastrigin_function,
    ackley_function,
    griewank_function
};

const char* func_names[] = {
    "Sphere",
    "Rosenbrock",
    "Rastrigin",
    "Ackley",
    "Griewank"
};

double func_min_bounds[] = {-5.12, -5.0, -5.12, -32.768, -600.0};
double func_max_bounds[] = { 5.12, 10.0,  5.12,  32.768,  600.0};

const int NUM_FUNCTIONS = sizeof(func_list) / sizeof(func_list[0]);


int main(int argc, char *argv[]) {

    int selected_function_index = 2;

    if (argc > 1) {
        int requested_index = atoi(argv[1]);
        if (requested_index >= 0 && requested_index < NUM_FUNCTIONS) {
            selected_function_index = requested_index;
        } else {
            printf("Invalid function index: %d. Available indices: 0-%d.\n", requested_index, NUM_FUNCTIONS - 1);
            printf("Using default function: %s\n", func_names[selected_function_index]);
            printf("Available functions:\n");
            for(int i=0; i<NUM_FUNCTIONS; ++i) {
                printf("  %d: %s\n", i, func_names[i]);
            }
        }
    } else {
         printf("No function index provided. Using default: %s (index %d)\n", func_names[selected_function_index], selected_function_index);
         printf("To select another function, run: %s <function_index>\n", argv[0]);
         printf("Available functions:\n");
            for(int i=0; i<NUM_FUNCTIONS; ++i) {
                printf("  %d: %s\n", i, func_names[i]);
            }
    }


    objective_func_t obj_func = func_list[selected_function_index];
    const char* func_name = func_names[selected_function_index];
    double search_range_min = func_min_bounds[selected_function_index];
    double search_range_max = func_max_bounds[selected_function_index];
    double search_range_width = search_range_max - search_range_min;

    int N = 10;

    long long max_evaluations = (long long)N * 1e5;
    double target_fitness = 1e-8;

     if (selected_function_index == 1 && N > 5) {
         printf("Warning: Target %.1e may be difficult to achieve for the Rosenbrock function with N=%d within the budget.\n", target_fitness, N);
     }


    double initial_sigma_base = search_range_width / 3.0;
    int stagnation_limit_base = 200;
    double stagnation_limit_factor = 50.0;
    double sigma_threshold = 1e-10;
    double fitness_improvement_threshold = 1e-7;
    double ipop_factor = 1.5;
    int default_lambda = 4 + floor(3 * log((double)N));
    int max_lambda = default_lambda * pow(ipop_factor, 8);

    long long total_evaluations = 0;
    double global_best_fitness = INFINITY;
    double* global_best_solution = (double*)malloc(N * sizeof(double));
    if (!global_best_solution) { return 1; }
    memset(global_best_solution, 0, N * sizeof(double));

    int restart_count = 0;
    int current_lambda = 0;
    double current_initial_sigma = initial_sigma_base;

    cmaes_t* cma_state = NULL;
    srand(time(NULL));

    printf("\nStarting optimization of function: %s (N=%d, Index=%d)\n", func_name, N, selected_function_index);
    printf("Search range for initialization: [%.2f, %.2f]\n", search_range_min, search_range_max);
    printf("Evaluation budget: %lld, Target fitness: %.1e\n", max_evaluations, target_fitness);
    printf("Restart parameters: BaseSigma=%.2f, IPOP Factor=%.1f, MaxLambda=%d, StagnationFactor=%.1f\n",
           initial_sigma_base, ipop_factor, max_lambda, stagnation_limit_factor);
    printf("-----------------------------------------------------------\n");

    while (total_evaluations < max_evaluations) {
        printf("\n--- Starting %s %d ---\n", (restart_count == 0) ? "main run" : "restart", restart_count);

        int run_lambda;
        if (current_lambda <= 0) { run_lambda = 4 + floor(3 * log((double)N)); }
        else { run_lambda = current_lambda; }
        if (run_lambda > max_lambda && restart_count > 0) { run_lambda = max_lambda; }

        double* initial_mean = (double*)malloc(N * sizeof(double));
        if (!initial_mean) { free(global_best_solution); return 1; }
        for(int i=0; i<N; ++i) {
            initial_mean[i] = search_range_min + (double)rand() / RAND_MAX * search_range_width;
        }

        cma_state = cmaes_init(N, initial_mean, current_initial_sigma, current_lambda, obj_func);
        free(initial_mean);

        if (!cma_state) { free(global_best_solution); return 1; }

        printf("Run Parameters: Lambda=%d, Initial Sigma=%.2e\n", run_lambda, current_initial_sigma);

        int stagnation_limit = stagnation_limit_base + (int)floor(stagnation_limit_factor * N * N / run_lambda);
        if (stagnation_limit < 50) stagnation_limit = 50;
        printf("Stagnation limit for this run: %d generations\n", stagnation_limit);

        int stagnation_counter = 0;
        double last_checked_fitness = INFINITY;
        int generations_this_run = 0;
        int generations_since_last_check = 0;
        int check_interval = fmax(10.0, floor((double)stagnation_limit / 10.0));

        while (total_evaluations < max_evaluations) {
            if (cmaes_run_generation(cma_state) != 0) { cmaes_free(&cma_state); free(global_best_solution); return 1; }
            total_evaluations += run_lambda;
            generations_this_run++;

            double current_run_best_fitness = cmaes_get_best_fitness(cma_state);
            double current_sigma = cmaes_get_sigma(cma_state);
            int current_gen_in_run = cmaes_get_generation(cma_state);

            if (current_run_best_fitness < global_best_fitness) {
                global_best_fitness = current_run_best_fitness;
                const double* sol = cmaes_get_best_solution(cma_state);
                if (sol) { memcpy(global_best_solution, sol, N * sizeof(double)); }
                stagnation_counter = 0;
                last_checked_fitness = current_run_best_fitness;
                generations_since_last_check = generations_this_run;
            }

            int log_interval = fmax(10.0, floor(1000.0 / sqrt(run_lambda)));
            if (generations_this_run % log_interval == 0 || generations_this_run == 1) {
                printf("Gen: %5d (Run %d) | BestFit Run: %.4e | Sigma: %.2e | TotEvals: %lld | Stagn: %d/%d\n",
                       current_gen_in_run, restart_count, current_run_best_fitness,
                       current_sigma, total_evaluations, stagnation_counter, stagnation_limit);
            }

            if (global_best_fitness <= target_fitness) {
                printf("\nTarget fitness %.2e reached.\n", target_fitness);
                goto end_optimization;
            }

            if (generations_this_run - generations_since_last_check >= check_interval) {
                 int gens_passed = generations_this_run - generations_since_last_check;
                 if (last_checked_fitness - current_run_best_fitness < fitness_improvement_threshold * gens_passed) {
                     stagnation_counter += gens_passed;
                 } else { stagnation_counter = 0; }
                 last_checked_fitness = current_run_best_fitness;
                 generations_since_last_check = generations_this_run;

                 bool restart_triggered = false;
                 if (stagnation_counter >= stagnation_limit) {
                     printf("--- Restart condition detected: STAGNATION (%d >= %d generations without improvement) ---\n", stagnation_counter, stagnation_limit);
                     restart_triggered = true;
                 } else if (current_sigma < sigma_threshold) {
                     printf("--- Restart condition detected: SIGMA TOO SMALL (%.2e < %.1e) ---\n", current_sigma, sigma_threshold);
                     restart_triggered = true;
                 }
                 if (restart_triggered) { break; }
            }
             if (current_sigma < sigma_threshold / 10.0) {
                 printf("--- Restart condition detected: CRITICALLY SMALL SIGMA (%.2e) ---\n", current_sigma);
                 break;
             }
        }

        cmaes_free(&cma_state);

        if (total_evaluations < max_evaluations && global_best_fitness > target_fitness) {
            restart_count++;
            int next_lambda = (int)round(run_lambda * ipop_factor);
            current_lambda = fmin(next_lambda, max_lambda);
            current_initial_sigma = initial_sigma_base;
            printf("Preparing for restart %d: New Lambda = %d, New Sigma = %.2e\n", restart_count, current_lambda, current_initial_sigma);
        } else { break; }

    }

end_optimization:
    printf("\n-----------------------------------------------------------\n");
    if (total_evaluations >= max_evaluations && global_best_fitness > target_fitness) {
        printf("Maximum number of evaluations (%lld) reached without achieving the target.\n", max_evaluations);
    } else if (global_best_fitness <= target_fitness) {
         printf("Target fitness %.2e reached!\n", target_fitness);
    }

    printf("\n=== Final result after %d restarts ===\n", restart_count);
    printf("Objective function: %s (N=%d, Index=%d)\n", func_name, N, selected_function_index);
    printf("Global best fitness: %.8e\n", global_best_fitness);
    printf("Global best solution (first 5 dimensions): [");
    for (int i = 0; i < fmin(N, 5); ++i) {
        printf("%.4f%s", global_best_solution[i], (i == fmin(N, 5) - 1) ? "" : ", ");
    }
    if (N > 5) printf(", ...");
    printf("]\n");
    printf("Total number of evaluations: %lld / %lld\n", total_evaluations, max_evaluations);

    free(global_best_solution);
    if (cma_state != NULL) {
        cmaes_free(&cma_state);
    }
     printf("Resources freed.\n");

    return 0;
}