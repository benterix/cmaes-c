#ifndef CMAES_H
#define CMAES_H

#include <stdbool.h>

typedef struct cmaes_state_s cmaes_t;
typedef double (*cmaes_objective_func_t)(const double* x, int N);


// --- Public API Function Prototypes ---

/**
 * @brief Initializes the CMA-ES algorithm state.
 *
 * Allocates and initializes the internal state for the CMA-ES optimizer.
 *
 * @param N The dimension of the search space. Must be > 0.
 * @param initial_mean A pointer to an array of size N containing the initial mean
 * of the search distribution. If NULL, the mean is initialized to zeros.
 * @param initial_sigma The initial step size (standard deviation). Must be > 0.
 * @param desired_lambda The desired population size (number of offspring per generation).
 * If <= 0, a default value (e.g., 4 + floor(3*log(N))) is used.
 * @param desired_mu The desired number of parents used for recombination.
 * If <= 0, a default value (e.g., lambda/2) is used. Must be <= lambda.
 * @param min_eigen_sqrt_thresh The minimum threshold for the square root of eigenvalues
 * of the covariance matrix. Used for numerical stability.
 * If <= 0.0, an internal default (e.g., 1e-8) is used.
 * @param rng_seed The seed for the internal GSL random number generator.
 * If 0, a default seeding mechanism (e.g., /dev/urandom or time-based) is used.
 * @param objective_func A pointer to the objective function to be minimized. Must not be NULL.
 * The function should take a const double* (solution vector) and int (dimension)
 * and return a double (fitness value).
 *
 * @return A pointer to the allocated cmaes_t state structure, or NULL on failure (e.g., invalid
 * arguments, memory allocation error). Check the error flag if needed.
 */
cmaes_t* cmaes_init(int N,
                    const double* initial_mean,
                    double initial_sigma,
                    int desired_lambda,
                    int desired_mu,
                    double min_eigen_sqrt_thresh,
                    unsigned long rng_seed,
                    cmaes_objective_func_t objective_func);

/**
 * @brief Runs one generation (iteration) of the CMA-ES algorithm.
 *
 * Samples a new population, evaluates their fitness using the objective function,
 * selects parents, updates the mean, step size (sigma), and covariance matrix.
 *
 * @param state A pointer to the cmaes_t state structure initialized by cmaes_init.
 * @return 0 on success, -1 on failure (e.g., NULL state, internal error like
 * numerical instability or allocation failure during the generation).
 * Check the error flag using cmaes_get_error_flag() after failure.
 */
int cmaes_run_generation(cmaes_t* state);

/**
 * @brief Frees all resources associated with the CMA-ES state.
 *
 * Deallocates memory used by the state structure, including GSL objects.
 * Sets the pointer pointed to by state_ptr to NULL.
 *
 * @param state_ptr A pointer to the cmaes_t* variable holding the state.
 */
void cmaes_free(cmaes_t** state_ptr);


// --- Accessor Functions ---

/**
 * @brief Gets a pointer to the best solution vector found so far.
 * @param state A pointer to the cmaes_t state.
 * @return A const double* pointer to the internal best solution vector, or NULL if state is NULL.
 * The data pointed to is managed by the cmaes_t state and should not be freed by the caller.
 */
const double* cmaes_get_best_solution(const cmaes_t* state);

/**
 * @brief Gets the fitness value of the best solution found so far.
 * @param state A pointer to the cmaes_t state.
 * @return The best fitness value, or INFINITY if state is NULL or no solution has been evaluated.
 */
double cmaes_get_best_fitness(const cmaes_t* state);

/**
 * @brief Gets a pointer to the current mean vector (xmean) of the search distribution.
 * @param state A pointer to the cmaes_t state.
 * @return A const double* pointer to the internal mean vector, or NULL if state is NULL.
 * The data pointed to is managed by the cmaes_t state and should not be freed by the caller.
 */
const double* cmaes_get_mean(const cmaes_t* state);

/**
 * @brief Gets the current step size (sigma).
 * @param state A pointer to the cmaes_t state.
 * @return The current sigma value, or -1.0 if state is NULL.
 */
double cmaes_get_sigma(const cmaes_t* state);

/**
 * @brief Gets the current generation number.
 * @param state A pointer to the cmaes_t state.
 * @return The current generation count (starting from 0 after init), or -1 if state is NULL.
 */
int cmaes_get_generation(const cmaes_t* state);

/**
 * @brief Gets the internal error flag status.
 * @param state A pointer to the cmaes_t state.
 * @return 0 if no error has occurred, 1 if an error has occurred (e.g., during init or run_generation).
 * Returns 1 if state is NULL.
 */
int cmaes_get_error_flag(const cmaes_t* state);

/**
 * @brief Gets the population size (lambda) being used.
 * @param state A pointer to the cmaes_t state.
 * @return The lambda value, or -1 if state is NULL.
 */
int cmaes_get_lambda(const cmaes_t* state);

/**
 * @brief Gets the parent population size (mu) being used.
 * @param state A pointer to the cmaes_t state.
 * @return The mu value, or -1 if state is NULL.
 */
int cmaes_get_mu(const cmaes_t* state);


#endif // CMAES_H