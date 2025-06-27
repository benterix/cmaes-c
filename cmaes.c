#include "cmaes.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct cmaes_state_s {
    int N;
    int lambda;
    int mu;
    int explicit_lambda;
    int explicit_mu;
    gsl_vector* weights;
    double mu_eff;
    cmaes_objective_func_t objective_func;

    double cc, cs, c1, cmu, damps;
    double chiN;
    double min_eigenvalue_sqrt_threshold;

    gsl_vector* xmean;
    double sigma;
    gsl_matrix* C;
    gsl_matrix* B;
    gsl_vector* D_diag;
    gsl_vector* pc;
    gsl_vector* ps;
    int generation;

    gsl_rng* rng;

    double* population_raw;
    double* fitness;
    int* fitness_indices;
    gsl_matrix* arz;
    gsl_vector* best_solution_so_far;
    double best_fitness_so_far;

    gsl_vector* eigen_eval;
    gsl_matrix* eigen_evec;
    gsl_eigen_symmv_workspace* eigen_workspace;

    int error_flag;
};

typedef struct {
    double fitness;
    int index;
} fitness_pair_t;

static int compare_fitness_pairs(const void* a, const void* b) {
    double diff = ((fitness_pair_t*)a)->fitness - ((fitness_pair_t*)b)->fitness;
    if (diff < 0) return -1;
    if (diff > 0) return 1;
    return 0;
}

static unsigned long get_random_seed() {
    unsigned long seed_value;
    int seed_read = 0;

    FILE* urandom = fopen("/dev/urandom", "rb");
    if (urandom) {
        if (fread(&seed_value, sizeof(seed_value), 1, urandom) == 1) {
            seed_read = 1;
        } else {
            fprintf(stderr, "CMAES Init Warning: Failed to read from /dev/urandom (errno=%d). Using fallback seed.\n", errno);
        }
        fclose(urandom);
    } else {
        fprintf(stderr, "CMAES Init Warning: Could not open /dev/urandom (errno=%d). Using fallback seed.\n", errno);
    }

    if (!seed_read) {
        seed_value = (unsigned long)time(NULL);
        seed_value ^= (unsigned long)getpid();
        seed_value ^= (unsigned long)clock();
    }
    return seed_value;
}

cmaes_t* cmaes_init(int N, const double* initial_mean, double initial_sigma,
                      int desired_lambda, int desired_mu,
                      double min_eigen_sqrt_thresh, unsigned long rng_seed,
                      cmaes_objective_func_t objective_func) {

    if (N <= 0 || initial_sigma <= 0.0 || objective_func == NULL) {
        fprintf(stderr, "CMAES Init Error: Invalid arguments (N>0, sigma>0, func!=NULL required).\n");
        return NULL;
    }

    cmaes_t* state = (cmaes_t*)calloc(1, sizeof(cmaes_t));
    if (!state) {
        perror("CMAES Init Error: Failed to allocate state structure");
        return NULL;
    }
    state->error_flag = 0;

    state->N = N;
    state->objective_func = objective_func;

    state->explicit_lambda = desired_lambda;
    state->explicit_mu = desired_mu;

    if (desired_lambda > 0) {
        state->lambda = desired_lambda;
    } else {
        state->lambda = 4 + floor(3 * log((double)N));
    }

    if (desired_mu > 0) {
        if (desired_mu > state->lambda) {
            fprintf(stderr, "CMAES Init Warning: desired_mu (%d) > lambda (%d). Setting mu to %d.\n",
                    desired_mu, state->lambda, state->lambda);
            state->mu = state->lambda;
        } else {
            state->mu = desired_mu;
        }
    } else {
        state->mu = state->lambda / 2;
    }
    if (state->mu <= 0) {
        state->mu = 1;
    }

    state->min_eigenvalue_sqrt_threshold = (min_eigen_sqrt_thresh > 0.0)
                                           ? min_eigen_sqrt_thresh
                                           : 1e-8;

    state->weights = gsl_vector_alloc(state->mu);
    state->xmean = gsl_vector_alloc(N);
    state->C = gsl_matrix_alloc(N, N);
    state->B = gsl_matrix_alloc(N, N);
    state->D_diag = gsl_vector_alloc(N);
    state->pc = gsl_vector_alloc(N);
    state->ps = gsl_vector_alloc(N);
    state->arz = gsl_matrix_alloc(state->lambda, N);
    state->best_solution_so_far = gsl_vector_alloc(N);
    state->eigen_eval = gsl_vector_alloc(N);
    state->eigen_evec = gsl_matrix_alloc(N, N);
    state->eigen_workspace = gsl_eigen_symmv_alloc(N);
    state->rng = gsl_rng_alloc(gsl_rng_default);

    state->population_raw = (double*)malloc(state->lambda * N * sizeof(double));
    state->fitness = (double*)malloc(state->lambda * sizeof(double));
    state->fitness_indices = (int*)malloc(state->lambda * sizeof(int));


    if (!state->weights || !state->xmean || !state->C || !state->B || !state->D_diag ||
        !state->pc || !state->ps || !state->arz || !state->best_solution_so_far ||
        !state->eigen_eval || !state->eigen_evec || !state->eigen_workspace || !state->rng ||
        !state->population_raw || !state->fitness || !state->fitness_indices) {
        state->error_flag = 1;
        fprintf(stderr, "CMAES Init Error: GSL or standard memory allocation failed.\n");
        cmaes_free(&state);
        return NULL;
    }


    double sum_weights = 0.0;
    for (int i = 0; i < state->mu; ++i) {
        double w = log(state->mu + 0.5) - log((double)(i + 1));
        gsl_vector_set(state->weights, i, w);
        sum_weights += w;
    }
    if (!isfinite(sum_weights) || sum_weights <= 1e-9) {
         fprintf(stderr, "CMAES Init Error: sum_weights is invalid or too small (%.3e)!\n", sum_weights);
         state->error_flag = 1; cmaes_free(&state); return NULL;
    }
    gsl_vector_scale(state->weights, 1.0 / sum_weights);

    double sum_sq_weights_double;
    gsl_blas_ddot(state->weights, state->weights, &sum_sq_weights_double);
    if (!isfinite(sum_sq_weights_double) || sum_sq_weights_double <= 1e-9) {
     fprintf(stderr, "CMAES Init Error: sum_sq_weights is invalid or too small (%.3e)!\n", sum_sq_weights_double);
         state->error_flag = 1; cmaes_free(&state); return NULL;
    }
    state->mu_eff = 1.0 / sum_sq_weights_double;
    if(!isfinite(state->mu_eff) || state->mu_eff <= 0) {
         fprintf(stderr, "CMAES Init Error: Invalid mu_eff (%.3e) calculated!\n", state->mu_eff);
         state->error_flag = 1; cmaes_free(&state); return NULL;
    }

    state->cc = (4.0 + state->mu_eff / N) / (N + 4.0 + 2.0 * state->mu_eff / N);
    state->cs = (state->mu_eff + 2.0) / (N + state->mu_eff + 5.0);
    state->c1 = 2.0 / ((N + 1.3) * (N + 1.3) + state->mu_eff);
    state->cmu = fmin(1.0 - state->c1, 2.0 * (state->mu_eff - 2.0 + 1.0 / state->mu_eff) / ((N + 2.0) * (N + 2.0) + state->mu_eff));
    state->chiN = sqrt((double)N) * (1.0 - 1.0 / (4.0 * N) + 1.0 / (21.0 * N * N));

    if(!isfinite(state->cs) || state->cs <= 0 || state->cs >= 1.0) {
        fprintf(stderr, "CMAES Init Error: Invalid cs value detected (%.3e)! Should be in (0, 1).\n", state->cs);
        state->error_flag = 1; cmaes_free(&state); return NULL;
    }
    state->damps = 1.0 + 2.0 * fmax(0.0, sqrt((state->mu_eff - 1.0) / (N + 1.0)) - 1.0) + state->cs;
    if(!isfinite(state->damps) || state->damps <= 0) {
        fprintf(stderr, "CMAES Init Error: Invalid damps value detected (%.3e)!\n", state->damps);
        state->error_flag = 1; cmaes_free(&state); return NULL;
    }
     if(!isfinite(state->cc) || state->cc <= 0 || state->cc >= 1.0 ||
        !isfinite(state->c1) || state->c1 < 0 || state->c1 > 1.0 ||
        !isfinite(state->cmu) || state->cmu < 0 || state->cmu > 1.0 ||
        !isfinite(state->chiN) || state->chiN <= 0) {
         fprintf(stderr, "CMAES Init Error: Invalid cc, cs, c1, cmu, or chiN value detected!\n");
         fprintf(stderr, "  cc=%.3e, cs=%.3e, c1=%.3e, cmu=%.3e, chiN=%.3e\n",
                 state->cc, state->cs, state->c1, state->cmu, state->chiN);
         state->error_flag = 1; cmaes_free(&state); return NULL;
     }


    if (initial_mean) {
        gsl_vector_const_view mean_view = gsl_vector_const_view_array(initial_mean, N);
        gsl_vector_memcpy(state->xmean, &mean_view.vector);
    } else {
        gsl_vector_set_zero(state->xmean);
    }
    state->sigma = initial_sigma;
    gsl_vector_set_zero(state->pc);
    gsl_vector_set_zero(state->ps);
    state->generation = 0;
    state->best_fitness_so_far = INFINITY;

    gsl_matrix_set_identity(state->C);
    gsl_matrix_set_identity(state->B);
    gsl_vector_set_all(state->D_diag, 1.0);

    unsigned long seed_value;
    if (rng_seed != 0) {
        seed_value = rng_seed;
    } else {
        seed_value = get_random_seed();
    }
    gsl_rng_set(state->rng, seed_value);


    return state;
}

int cmaes_run_generation(cmaes_t* state) {
    if (!state) {
        fprintf(stderr, "CMAES Gen Error: state is NULL.\n");
        return -1;
    }
     if (state->error_flag) {
         fprintf(stderr, "CMAES Gen Error: state is in error mode (error_flag=1).\n");
         return -1;
     }

    int N = state->N;
    int lambda = state->lambda;
    int mu = state->mu;
    gsl_rng* rng = state->rng;
    int status = 0;

    gsl_matrix* BD = NULL;
    gsl_vector* z = NULL;
    gsl_vector* BDz = NULL;
    fitness_pair_t* fitness_pairs = NULL;
    gsl_vector* x_mean_old = NULL;
    gsl_vector* step = NULL;
    gsl_vector* D_inv_diag = NULL;
    gsl_matrix* B_T = NULL;
    gsl_matrix* Temp1 = NULL;
    gsl_matrix* C_inv_sqrt = NULL;
    gsl_vector* C_inv_sqrt_step = NULL;
    gsl_matrix* rank_mu_update_term = NULL;
    gsl_vector* dy = NULL;
    gsl_matrix* rank_one_update_term = NULL;
    gsl_matrix* C_symm_copy = NULL;
    gsl_matrix* C_T = NULL;


    BD = gsl_matrix_alloc(N, N);
    if (!BD) { perror("CMAES Gen Error: alloc BD failed"); status = -1; goto cleanup; }
    gsl_matrix_memcpy(BD, state->B);
    for (int j = 0; j < N; ++j) {
        gsl_vector_view col = gsl_matrix_column(BD, j);
        double d_val = gsl_vector_get(state->D_diag, j);
         if (!isfinite(d_val) || d_val < state->min_eigenvalue_sqrt_threshold * 0.99 ) {
             fprintf(stderr, "CMAES Gen Error: Invalid D_diag[%d] = %.3e encountered during sampling.\n", j, d_val);
             status = -1; goto cleanup;
         }
        gsl_vector_scale(&col.vector, d_val);
    }

    z = gsl_vector_alloc(N);
    BDz = gsl_vector_alloc(N);
    if (!z || !BDz) {
        perror("CMAES Gen Error: alloc z/BDz failed"); status = -1; goto cleanup;
    }

    for (int k = 0; k < lambda; ++k) {
        for (int i = 0; i < N; ++i) {
            gsl_vector_set(z, i, gsl_ran_gaussian(rng, 1.0));
        }
        gsl_matrix_set_row(state->arz, k, z);

        gsl_blas_dgemv(CblasNoTrans, 1.0, BD, z, 0.0, BDz);

        gsl_vector_view xk_view = gsl_vector_view_array(state->population_raw + k*N, N);
        gsl_vector_memcpy(&xk_view.vector, state->xmean);
        gsl_blas_daxpy(state->sigma, BDz, &xk_view.vector);
    }
    gsl_vector_free(z); z = NULL;
    gsl_vector_free(BDz); BDz = NULL;
    gsl_matrix_free(BD); BD = NULL;


    for (int k = 0; k < lambda; ++k) {
        state->fitness[k] = state->objective_func(state->population_raw + k * N, N);
        if (!isfinite(state->fitness[k])) {
            fprintf(stderr, "CMAES Warning: Non-finite fitness (%.3e) for individual %d in generation %d. Setting to INFINITY.\n",
                    state->fitness[k], k, state->generation+1);
            state->fitness[k] = INFINITY;
        }
        if (state->fitness[k] < state->best_fitness_so_far) {
            state->best_fitness_so_far = state->fitness[k];
            gsl_vector_const_view best_raw_view = gsl_vector_const_view_array(state->population_raw + k * N, N);
            gsl_vector_memcpy(state->best_solution_so_far, &best_raw_view.vector);
        }
    }


    fitness_pairs = (fitness_pair_t*)malloc(lambda * sizeof(fitness_pair_t));
    if (!fitness_pairs) { perror("CMAES Gen Error: alloc fitness_pairs failed"); status = -1; goto cleanup; }
    for (int k = 0; k < lambda; ++k) {
        fitness_pairs[k].fitness = state->fitness[k];
        fitness_pairs[k].index = k;
    }
    qsort(fitness_pairs, lambda, sizeof(fitness_pair_t), compare_fitness_pairs);
    for (int k = 0; k < lambda; ++k) state->fitness_indices[k] = fitness_pairs[k].index;


    x_mean_old = gsl_vector_alloc(N);
    if (!x_mean_old) { perror("CMAES Gen Error: alloc x_mean_old failed"); status = -1; goto cleanup; }
    gsl_vector_memcpy(x_mean_old, state->xmean);

    gsl_vector_set_zero(state->xmean);
    for (int i = 0; i < mu; ++i) {
        int best_raw_idx = state->fitness_indices[i];
        double weight = gsl_vector_get(state->weights, i);
        gsl_vector_const_view selected_sol_view = gsl_vector_const_view_array(state->population_raw + best_raw_idx * N, N);
        gsl_blas_daxpy(weight, &selected_sol_view.vector, state->xmean);
    }
    free(fitness_pairs); fitness_pairs = NULL;


    step = gsl_vector_alloc(N);
    if (!step) { perror("CMAES Gen Error: alloc step failed"); status = -1; goto cleanup; }
    gsl_vector_memcpy(step, state->xmean);
    gsl_vector_sub(step, x_mean_old);
    if (!isfinite(state->sigma) || state->sigma < 1e-16) {
         fprintf(stderr, "CMAES Gen Error: Invalid sigma (%.3e) before scaling step.\n", state->sigma);
         status = -1; goto cleanup;
    }
    gsl_vector_scale(step, 1.0 / state->sigma);

    D_inv_diag = gsl_vector_alloc(N);
    B_T = gsl_matrix_alloc(N, N);
    Temp1 = gsl_matrix_alloc(N, N);
    C_inv_sqrt = gsl_matrix_alloc(N, N);
    C_inv_sqrt_step = gsl_vector_alloc(N);
    if (!D_inv_diag || !B_T || !Temp1 || !C_inv_sqrt || !C_inv_sqrt_step) {
        perror("CMAES Gen Error: Alloc failed for sigma update intermediates");
        status = -1; goto cleanup;
    }

    gsl_vector_set_all(D_inv_diag, 1.0);
    int D_diag_ok = 1;
    for(int i=0; i<N; ++i) {
        double d_val = gsl_vector_get(state->D_diag, i);
        if(!isfinite(d_val) || d_val < state->min_eigenvalue_sqrt_threshold * 0.99) {
            fprintf(stderr, "CMAES Gen Error (Gen %d): D_diag[%d] = %.3e is invalid or below threshold (%.1e) for inversion!\n",
                    state->generation + 1,
                    i,
                    d_val,
                    state->min_eigenvalue_sqrt_threshold);
            D_diag_ok = 0; break;
        }
    }
    if (!D_diag_ok) { status = -1; goto cleanup; }
    gsl_vector_div(D_inv_diag, state->D_diag);
    if(!isfinite(gsl_blas_dnrm2(D_inv_diag))) {
         fprintf(stderr, "CMAES Gen Error: D_inv_diag became non-finite after division!\n");
         status = -1; goto cleanup;
    }


    gsl_matrix_transpose_memcpy(B_T, state->B);
    gsl_matrix_memcpy(Temp1, B_T);
    for(int i=0; i<N; ++i) {
        gsl_vector_view row_i = gsl_matrix_row(Temp1, i);
        gsl_vector_scale(&row_i.vector, gsl_vector_get(D_inv_diag, i));
    }
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, state->B, Temp1, 0.0, C_inv_sqrt);


    gsl_blas_dgemv(CblasNoTrans, 1.0, C_inv_sqrt, step, 0.0, C_inv_sqrt_step);
    double C_inv_sqrt_step_norm = gsl_blas_dnrm2(C_inv_sqrt_step);
    if(!isfinite(C_inv_sqrt_step_norm)){
        fprintf(stderr, "CMAES Gen Error: C_inv_sqrt_step_norm is NAN/INF!\n");
        status = -1; goto cleanup;
    }

    double ps_coeff = sqrt(state->cs * (2.0 - state->cs) * state->mu_eff);
    if (!isfinite(ps_coeff)) {
         fprintf(stderr, "CMAES Gen Error: ps update coefficient is NAN/INF!\n");
         status = -1; goto cleanup;
    }
    gsl_vector_scale(state->ps, 1.0 - state->cs);
    gsl_blas_daxpy(ps_coeff, C_inv_sqrt_step, state->ps);
    double ps_norm = gsl_blas_dnrm2(state->ps);
    if(!isfinite(ps_norm)){
        fprintf(stderr, "CMAES Gen Error: ps_norm (after update) is NAN/INF!\n");
        status = -1; goto cleanup;
    }

    double sigma_update_exponent = (state->cs / state->damps) * (ps_norm / state->chiN - 1.0);
    if (!isfinite(sigma_update_exponent)) {
         fprintf(stderr, "CMAES Gen Error: sigma_update_exponent is not finite!\n");
         status = -1; goto cleanup;
    }

    double sigma_multiplier = exp(sigma_update_exponent);
    if (!isfinite(sigma_multiplier) || sigma_multiplier <= 1e-10) {
         fprintf(stderr, "CMAES Warning: sigma_multiplier is not finite or too small (%.3e) in gen %d. Skipping sigma update.\n",
                 sigma_multiplier, state->generation+1);
    } else {
        state->sigma *= sigma_multiplier;
    }
    if (!isfinite(state->sigma) || state->sigma <= 0) {
        fprintf(stderr, "CMAES Gen Error: New sigma is non-finite or non-positive (%.3e) after update!\n", state->sigma);
        status = -1; goto cleanup;
    }

    gsl_vector_free(D_inv_diag); D_inv_diag = NULL;
    gsl_matrix_free(B_T); B_T = NULL;
    gsl_matrix_free(Temp1); Temp1 = NULL;
    gsl_matrix_free(C_inv_sqrt); C_inv_sqrt = NULL;
    gsl_vector_free(C_inv_sqrt_step); C_inv_sqrt_step = NULL;


    double h_sig_term = ps_norm / sqrt(1.0 - pow(1.0 - state->cs, 2.0 * (state->generation + 1.0))) / state->chiN;
     if (!isfinite(h_sig_term)) {
         fprintf(stderr, "CMAES Gen Error: h_sig_term became non-finite!\n");
         status = -1; goto cleanup;
     }
    int h_sig = (h_sig_term < (1.4 + 2.0 / (N + 1.0)));

    double pc_coeff1 = 1.0 - state->cc;
    double pc_coeff2 = h_sig * sqrt(state->cc * (2.0 - state->cc) * state->mu_eff);
     if (!isfinite(pc_coeff2)) {
         fprintf(stderr, "CMAES Gen Error: pc update coefficient 2 is NAN/INF!\n");
         status = -1; goto cleanup;
     }
    gsl_vector_scale(state->pc, pc_coeff1);
    gsl_blas_daxpy(pc_coeff2, step, state->pc);

    rank_mu_update_term = gsl_matrix_alloc(N, N);
    dy = gsl_vector_alloc(N);
    if (!rank_mu_update_term || !dy) {
        perror("CMAES Gen Error: alloc for rank-mu update failed");
        status = -1; goto cleanup;
    }
    gsl_matrix_set_zero(rank_mu_update_term);

    for (int i = 0; i < mu; ++i) {
        int best_raw_idx = state->fitness_indices[i];
        double weight = gsl_vector_get(state->weights, i);
         if (weight < 0) {
         }
        gsl_vector_const_view x_selected_view = gsl_vector_const_view_array(state->population_raw + best_raw_idx * N, N);
        gsl_vector_memcpy(dy, &x_selected_view.vector);
        gsl_vector_sub(dy, x_mean_old);
        if (!isfinite(state->sigma) || state->sigma < 1e-16) {
             fprintf(stderr, "CMAES Gen Error: Invalid sigma (%.3e) before scaling dy.\n", state->sigma);
             status = -1; goto cleanup;
        }
        gsl_vector_scale(dy, 1.0 / state->sigma);
        gsl_blas_dger(weight, dy, dy, rank_mu_update_term);
    }
    gsl_vector_free(dy); dy = NULL;

    rank_one_update_term = gsl_matrix_alloc(N, N);
    if (!rank_one_update_term) {
        perror("CMAES Gen Error: alloc for rank-one update failed");
        status = -1; goto cleanup;
    }
    gsl_matrix_set_zero(rank_one_update_term);
    gsl_blas_dger(1.0, state->pc, state->pc, rank_one_update_term);


    double C_term1_coeff = (1.0 - state->c1 - state->cmu) + state->c1 * (1.0 - h_sig) * state->cc * (2.0 - state->cc);
    double C_term2_coeff = state->c1;
    double C_term3_coeff = state->cmu;

     if (!isfinite(C_term1_coeff) || !isfinite(C_term2_coeff) || !isfinite(C_term3_coeff)) {
         fprintf(stderr, "CMAES Gen Error: Coefficients for C update are non-finite!\n");
         fprintf(stderr, " C1=%.3e, Cmu=%.3e, cc=%.3e, hsig=%d\n", state->c1, state->cmu, state->cc, h_sig);
         fprintf(stderr, " Coeffs: %.3e, %.3e, %.3e\n", C_term1_coeff, C_term2_coeff, C_term3_coeff);
         status = -1; goto cleanup;
     }

    gsl_matrix_scale(state->C, C_term1_coeff);

    gsl_matrix_scale(rank_one_update_term, C_term2_coeff);
    gsl_matrix_add(state->C, rank_one_update_term);

    gsl_matrix_scale(rank_mu_update_term, C_term3_coeff);
    gsl_matrix_add(state->C, rank_mu_update_term);

    gsl_matrix_free(rank_one_update_term); rank_one_update_term = NULL;
    gsl_matrix_free(rank_mu_update_term); rank_mu_update_term = NULL;
    gsl_vector_free(step); step = NULL;
    gsl_vector_free(x_mean_old); x_mean_old = NULL;


    C_symm_copy = gsl_matrix_alloc(N, N);
    if(!C_symm_copy) { perror("CMAES Gen Error: alloc C_symm_copy failed"); status = -1; goto cleanup; }

     for(size_t i=0; i<state->C->size1; ++i) {
         for (size_t j=0; j<state->C->size2; ++j) {
             if (!isfinite(gsl_matrix_get(state->C, i, j))) {
                 fprintf(stderr, "CMAES Gen Error: Non-finite value in C[%zu,%zu] before eigen decomposition!\n", i,j);
                 status = -1; goto cleanup;
             }
         }
     }
    gsl_matrix_memcpy(C_symm_copy, state->C);
    C_T = gsl_matrix_alloc(N, N);
    if(!C_T) { perror("CMAES Gen Error: alloc C_T failed"); status = -1; goto cleanup; }
    gsl_matrix_transpose_memcpy(C_T, C_symm_copy);
    gsl_matrix_add(C_symm_copy, C_T);
    gsl_matrix_scale(C_symm_copy, 0.5);
    gsl_matrix_free(C_T); C_T = NULL;

    int eigen_status = gsl_eigen_symmv(C_symm_copy, state->eigen_eval, state->eigen_evec, state->eigen_workspace);
    gsl_matrix_free(C_symm_copy); C_symm_copy = NULL;

    if (eigen_status != 0) {
        fprintf(stderr, "CMAES Error: GSL eigen decomposition failed with status %d in generation %d.\n",
                eigen_status, state->generation + 1);
        status = -1; goto cleanup;
    }


    gsl_matrix_memcpy(state->B, state->eigen_evec);

    for (int i = 0; i < N; ++i) {
        double eval = gsl_vector_get(state->eigen_eval, i);
        double eval_sqrt;

        if (!isfinite(eval)) {
            fprintf(stderr, "CMAES Gen Error: Non-finite eigenvalue %.3e encountered in gen %d.\n",
                    eval, state->generation+1);
            status = -1; goto cleanup;
        }

        if (eval <= 0) {
             if (eval < -1e-9) {
                fprintf(stderr, "CMAES Warning: Negative eigenvalue %.3e in gen %d. Clamping sqrt to %.1e.\n",
                        eval, state->generation+1, state->min_eigenvalue_sqrt_threshold);
            }
            eval_sqrt = state->min_eigenvalue_sqrt_threshold;
        } else {
            eval_sqrt = sqrt(eval);
            if (eval_sqrt < state->min_eigenvalue_sqrt_threshold) {
                eval_sqrt = state->min_eigenvalue_sqrt_threshold;
            }
        }
         if (!isfinite(eval_sqrt)) {
             fprintf(stderr, "CMAES Gen Error: eval_sqrt became non-finite for eigenvalue %.3e in gen %d.\n",
                     eval, state->generation+1);
             status = -1; goto cleanup;
         }
        gsl_vector_set(state->D_diag, i, eval_sqrt);
    }


    state->generation++;
    status = 0;

cleanup:
    gsl_matrix_free(BD);
    gsl_vector_free(z);
    gsl_vector_free(BDz);
    free(fitness_pairs);
    gsl_vector_free(x_mean_old);
    gsl_vector_free(step);
    gsl_vector_free(D_inv_diag);
    gsl_matrix_free(B_T);
    gsl_matrix_free(Temp1);
    gsl_matrix_free(C_inv_sqrt);
    gsl_vector_free(C_inv_sqrt_step);
    gsl_matrix_free(rank_mu_update_term);
    gsl_vector_free(dy);
    gsl_matrix_free(rank_one_update_term);
    gsl_matrix_free(C_symm_copy);
    gsl_matrix_free(C_T);

    if (status == -1 && state) {
        state->error_flag = 1;
    }
    return status;
}


void cmaes_free(cmaes_t** state_ptr) {
    if (state_ptr && *state_ptr) {
        cmaes_t* state = *state_ptr;
        gsl_vector_free(state->weights);
        gsl_vector_free(state->xmean);
        gsl_matrix_free(state->C);
        gsl_matrix_free(state->B);
        gsl_vector_free(state->D_diag);
        gsl_vector_free(state->pc);
        gsl_vector_free(state->ps);
        gsl_matrix_free(state->arz);
        gsl_vector_free(state->best_solution_so_far);
        gsl_vector_free(state->eigen_eval);
        gsl_matrix_free(state->eigen_evec);
        gsl_eigen_symmv_free(state->eigen_workspace);
        gsl_rng_free(state->rng);

        free(state->population_raw);
        free(state->fitness);
        free(state->fitness_indices);

        free(state);
        *state_ptr = NULL;
    }
}

const double* cmaes_get_best_solution(const cmaes_t* state) {
    return (state && state->best_solution_so_far) ? state->best_solution_so_far->data : NULL;
}

double cmaes_get_best_fitness(const cmaes_t* state) {
    return state ? state->best_fitness_so_far : INFINITY;
}

const double* cmaes_get_mean(const cmaes_t* state) {
    return (state && state->xmean) ? state->xmean->data : NULL;
}

double cmaes_get_sigma(const cmaes_t* state) {
    return state ? state->sigma : -1.0;
}

int cmaes_get_generation(const cmaes_t* state) {
    return state ? state->generation : -1;
}

int cmaes_get_error_flag(const cmaes_t* state) {
     return state ? state->error_flag : 1;
}

int cmaes_get_lambda(const cmaes_t* state) {
     return state ? state->lambda : -1;
}

int cmaes_get_mu(const cmaes_t* state) {
     return state ? state->mu : -1;
}
