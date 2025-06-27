import sys
import numpy as np
import cma
import time

def sphere_function(x):
    return sum(x**2)

def rosenbrock_function(x):
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def rastrigin_function(x):
    N = len(x)
    return 10.0 * N + sum(x**2 - 10.0 * np.cos(2 * np.pi * x))

def ackley_function(x):
    N = len(x)
    a = 20.0
    b = 0.2
    c = 2.0 * np.pi
    sum_sq_term = sum(x**2)
    sum_cos_term = sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum_sq_term / N))
    term2 = -np.exp(sum_cos_term / N)
    return term1 + term2 + a + np.e

def griewank_function(x):
    N = len(x)
    sum_term = sum(x**2) / 4000.0
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, N + 1))))
    return sum_term - prod_term + 1.0

func_list = [
    sphere_function, rosenbrock_function, rastrigin_function,
    ackley_function, griewank_function
]
func_names = [
    "Sphere", "Rosenbrock", "Rastrigin", "Ackley", "Griewank"
]
func_min_bounds = np.array([-5.12, -5.0, -5.12, -32.768, -600.0])
func_max_bounds = np.array([5.12, 10.0, 5.12, 32.768, 600.0])
NUM_FUNCTIONS = len(func_list)

def print_usage():
    prog_name = sys.argv[0]
    print(f"Usage: python3 {prog_name} [function_index] [dimension]")
    print(f"  function_index (optional): Index of the function to optimize (0-{NUM_FUNCTIONS - 1}, default: 2).")
    print("  dimension (optional): Problem dimension N (e.g., 10, default: 10).")
    print("\nAvailable functions:")
    for i in range(NUM_FUNCTIONS):
        print(f"  {i}: {func_names[i]} (Range: [{func_min_bounds[i]}, {func_max_bounds[i]}])")

def main():
    selected_function_index = 2
    dimension_N = 10

    if len(sys.argv) > 1:
        if sys.argv[1] in ('-h', '--help'):
            print_usage()
            return 0
        try:
            val = int(sys.argv[1])
            if 0 <= val < NUM_FUNCTIONS:
                selected_function_index = val
            else:
                raise ValueError
        except ValueError:
            print(f"Warning: Invalid function index '{sys.argv[1]}'. Using default {selected_function_index} ({func_names[selected_function_index]}).", file=sys.stderr)

    if len(sys.argv) > 2:
        try:
            val = int(sys.argv[2])
            if val > 0:
                dimension_N = val
            else:
                raise ValueError
        except ValueError:
            print(f"Warning: Invalid dimension '{sys.argv[2]}'. Using default {dimension_N}.", file=sys.stderr)

    if len(sys.argv) <= 1:
       print(f"No function index provided. Using default: {selected_function_index} ({func_names[selected_function_index]})")
       print(f"No dimension provided. Using default: {dimension_N}")
       print("Run with '-h' or '--help' for options.")

    selected_func = func_list[selected_function_index]
    selected_func_name = func_names[selected_function_index]
    min_bound = func_min_bounds[selected_function_index]
    max_bound = func_max_bounds[selected_function_index]

    x0 = np.random.uniform(min_bound, max_bound, dimension_N)
    sigma0 = (max_bound - min_bound) / 3.0
    target_fitness = 1e-8
    max_evaluations = int(dimension_N * 1e5)
    
    options = {
        'maxfevals': max_evaluations,
        'tolfun': target_fitness,
        'verb_disp': 100
    }

    print(f"\nSelected function: {selected_func_name} (N={dimension_N}, Index={selected_function_index})")
    print(f"Budget: {max_evaluations} evaluations, Target Fitness: {target_fitness:.1e}")

    print("Starting optimization with pycma...\n")
    start_time = time.time()
    
    result = cma.fmin2(selected_func, x0, sigma0, options, restarts=9, bipop=True)
    
    end_time = time.time()
    elapsed_time_sec = end_time - start_time
    
    es = result[8]

    print("\n------------------- Optimization Finished (pycma) -------------------")
    print(f"Status: {'Target fitness reached' if es.result.fbest < target_fitness else es.result.stop.get('tolfun', 'Target not reached')}")
    print(f"Total evaluations: {es.result.evaluations}")
    print(f"Number of restarts: {es.result.restarts}")
    print(f"Best fitness found: {es.result.fbest:.8e}")
    print("---------------------------------------------------------------------")

    print(f"\n=== Final Result for {selected_func_name} (pycma) ===")
    print(f"Optimization completed successfully in {elapsed_time_sec:.2f} seconds.")
    print(f"Global best fitness: {es.result.fbest:.8e}")
    print(f"Target reached: {'Yes' if es.result.fbest < target_fitness else 'No'}")
    
    best_solution = es.result.xbest
    print(f"Global best solution (first {min(dimension_N, 5)} dims): [", end="")
    print(", ".join([f"{x:.4f}" for x in best_solution[:5]]), end="")
    if dimension_N > 5:
        print(", ...", end="")
    print("]")
    
    print(f"Total evaluations consumed: {es.result.evaluations} / {max_evaluations}")
    print(f"Number of restarts performed: {es.result.restarts}")

if __name__ == "__main__":
    main()
