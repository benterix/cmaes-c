#!/usr/bin/env python3

import argparse, subprocess, re, time, textwrap, os, sys
import numpy as np
import cma

def sphere(x):     return np.sum(x ** 2)
def rosenbrock(x): return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
def rastrigin(x):  return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
def ackley(x):
    n = len(x); a, b, c = 20.0, 0.2, 2 * np.pi
    s1 = np.sum(x ** 2); s2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(s1 / n)) - np.exp(s2 / n) + a + np.e
def griewank(x):
    idx = np.arange(1, len(x) + 1)
    return np.sum(x ** 2) / 4000.0 - np.prod(np.cos(x / np.sqrt(idx))) + 1

FUNC_LIST  = [sphere, rosenbrock, rastrigin, ackley, griewank]
FUNC_NAMES = ["Sphere", "Rosenbrock", "Rastrigin", "Ackley", "Griewank"]
BOUNDS_MIN = np.array([-5.12,  -5.0,   -5.12,  -32.768, -600.0])
BOUNDS_MAX = np.array([ 5.12,  10.0,    5.12,   32.768,  600.0])


C_BIN = "./benchmark_optimizer"
TARGET = 1e-8

def run_c(index: int, dim: int):
    cmd = [C_BIN, str(index), str(dim)]
    start = time.perf_counter()
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        sys.exit(f"Could not find the C binary at {C_BIN}")
    except subprocess.CalledProcessError as e:
        sys.exit(f"C run failed:\n{e.output}")

    elapsed = time.perf_counter() - start
    best   = float(re.search(r"Global best fitness:\s+([0-9.eE+-]+)", out).group(1))
    evals  = int(re.search(r"Total evaluations consumed:\s+(\d+)", out).group(1))
    return evals, elapsed, best

def run_pycma(index: int, dim: int, budget: int):
    import cma, numpy as np, time
    func   = FUNC_LIST[index]
    bounds = [BOUNDS_MIN[index], BOUNDS_MAX[index]]
    x0     = np.random.uniform(bounds[0], bounds[1], dim)
    sigma0 = 0.3 * (bounds[1] - bounds[0])

    opts = dict(bounds=bounds,
                maxfevals=budget,
                ftarget=TARGET,
                popsize=None,
                seed=42,
                verb_log=0, verb_disp=0,
                tolflatfitness=0, tolstagnation=0,
                tolfunhist=0,     tolx=0,
                tolupsigma=np.inf)

    start = time.perf_counter()
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    best_f = np.inf

    while not es.stop():
        X = es.ask()
        F = [func(x) for x in X]
        es.tell(X, F)
        best_f = min(best_f, min(F))

        # simple IPOP restart
        if es.stop() and best_f > TARGET:
            pop2 = es.sp.popsize * 2
            es = cma.CMAEvolutionStrategy(es.result.xbest, sigma0, {**opts, "popsize": pop2})

    elapsed = time.perf_counter() - start
    return es.result.evaluations, elapsed, best_f


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Compare C CMA-ES against pycma on classic benchmarks.",
        epilog=textwrap.dedent("""
            Examples
            --------
            # all five functions, 10-D
            python compare_c_vs_py.py

            # just Ackley (index 3), 20-D
            python compare_c_vs_py.py -f 3 -n 20
        """))
    ap.add_argument("-f", "--function", type=int, choices=range(len(FUNC_LIST)),
                    help="function index (0-4). Omit to run all.")
    ap.add_argument("-n", "--dim", type=int, default=10, help="problem dimension N (default 10)")
    ap.add_argument("-b", "--budget", type=int, default=None,
                    help="max evaluations (default: N*1e5 for parity with C)")
    args = ap.parse_args()

    dim    = args.dim
    budget = args.budget or int(dim * 1e5)

    idx_list = [args.function] if args.function is not None else range(len(FUNC_LIST))

    print(f"{'Function':<11} | {'C evals':>8}  | {'Py evals':>8}  | "
          f"{'C t[s]':>6} | {'Py t[s]':>6} | {'C best':>10} | {'Py best':>10}")
    print("-"*80)

    for i in idx_list:
        c_e, c_t, c_f = run_c(i, dim)
        p_e, p_t, p_f = run_pycma(i, dim, budget)

        print(f"{FUNC_NAMES[i]:<11} | {c_e:8d} | {p_e:8d} | "
              f"{c_t:6.2f} | {p_t:6.2f} | {c_f:10.2e} | {p_f:10.2e}")

    print("\nDone.")

if __name__ == "__main__":
    main()

