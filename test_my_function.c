// test_my_function.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "optimizer.h" // Wystarczy dołączyć interfejs optymalizatora

// 1. ZDEFINIUJ SWOJĄ FUNKCJĘ CELU
// Przykład: Prosta funkcja kwadratowa przesunięta
// Optimum: f(3, 3, ..., 3) = 0
double my_shifted_sphere(const double* x, int N) {
    double sum_sq = 0.0;
    for (int i = 0; i < N; ++i) {
        double diff = x[i] - 3.0; // Przesunięcie optimum do x_i = 3
        sum_sq += diff * diff;
    }
    return sum_sq;
}

int main() {
    // 2. USTAW PARAMETRY OPTYMALIZACJI DLA TWOJEJ FUNKCJI
    OptimizationParams params;
    params.N = 5;               // Wymiar twojego problemu
    params.max_evaluations = 50000; // Twój budżet ewaluacji
    params.target_fitness = 1e-9; // Twój cel dokładności
    params.search_range_min = -5.0; // Szacowany dolny zakres zmiennych
    params.search_range_max = 10.0; // Szacowany górny zakres zmiennych

    printf("Testowanie optymalizatora na funkcji: My Shifted Sphere (N=%d)\n", params.N);

    // 3. WYWOŁAJ OPTYMALIZATOR, PRZEKAZUJĄC WSKAŹNIK DO SWOJEJ FUNKCJI
    OptimizationResult result = run_optimization_with_restarts(params, my_shifted_sphere);

    // 4. WYŚWIETL WYNIKI (i zwolnij pamięć)
    if (result.best_solution != NULL) {
        printf("\n=== Wynik dla My Shifted Sphere ===\n");
        printf("Najlepszy fitness: %.8e\n", result.best_fitness);
        printf("Osiągnięto cel: %s\n", result.target_reached ? "Tak" : "Nie");
        printf("Najlepsze rozwiązanie (pierwsze %d wym.): [", params.N);
        for (int i = 0; i < params.N; ++i) {
            printf("%.4f%s", result.best_solution[i], (i == params.N - 1) ? "" : ", ");
        }
        printf("]\n");
        printf("Ewaluacje: %lld / %lld\n", result.total_evaluations, params.max_evaluations);
        printf("Restarty: %d\n", result.restarts);

        free(result.best_solution); // Pamiętaj o zwolnieniu!
    } else {
        printf("\nOptymalizacja nie powiodła się.\n");
    }
     printf("Koniec testu.\n");

    return 0;
}