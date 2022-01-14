#include <cstdlib>
#include <cstdio>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <omp.h>
#include <iostream>
#include <windows.h>
#include <bits/stdc++.h>
#include "reduce_par.h"
#include <numeric>
#include <memory>


constexpr std::size_t CACHE_LINE = 64;

typedef float (*f_t) (float);

#define n 100000000u

typedef struct element_t_
{
    alignas(CACHE_LINE) double value;
} element_t;

typedef struct uint32_t_
{
    alignas(CACHE_LINE) uint64_t value;
} _uint32_t;

float integrate_seq(float a, float b, f_t f)
{
    double res = 0.;
    double dx = (b - a) / n;
    for (size_t i = 0; i < n; ++i)
        res += f((float) (dx * i + a));
    return (float) (res * dx);
}

float randomizer(std::vector<uint32_t>& rand_values, uint32_t min, uint32_t max, uint32_t seed) {
    unsigned T = get_num_threads();
    unsigned N = n;
    uint32_t C = 90626639;
    uint32_t a = 1234;
    uint32_t b = 4321;
    uint32_t a_1 = a-1;
    uint32_t A_c = a_1;
    unsigned power = C-2;
    while (power > 0) {
        if (power & 1) {
            A_c = A_c * a_1 % C;
        }
        power >>= 1;
        a_1 = (a_1 * a_1) % C;
    }

    std::vector<uint32_t> A(n);
    A[0] = a % C;
    unsigned shift = (n / T) + 1;
    for(unsigned i = shift; i < n; i+=shift)
    {
        _uint32_t result{1};
        unsigned power = 1;
        uint32_t base = a;
        while (power > 0) {
            if (power & 1) {
                result.value = result.value * base % C;
            }
            power >>= 1;
            base = (base * base) % C;
        }
        A[shift] = result.value;
    }
    #pragma omp parallel firstprivate(a, b, C, T, shift) shared(A)
    {
        unsigned t = (unsigned) omp_get_thread_num();
        unsigned shift = (n / T) + 1;
        _uint32_t frst{0};
        for (unsigned i = shift * t + 1; i < shift * (t + 1) && i < n; ++i)
            A[i] = A[i-1] * a % C;
    }
    double sum = 0;
    #pragma omp parallel firstprivate(a, b, C, T, shift, A_c) shared(A)
    {
        unsigned t = (unsigned) omp_get_thread_num();
        uint32_t x = 0;
        if (t == 0)
            t+=T;
        for (unsigned i = t; i < n; i += T)
        {
            x = (A[i] * seed % C + (b % C)*((A[i-1]-1) % C)*(A_c) % C) % C;
            x = x % (max - min) + min;
            rand_values[i] = x;
        }

    }
    #pragma omp parallel for reduction(+: sum) schedule(static)
        for (size_t i = 0; i < n; ++i)
            sum += rand_values[i];
    return 0.1+(float)rand_values[2];
}

float integrate_omp(float a, float b, f_t f) {
    double dx = (b - a) / n;
    element_t* results;
    double res = 0.0;
    unsigned T;
#pragma omp parallel shared(results, T)
    {
        unsigned t = (unsigned) omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) omp_get_num_threads();
            results = (element_t*) calloc(CACHE_LINE, T * sizeof(element_t)); //Alignment: aligned_alloc
            if (!results)
                abort();
        } //Барьер
        results[t].value = 0.0;
        for (size_t i = t; i < n; i += T)
            results[t].value += f((float) (dx * i + a));
    }
    for (size_t i = 0; i < T; ++i)
        res += results[i].value;
    free(results);
    return (float) (res * dx);
}

float integrate_omp_fs(float a, float b, f_t f)
{
    double dx = (b - a) / n;
    double* results;
    double res = 0.0;
    unsigned T;
#pragma omp parallel shared(results, T)
    {
        unsigned t = (unsigned) omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) omp_get_num_threads();
            results = (double*) calloc(sizeof(double), T);
            if (!results)
                abort();
        } //Барьер
        for (size_t i = t; i < n; i += T)
            results[t] += f((float) (dx * i + a));
    }

    for (size_t i = 0; i < T; ++i)
        res += results[i];
    free(results);
    return (float) (res * dx);
}

float integrate_omp_reduce(float a, float b, f_t f)
{
    double res = 0.0;
    double dx = (b - a) / n;
    int i;
#pragma omp parallel for reduction(+: res) schedule(static)
    for (i = 0; i < n; ++i)
        res += f((float) (dx * i + a));
    return (float) (res * dx);
}

float integrate_omp_reduce_dynamic(float a, float b, f_t f) {
    double res = 0.0;
    double dx = (b - a) / n;
    int i;
#pragma omp parallel for reduction(+: res) schedule(dynamic, 10000)
    for (i = 0; i < n; ++i)
        res += f((float) (dx * i + a));
    return (float) (res * dx);
}

float integrate_omp_atomic(float a, float b, f_t f)
{
    double res = 0.0;
    double dx = (b - a) / n;
#pragma omp parallel shared(res)
    {
        unsigned t = (unsigned) omp_get_thread_num();
        unsigned T = (unsigned) omp_get_num_threads();
        double val = 0.0;
        for (size_t i = t; i < n; i += T)
        {
            val += f((float) (dx * i + a));
        }
#pragma omp atomic
        res += val;
    }
    return (float) (res * dx);
}

float integrate_omp_cs(float a, float b, f_t f)
{
    double res = 0.0;
    double dx = (b - a) / n;
#pragma omp parallel shared(res)
    {
        unsigned t = (unsigned) omp_get_thread_num();
        unsigned T = (unsigned) omp_get_num_threads();
        double local_sum = 0.0;
        for (size_t i = t; i < n; i += T)
            local_sum += f((float) (dx * i + a));
#pragma omp critical
        {
            res += local_sum;
        }
    }
    return (float) (res * dx);
}

float integrate_omp_mtx(float a, float b, f_t f)
{
    double res = 0.0;
    double dx = (b - a) / n;
    omp_lock_t mtx;
    omp_init_lock(&mtx);
#pragma omp parallel shared(res)
    {
        unsigned t = (unsigned) omp_get_thread_num();
        unsigned T = (unsigned) omp_get_num_threads();
        double val = 0.0;
        for(size_t i = t; i < n; i+=T) {
            val += f(a + i * dx);
        }
        omp_set_lock(&mtx);
        res += val;
        omp_unset_lock(&mtx);
    }
    return res * dx;
}

float integrate_cpp(float a, float b, f_t f)
{
    double dx = (b - a) / n;
    unsigned T = get_num_threads();
    std::vector results(T, element_t{ 0.0 });
    auto thread_proc = [=, &results](unsigned t) {
        results[t].value = 0.0;
        for (size_t i = t; i < n; i += T)
            results[t].value += f((float)(dx * i + a));
    };
    std::vector<std::thread> threads;
    for (unsigned t = 1; t < T; ++t)
        threads.emplace_back(thread_proc, t);
    thread_proc(0);
    for (auto& thread : threads)
        thread.join();
    double res = 0.0;
    for (size_t i = 0; i < T; ++i)
        res += results[i].value;
    return (float)(res * dx);
}

float integrate_cpp_cs(float a, float b, f_t f)
{
    double res = 0.0;
    double dx = (b - a) / n;
    unsigned T = get_num_threads();
    std::mutex mtx;
    auto thread_proc = [=, &res, &mtx](unsigned t) {
        double l_res = 0.0;
        for (size_t i = t; i < n; i += T)
            l_res += f((float)(dx * i + a));
        {
            std::scoped_lock lock(mtx);
            res += l_res;
        }
    };
    std::vector<std::thread> threads;
    for (unsigned t = 1; t < T; ++t)
        threads.emplace_back(thread_proc, t);
    thread_proc(0);
    for (auto& thread : threads)
        thread.join();
    return res * dx;
}

float integrate_cpp_atomic(float a, float b, f_t f) //C++20
{
    std::atomic<double> res = 0.0;
    double dx = (b - a) / n;
    unsigned T = get_num_threads();
    auto thread_proc = [=, &res](unsigned t) {
        double l_res = 0.0;
        for (size_t i = t; i < n; i += T)
            l_res += f((float)(dx * i + a));
        res = res + l_res;
    };
    std::vector<std::thread> threads;
    for (unsigned t = 1; t < T; ++t)
        threads.emplace_back(thread_proc, t);
    thread_proc(0);
    for (auto& thread : threads)
        thread.join();
    return res * dx;
}

float integrate_cpp_reduce_2(float a, float b, f_t f)
{
    double dx = (b - a) / n;
    return reduce_par_2([f, dx](double x, double y) {return x + y; }, f, (double)a, (double)b, (double)dx, 0.0) * dx;
}

float g(float x)
{
    return x * x;
}

uint64_t fibonacci_seq(uint32_t n_) {
    if (n_ < 2) return n_;

    uint64_t first = fibonacci_seq(n_ - 1);
    uint64_t second = fibonacci_seq(n_ - 2);

    return first + second;
}

uint64_t fibonacci_omp(uint32_t n_) {
    if (n_ < 2) return n_;

    uint64_t result1;
    #pragma omp task shared(result1)
    {
        result1 = fibonacci_omp(n_ - 1);
    }

    uint64_t result2;
    #pragma omp task shared(result2)
    {
        result2 = fibonacci_omp(n_ - 2);
    }

    #pragma omp taskwait

    return result1 + result2;
}

uint64_t fibonacci_async(uint32_t n_) {
    if (n_ < 2) return n_;

    std::future first = std::async(std::launch::async, fibonacci_async, n_ - 1);
    std::future second = std::async(std::launch::async, fibonacci_async, n_ - 2);

    return first.get() + second.get();
}

typedef struct experiment_result_t_
{
    float result;
    double time;
} experiment_result_t;

typedef float (*integrate_t)(float a, float b, f_t f);
experiment_result_t run_experiment(integrate_t integrate)
{
    experiment_result_t result;
    double t0 = omp_get_wtime();
    result.result = integrate(-1, 1, g);
    result.time = omp_get_wtime() - t0;
    return result;
}

typedef uint64_t (*fibonachi_t)(uint32_t n_);
experiment_result_t run_experiment(fibonachi_t fibonachi)
{
    experiment_result_t result;
    double t0 = omp_get_wtime();
    result.result = fibonachi(20);
    result.time = omp_get_wtime() - t0;
    return result;
}

typedef float (*generator_t)(std::vector<uint32_t>& rand_values, uint32_t min, uint32_t max, uint32_t seed);
experiment_result_t run_experiment(generator_t generator)
{
    std::vector<uint32_t> output(n);
    experiment_result_t result;
    double t0 = omp_get_wtime();
    result.result = generator(output, 0, 100, 12);
    result.time = omp_get_wtime() - t0;
    return result;
}

void run_experiments(experiment_result_t* results, uint64_t (*I) (uint32_t))
{
    for (unsigned T = 1; T <= std::thread::hardware_concurrency(); ++T)
    {
        set_num_threads(T);
        results[T - 1] = run_experiment(I);
    }
}

void run_experiments(experiment_result_t* results, float (*I) (float, float, f_t))
{
    for (unsigned T = 1; T <= std::thread::hardware_concurrency(); ++T)
    {
        set_num_threads(T);
        results[T - 1] = run_experiment(I);
    }
}

void run_experiments(experiment_result_t* results, float (*I) (std::vector<uint32_t>&, uint32_t, uint32_t, uint32_t))
{
    for (unsigned T = 1; T <= std::thread::hardware_concurrency(); ++T)
    {
        set_num_threads(T);
        results[T - 1] = run_experiment(I);
    }
}

void show_results_for(const char* name, const experiment_result_t* results)
{
    std::cout.precision(15);

    std::cout << name << "\n";
    std::cout << "#\texecution_time\trelative_time\tanswer\n";
    for (unsigned T = 1; T <= omp_get_num_procs(); T++) {
        std::cout << T << "\t" << results[T - 1].time << "\t" << results[0].time / results[T - 1].time << "\t"
                    << results[T - 1].result
                  << "\n";
    }
    std::cout << '\n';
}

int main(int argc, char** argv)
{

    //freopen("output.txt", "w", stdout);
    experiment_result_t* results = (experiment_result_t*)malloc(get_num_threads() * sizeof(experiment_result_t));
    run_experiments(results, integrate_seq);
    show_results_for("'integrate C sequence'", results);
    run_experiments(results, integrate_omp);
    show_results_for("'integrate C OMP'", results);
    run_experiments(results, integrate_omp_fs);
    show_results_for("'integrate C OMP false sharing'", results);
    run_experiments(results, integrate_omp_reduce);
    show_results_for("'integrate C OMP reduce'", results);
    run_experiments(results, integrate_omp_reduce_dynamic);
    show_results_for("'integrate C OMP reduce dynamic'", results);
    run_experiments(results, integrate_omp_atomic);
    show_results_for("'integrate C OMP atomic'", results);
    run_experiments(results, integrate_omp_cs);
    show_results_for("'integrate C OMP critical section'", results);
    run_experiments(results, integrate_omp_mtx);
    show_results_for("'integrate C OMP mutex'", results);
    run_experiments(results, integrate_cpp);
    show_results_for("'integrate C++'", results);
    run_experiments(results, integrate_cpp_cs);
    show_results_for("'integrate C++ Critical section'", results);
    run_experiments(results, integrate_cpp_atomic);
    show_results_for("'integrate C++ Atomic'", results);
    run_experiments(results, integrate_cpp_reduce_2);
    show_results_for("'integrate C++ Reduce'", results);
    run_experiments(results, randomizer);
    show_results_for("'random gen'", results);
    run_experiments(results, fibonacci_seq);
    show_results_for("'fibonachi sequence'", results);
    run_experiments(results, fibonacci_omp);
    show_results_for("'fibonachi omp'", results);
    run_experiments(results, fibonacci_async);
    show_results_for("'fibonachi async'", results);
    free(results);
    experiment_result_t r;
//    r = run_experiment(integrate_seq);
//    printf("'integrate C sequence': {'answer': %g, 'execution_time': %g}\n", (double) r.result, r.time);
//    r = run_experiment(integrate_omp);
//    printf("'integrate C opm': {'answer': %g, 'execution_time': %g}\n", (double) r.result, r.time);
//    r = run_experiment(integrate_omp_fs);
//    printf("'integrate C opm false sharing': {'answer': %g, 'execution_time': %g}\n", (double) r.result, r.time);
//    r = run_experiment(integrate_omp_reduce);
//    printf("'integrate C opm reduce': {'answer': %g, 'execution_time': %g}\n", (double) r.result, r.time);
//    r = run_experiment(integrate_omp_reduce_dynamic);
//    printf("'integrate C opm reduce dynamic': {'answer': %g, 'execution_time': %g}\n", (double) r.result, r.time);
//    r = run_experiment(integrate_omp_atomic);
//    printf("'integrate C opm atomic': {'answer': %g, 'execution_time': %g}\n", (double) r.result, r.time);
//    r = run_experiment(integrate_omp_cs);
//    printf("'integrate C opm critical section': {'answer': %g, 'execution_time': %g}\n", (double) r.result, r.time);
//    r = run_experiment(integrate_omp_mtx);
//    printf("'integrate C opm mutex': {'answer': %g, 'execution_time': %g}\n", (double) r.result, r.time);
//    r = run_experiment(integrate_cpp);
//    printf("'integrate C++': {'answer': %g, 'execution_time': %g}\n", (double) r.result, r.time);
//    r = run_experiment(integrate_cpp_cs);
//    printf("'integrate C++ Critical section': {'answer': %g, 'execution_time': %g}\n", (double) r.result, r.time);
//    r = run_experiment(integrate_cpp_atomic);
//    printf("'integrate C++ Atomic': {'answer': %g, 'execution_time': %g}\n", (double) r.result, r.time);
//    r = run_experiment(integrate_cpp_reduce_2);
//    printf("'integrate C++ Reduce': {'answer': %g, 'execution_time': %g}\n", (double) r.result, r.time);

    return 0;
}