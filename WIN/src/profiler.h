// ============================================================================
// profiler.h — Lightweight pipeline profiler for FlareSim
//
// Usage:
//   FlareProfiler prof;
//   prof.begin("stage_name");
//   ... do work ...
//   prof.end();
//
//   prof.print_summary();   // prints a formatted table to stdout
//
// For CUDA stages, use begin_gpu() / end_gpu() which inserts
// cudaDeviceSynchronize() before and after to isolate kernel time.
// ============================================================================
#pragma once

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

struct FlareProfiler
{
    struct Entry
    {
        std::string name;
        double      ms;
    };

    std::vector<Entry>                              entries;
    std::chrono::time_point<std::chrono::steady_clock> t0;
    std::chrono::time_point<std::chrono::steady_clock> frame_t0;
    std::string                                     current;

    FlareProfiler()
    {
        frame_t0 = std::chrono::steady_clock::now();
    }

    void begin(const char* name)
    {
        current = name;
        t0 = std::chrono::steady_clock::now();
    }

    void end()
    {
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        entries.push_back({current, ms});
    }

    // Record a manually-timed entry (e.g. from CUDA events).
    void record(const char* name, double ms)
    {
        entries.push_back({name, ms});
    }

    void print_summary() const
    {
        auto frame_t1 = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(frame_t1 - frame_t0).count();

        printf("\n");
        printf("┌─────────────────────────────────────────────────────────┐\n");
        printf("│              FlareSim Profiler — Frame Breakdown        │\n");
        printf("├────────────────────────────────────┬──────────┬─────────┤\n");
        printf("│ Stage                              │  Time ms │   %%     │\n");
        printf("├────────────────────────────────────┼──────────┼─────────┤\n");

        for (const auto& e : entries)
        {
            double pct = (total_ms > 0.0) ? (e.ms / total_ms * 100.0) : 0.0;
            printf("│ %-34s │ %8.2f │ %5.1f%%  │\n", e.name.c_str(), e.ms, pct);
        }

        printf("├────────────────────────────────────┼──────────┼─────────┤\n");
        printf("│ TOTAL                              │ %8.2f │ 100.0%%  │\n", total_ms);
        printf("└────────────────────────────────────┴──────────┴─────────┘\n");
        fflush(stdout);
    }
};
