#pragma once
#include <algorithm>
#include <chrono>
#include <vector>

#include <CL/cl.h>

using Clock = std::chrono::steady_clock;

// Milliseconds elapsed since t0 (wall clock).
inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// GPU-side duration of a finished command, from OpenCL profiling counters.
// Requires the queue to be created with CL_QUEUE_PROFILING_ENABLE.
// Measures only the time the device spent on the command, not host overhead.
inline double event_ms(cl_event e) {
    cl_ulong start = 0, end = 0;
    clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
    clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
    return double(end - start) * 1e-6;  // ns -> ms
}

// Median is robust to the occasional slow frame (OS scheduling, first-touch
// allocations), so it is a fairer summary than the mean.
inline double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
