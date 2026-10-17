#include "cl_context.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

static std::string device_string(cl_device_id d, cl_device_info what) {
    size_t n = 0;
    clGetDeviceInfo(d, what, 0, nullptr, &n);
    std::string s(n, '\0');
    clGetDeviceInfo(d, what, n, s.data(), nullptr);
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

ClContext::ClContext(bool verbose) {
    cl_uint np = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &np));
    std::vector<cl_platform_id> platforms(np);
    CL_CHECK(clGetPlatformIDs(np, platforms.data(), nullptr));

    // First GPU on any platform (the NVIDIA platform on the target laptop).
    for (cl_platform_id p : platforms) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &device_, &nd) == CL_SUCCESS && nd > 0) break;
        device_ = nullptr;
    }
    if (!device_) throw std::runtime_error("no OpenCL GPU device found");

    name_ = device_string(device_, CL_DEVICE_NAME);
    if (verbose) {
        size_t wg = 0;
        cl_ulong lmem = 0, gmem = 0;
        cl_uint cus = 0;
        clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wg), &wg, nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, nullptr);
        std::cout << "[OpenCL] device      : " << name_ << "\n"
                  << "[OpenCL] version     : " << device_string(device_, CL_DEVICE_VERSION) << "\n"
                  << "[OpenCL] driver      : " << device_string(device_, CL_DRIVER_VERSION) << "\n"
                  << "[OpenCL] compute units: " << cus << "\n"
                  << "[OpenCL] max WG size : " << wg << "\n"
                  << "[OpenCL] local mem   : " << lmem / 1024 << " KB\n"
                  << "[OpenCL] global mem  : " << gmem / (1024 * 1024) << " MB\n";
    }

    cl_int err = CL_SUCCESS;
    ctx_.reset(clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err));
    CL_CHECK(err);
    // Profiling must be enabled at queue creation to read event timestamps.
    queue_.reset(clCreateCommandQueue(ctx_.get(), device_, CL_QUEUE_PROFILING_ENABLE, &err));
    CL_CHECK(err);
}

ClProgramPtr ClContext::build_program(const std::string& path, const std::string& options) const {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open kernel file: " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string src = ss.str();
    const char* p = src.c_str();
    size_t len = src.size();

    cl_int err = CL_SUCCESS;
    ClProgramPtr prog(clCreateProgramWithSource(ctx_.get(), 1, &p, &len, &err));
    CL_CHECK(err);
    err = clBuildProgram(prog.get(), 1, &device_, options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t n = 0;
        clGetProgramBuildInfo(prog.get(), device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::string log(n, '\0');
        clGetProgramBuildInfo(prog.get(), device_, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
        std::cerr << "Build log for " << path << ":\n" << log << "\n";
        CL_CHECK(err);
    }
    return prog;
}
