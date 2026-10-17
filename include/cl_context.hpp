#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <CL/cl.h>

// Every OpenCL call is checked; on failure we throw with the error code and
// the exact source location so problems are easy to track down.
#define CL_CHECK(expr)                                                              \
    do {                                                                            \
        cl_int _err = (expr);                                                       \
        if (_err != CL_SUCCESS)                                                     \
            throw std::runtime_error("OpenCL error " + std::to_string(_err) + " at " + \
                                     __FILE__ + ":" + std::to_string(__LINE__) +    \
                                     " (" #expr ")");                               \
    } while (0)

// unique_ptr wrappers so OpenCL objects are released automatically (RAII).
#define CL_RAII(Name, cl_type, release_fn)                                \
    struct Name##Deleter {                                                \
        void operator()(cl_type p) const { if (p) release_fn(p); }       \
    };                                                                    \
    using Name = std::unique_ptr<std::remove_pointer_t<cl_type>, Name##Deleter>;

CL_RAII(ClContextPtr, cl_context, clReleaseContext)
CL_RAII(ClQueuePtr, cl_command_queue, clReleaseCommandQueue)
CL_RAII(ClProgramPtr, cl_program, clReleaseProgram)
CL_RAII(ClKernelPtr, cl_kernel, clReleaseKernel)
CL_RAII(ClMemPtr, cl_mem, clReleaseMemObject)
CL_RAII(ClEventPtr, cl_event, clReleaseEvent)

// Owns the platform/device/context/queue. Picks the first GPU it finds.
class ClContext {
public:
    explicit ClContext(bool verbose = true);

    cl_context context() const { return ctx_.get(); }
    cl_command_queue queue() const { return queue_.get(); }
    cl_device_id device() const { return device_; }
    const std::string& device_name() const { return name_; }

    // Compile a .cl file; prints the full build log if compilation fails.
    ClProgramPtr build_program(const std::string& path, const std::string& options = "") const;

private:
    cl_device_id device_ = nullptr;  // devices are not reference counted in 1.2
    ClContextPtr ctx_;
    ClQueuePtr queue_;
    std::string name_;
};
