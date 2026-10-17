cmake_minimum_required(VERSION 3.20)
project(gpu_video_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# All executables go to build/bin (for every config) so the FFmpeg DLLs and
# kernels/ only need to be copied to one place.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
foreach(cfg Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER ${cfg} CFG)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/bin)
endforeach()

# ---------------------------------------------------------------- OpenCL ----
# Windows: Khronos OpenCL-SDK zip extracted to third_party/opencl (headers +
# OpenCL.lib). The actual runtime (OpenCL.dll) comes from the NVIDIA driver.
set(OPENCL_ROOT "${CMAKE_SOURCE_DIR}/third_party/opencl" CACHE PATH "OpenCL SDK root")
if(EXISTS "${OPENCL_ROOT}/include/CL/cl.h")
  set(OpenCL_INCLUDE_DIR "${OPENCL_ROOT}/include")
  find_library(OpenCL_LIBRARY OpenCL PATHS "${OPENCL_ROOT}/lib" NO_DEFAULT_PATH)
endif()
find_package(OpenCL REQUIRED)

# ---------------------------------------------------------------- FFmpeg ----
# Windows: BtbN "win64-gpl-shared" build extracted to third_party/ffmpeg.
# Linux: system packages found through pkg-config.
set(FFMPEG_ROOT "${CMAKE_SOURCE_DIR}/third_party/ffmpeg" CACHE PATH "FFmpeg dev root")
set(FF_LIBS avformat avcodec avutil swscale swresample)
if(EXISTS "${FFMPEG_ROOT}/include/libavcodec/avcodec.h")
  add_library(ffmpeg INTERFACE)
  target_include_directories(ffmpeg INTERFACE "${FFMPEG_ROOT}/include")
  foreach(l ${FF_LIBS})
    find_library(FF_${l} ${l} PATHS "${FFMPEG_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
    target_link_libraries(ffmpeg INTERFACE ${FF_${l}})
  endforeach()
  file(GLOB FFMPEG_DLLS "${FFMPEG_ROOT}/bin/*.dll")
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FF REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil libswscale libswresample)
  add_library(ffmpeg INTERFACE)
  target_link_libraries(ffmpeg INTERFACE PkgConfig::FF)
endif()

find_package(OpenMP)  # used only to give the CPU baseline a fair multi-core option

# ------------------------------------------------------------- core lib ----
add_library(pipeline_core STATIC
  src/video_decoder.cpp
  src/video_encoder.cpp
  src/cl_context.cpp
  src/cpu_ops.cpp
  src/gpu_ops.cpp
  src/metrics.cpp)
target_include_directories(pipeline_core PUBLIC include)
target_compile_definitions(pipeline_core PUBLIC
  CL_TARGET_OPENCL_VERSION=120 CL_USE_DEPRECATED_OPENCL_1_2_APIS)
target_link_libraries(pipeline_core PUBLIC ffmpeg OpenCL::OpenCL)
if(OpenMP_CXX_FOUND)
  target_link_libraries(pipeline_core PUBLIC OpenMP::OpenMP_CXX)
endif()
if(MSVC)
  target_compile_options(pipeline_core PUBLIC /W3 /fp:precise /utf-8)
else()
  # Disable FMA contraction so CPU float math matches the GPU (which uses
  # "#pragma OPENCL FP_CONTRACT OFF") bit-for-bit.
  target_compile_options(pipeline_core PUBLIC -Wall -ffp-contract=off)
endif()

add_executable(video_pipeline src/main_video.cpp)
target_link_libraries(video_pipeline PRIVATE pipeline_core)

add_executable(yuv_psnr src/main_psnr.cpp)
target_link_libraries(yuv_psnr PRIVATE pipeline_core)

add_executable(kernel_bench src/main_kernel_bench.cpp)
target_link_libraries(kernel_bench PRIVATE pipeline_core)

# ----------------------------------------------------------------- tests ----
enable_testing()
foreach(t test_color_roundtrip test_cpu_vs_gpu test_psnr)
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} PRIVATE pipeline_core)
endforeach()
add_test(NAME color_roundtrip COMMAND test_color_roundtrip)
add_test(NAME cpu_vs_gpu COMMAND test_cpu_vs_gpu ${CMAKE_SOURCE_DIR}/kernels)
add_test(NAME psnr COMMAND test_psnr)

# ------------------------------------------- runtime files next to exes ----
add_custom_target(copy_runtime ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_SOURCE_DIR}/kernels ${CMAKE_BINARY_DIR}/bin/kernels)
if(FFMPEG_DLLS)
  add_custom_command(TARGET copy_runtime POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FFMPEG_DLLS} ${CMAKE_BINARY_DIR}/bin)
endif()
add_dependencies(video_pipeline copy_runtime)
