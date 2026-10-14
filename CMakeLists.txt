cmake_minimum_required(VERSION 3.22)

project(redline VERSION 0.1.0 LANGUAGES C CXX)

option(REDLINE_BUILD_CUDA "Build the CUDA inference runtime (needs nvcc)" ON)
option(REDLINE_BUILD_TESTS "Build unit tests" ON)
option(REDLINE_WITH_NVTX "Emit NVTX ranges for Nsight Systems" ON)
option(REDLINE_WITH_CUTLASS "Build the CUTLASS mixed-input (FP16 x INT8) GEMM backend" OFF)
set(REDLINE_CUTLASS_DIR "" CACHE PATH "CUTLASS checkout (directory containing include/cutlass)")
option(REDLINE_WITH_TRTLLM "Build the TensorRT-LLM weight-only GEMM backend" OFF)
set(REDLINE_TRTLLM_ROOT "" CACHE PATH "TensorRT-LLM source checkout (contains cpp/include)")
set(REDLINE_TRTLLM_LIB "" CACHE FILEPATH "Path to libtensorrt_llm.so")
option(REDLINE_BUILD_PYTHON "Build the pybind11 extension module" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)

# ----------------------------------------------------------------------------
# Host-only core: model format, config, scheduler, CPU reference model.
# ----------------------------------------------------------------------------
add_library(redline_core STATIC
  src/core/batch_builder.cpp
  src/core/block_allocator.cpp
  src/core/common.cpp
  src/core/json.cpp
  src/core/model_config.cpp
  src/core/safetensors.cpp
  src/core/scheduler.cpp
  src/core/sequence.cpp
  src/core/stats.cpp
  src/core/workload.cpp
  src/cpu/random_model.cpp
  src/cpu/reference_model.cpp
)
target_include_directories(redline_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(redline_core PRIVATE -Wall -Wextra)
target_link_libraries(redline_core PUBLIC Threads::Threads)

add_executable(redline_ref apps/redline_ref.cpp)
target_link_libraries(redline_ref PRIVATE redline_core)

if(REDLINE_BUILD_TESTS)
  enable_testing()
  add_executable(redline_core_tests tests/cpp/test_core.cpp tests/cpp/test_reference.cpp)
  target_include_directories(redline_core_tests PRIVATE tests)
  target_link_libraries(redline_core_tests PRIVATE redline_core)
  add_test(NAME core_tests COMMAND redline_core_tests)
endif()

# ----------------------------------------------------------------------------
# CUDA runtime.
# ----------------------------------------------------------------------------
if(REDLINE_BUILD_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # RTX 4090 is sm_89; add more with -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90".
    set(CMAKE_CUDA_ARCHITECTURES 89)
  endif()
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "nvcc not found. Configure with -DREDLINE_BUILD_CUDA=OFF to build "
                        "only the host components.")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)

  set(REDLINE_CUDA_SOURCES
    src/cuda/device_memory.cpp
    src/cuda/kernels/attention.cu
    src/cuda/kernels/elementwise.cu
    src/cuda/kernels/gemv_weight_only.cu
    src/cuda/kernels/quant.cu
    src/cuda/kernels/rope_kv.cu
    src/cuda/kernels/sampling.cu
    src/cuda/gemm/cublaslt_gemm.cpp
    src/cuda/gemm/linear.cpp
    src/cuda/model/kv_cache.cpp
    src/cuda/model/llama.cpp
    src/cuda/model/weights.cpp
    src/cuda/engine/device_batch.cpp
    src/cuda/engine/engine.cpp
    src/cuda/bench/harness.cpp
  )
  if(REDLINE_WITH_CUTLASS)
    list(APPEND REDLINE_CUDA_SOURCES src/cuda/gemm/cutlass_gemm.cu)
  endif()
  if(REDLINE_WITH_TRTLLM)
    list(APPEND REDLINE_CUDA_SOURCES src/cuda/gemm/trtllm_gemm.cpp)
  endif()

  add_library(redline_cuda STATIC ${REDLINE_CUDA_SOURCES})
  target_include_directories(redline_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(redline_cuda PUBLIC redline_core CUDA::cudart CUDA::cublas CUDA::cublasLt
                        ${CMAKE_DL_LIBS})
  target_compile_options(redline_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo -Xcompiler=-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
  set_target_properties(redline_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

  if(REDLINE_WITH_NVTX)
    # NVTX v3 is header-only and ships with the CUDA toolkit.
    target_compile_definitions(redline_cuda PUBLIC REDLINE_WITH_NVTX=1)
  endif()

  if(REDLINE_WITH_CUTLASS)
    if(NOT REDLINE_CUTLASS_DIR)
      include(FetchContent)
      FetchContent_Declare(cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
        GIT_TAG v3.5.1
        GIT_SHALLOW TRUE)
      FetchContent_GetProperties(cutlass)
      if(NOT cutlass_POPULATED)
        FetchContent_Populate(cutlass)
      endif()
      set(REDLINE_CUTLASS_DIR ${cutlass_SOURCE_DIR})
    endif()
    target_include_directories(redline_cuda PRIVATE ${REDLINE_CUTLASS_DIR}/include
                               ${REDLINE_CUTLASS_DIR}/tools/util/include)
    target_compile_definitions(redline_cuda PUBLIC REDLINE_WITH_CUTLASS=1)
  endif()

  if(REDLINE_WITH_TRTLLM)
    if(NOT REDLINE_TRTLLM_ROOT OR NOT REDLINE_TRTLLM_LIB)
      message(FATAL_ERROR "REDLINE_WITH_TRTLLM needs REDLINE_TRTLLM_ROOT and REDLINE_TRTLLM_LIB")
    endif()
    target_include_directories(redline_cuda PRIVATE
      ${REDLINE_TRTLLM_ROOT}/cpp
      ${REDLINE_TRTLLM_ROOT}/cpp/include
      ${REDLINE_TRTLLM_ROOT}/cpp/tensorrt_llm/cutlass_extensions/include
      ${REDLINE_TRTLLM_ROOT}/3rdparty/cutlass/include)
    target_link_libraries(redline_cuda PUBLIC ${REDLINE_TRTLLM_LIB})
    target_compile_definitions(redline_cuda PUBLIC REDLINE_WITH_TRTLLM=1)
  endif()

  add_executable(redline_generate apps/redline_generate.cpp apps/cli_common.cpp)
  target_link_libraries(redline_generate PRIVATE redline_cuda)
  add_executable(redline_bench apps/redline_bench.cpp apps/cli_common.cpp)
  target_link_libraries(redline_bench PRIVATE redline_cuda)

  if(REDLINE_BUILD_TESTS)
    add_executable(redline_cuda_tests
      tests/cuda/test_kernels.cu
      tests/cuda/test_gemm.cu
      tests/cuda/test_sampling.cu
      tests/cuda/test_engine.cpp)
    target_include_directories(redline_cuda_tests PRIVATE tests)
    target_link_libraries(redline_cuda_tests PRIVATE redline_cuda)
    target_compile_options(redline_cuda_tests PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    add_test(NAME cuda_tests COMMAND redline_cuda_tests)
  endif()

  if(REDLINE_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    execute_process(COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
                    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_redline python/bindings/redline_ext.cpp)
    target_link_libraries(_redline PRIVATE redline_cuda)
    set_target_properties(_redline PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                          ${CMAKE_CURRENT_SOURCE_DIR}/python/redline)
  endif()
endif()
