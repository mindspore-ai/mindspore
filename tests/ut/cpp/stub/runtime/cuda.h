/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TESTS_UT_STUB_RUNTIME_INCLUDE_CUDA_H_
#define TESTS_UT_STUB_RUNTIME_INCLUDE_CUDA_H_

typedef enum cudaError_enum {
  CUDA_SUCCESS = 0,
  CUDA_ERROR_INVALID_IMAGE = 1,
  CUDA_ERROR_DEINITIALIZED = 2,
} CUresult;

typedef enum CUjit_option_enum {
  CU_JIT_MAX_REGISTERS = 0,
  CU_JIT_THREADS_PER_BLOCK,
  CU_JIT_WALL_TIME,
  CU_JIT_INFO_LOG_BUFFER,
  CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
  CU_JIT_ERROR_LOG_BUFFER,
  CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
  CU_JIT_OPTIMIZATION_LEVEL,
  CU_JIT_TARGET_FROM_CUCONTEXT,
  CU_JIT_TARGET,
  CU_JIT_FALLBACK_STRATEGY,
  CU_JIT_GENERATE_DEBUG_INFO,
  CU_JIT_LOG_VERBOSE,
  CU_JIT_GENERATE_LINE_INFO,
  CU_JIT_CACHE_MODE,
  CU_JIT_NEW_SM3X_OPT,
  CU_JIT_FAST_COMPILE,
  CU_JIT_GLOBAL_SYMBOL_NAMES,
  CU_JIT_GLOBAL_SYMBOL_ADDRESSES,
  CU_JIT_GLOBAL_SYMBOL_COUNT,
  CU_JIT_NUM_OPTIONS
} CUjit_option;

typedef enum CUdevice_attribute_enum {
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
} CUdevice_attribute;

struct CUctx_st {
  int arch;
};
struct CUmod_st {
  int arch;
};
struct CUfunc_st {
  int arch;
};
struct CUstream_st {
  int arch;
};

typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;

CUresult cuModuleLoadData(CUmodule *module, const void *image);
CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options,
                            void **optionValues);
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
CUresult cuModuleUnload(CUmodule hmod);
CUresult cuGetErrorName(CUresult error, const char **pStr);
CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, int dev);
CUresult cuStreamSynchronize(CUstream hStream);
#endif  // TESTS_UT_STUB_RUNTIME_INCLUDE_CUDA_H_
