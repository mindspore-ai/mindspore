/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef TESTS_UT_STUB_RUNTIME_INCLUDE_CUDA_RUNTIME_API_H_
#define TESTS_UT_STUB_RUNTIME_INCLUDE_CUDA_RUNTIME_API_H_

#include <cstddef>
typedef enum { cudaSuccess = 0, cudaErrorNotReady = 1 } cudaError_t;

unsigned int cudaEventDefault = 0;

enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3
};

struct CUstream_st {
  int arch;
};

typedef struct CUStream_st *cudaStream_t;

cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemGetInfo(size_t *free, size_t *total);
cudaError_t cudaStreamCreate(cudaStream_t *pStream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaGetDeviceCount(int *count);
cudaError_t cudaSetDevice(int device);

#endif  // TESTS_UT_STUB_RUNTIME_INCLUDE_CUDA_RUNTIME_API_H_
