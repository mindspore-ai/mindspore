/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/assert_impl.cuh"
#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "mindspore/core/mindapi/base/type_id.h"
#include "include/cuda_fp16.h"

__device__ __forceinline__ void PrintData(float *input, int summarize) {
  for (int i = 0; i < summarize - 1; i++) {
    printf("%f ", input[i]);
  }
  printf("%f]\n", input[summarize - 1]);
}

__device__ __forceinline__ void PrintData(half *input, int summarize) {
  for (int i = 0; i < summarize - 1; i++) {
    printf("%f ", __half2float(input[i]));
  }
  printf("%f]\n", __half2float(input[summarize - 1]));
}

__device__ __forceinline__ void PrintData(double *input, int summarize) {
  for (int i = 0; i < summarize - 1; i++) {
    printf("%lf ", input[i]);
  }
  printf("%lf]\n", input[summarize - 1]);
}

__device__ __forceinline__ void PrintData(int64_t *input, int summarize) {
  for (int i = 0; i < summarize - 1; i++) {
    printf("%lld ", input[i]);
  }
  printf("%lld]\n", input[summarize - 1]);
}

__device__ __forceinline__ void PrintData(int32_t *input, int summarize) {
  for (int i = 0; i < summarize - 1; i++) {
    printf("%d ", input[i]);
  }
  printf("%d]\n", input[summarize - 1]);
}

__device__ __forceinline__ void PrintData(int16_t *input, int summarize) {
  for (int i = 0; i < summarize - 1; i++) {
    printf("%d ", static_cast<int32_t>(input[i]));
  }
  printf("%d]\n", static_cast<int32_t>(input[summarize - 1]));
}

__device__ __forceinline__ void PrintData(int8_t *input, int summarize) {
  for (int i = 0; i < summarize - 1; i++) {
    printf("%d ", static_cast<int32_t>(input[i]));
  }
  printf("%d]\n", static_cast<int32_t>(input[summarize - 1]));
}

__device__ __forceinline__ void PrintData(uint64_t *input, int summarize) {
  for (int i = 0; i < summarize - 1; i++) {
    printf("%lu ", input[i]);
  }
  printf("%lu]\n", input[summarize - 1]);
}
__device__ __forceinline__ void PrintData(uint32_t *input, int summarize) {
  for (int i = 0; i < summarize - 1; i++) {
    printf("%u ", input[i]);
  }
  printf("%u]\n", input[summarize - 1]);
}

__device__ __forceinline__ void PrintData(uint16_t *input, int summarize) {
  for (int i = 0; i < summarize - 1; i++) {
    printf("%u ", static_cast<uint32_t>(input[i]));
  }
  printf("%u]\n", static_cast<uint32_t>(input[summarize - 1]));
}

__device__ __forceinline__ void PrintData(uint8_t *input, int summarize) {
  for (int i = 0; i < summarize - 1; i++) {
    printf("%u ", static_cast<uint32_t>(input[i]));
  }
  printf("%u]\n", static_cast<uint32_t>(input[summarize - 1]));
}

__device__ __forceinline__ void PrintData(bool *input, int summarize) {
  for (int i = 0; i < summarize - 1; i++) {
    printf("%d ", input[i]);
  }
  printf("%d]\n", input[summarize - 1]);
}

__global__ void CalculateAssertKernel(const bool *cond, void **inputs, int *summarizes, int *types,
                                      const size_t input_num) {
  if (cond[0]) {
    return;
  }
  printf("For 'Assert' condition is false.\n");
  for (size_t i = 0; i < input_num; i++) {
    printf("input data: [");
    switch (types[i]) {
      case mindspore::kNumberTypeFloat32:
        PrintData(static_cast<float *>(inputs[i]), summarizes[i]);
        break;
      case mindspore::kNumberTypeFloat16:
        PrintData(static_cast<half *>(inputs[i]), summarizes[i]);
        break;
      case mindspore::kNumberTypeFloat64:
        PrintData(static_cast<double *>(inputs[i]), summarizes[i]);
        break;
      case mindspore::kNumberTypeInt32:
        PrintData(static_cast<int32_t *>(inputs[i]), summarizes[i]);
        break;
      case mindspore::kNumberTypeInt64:
        PrintData(static_cast<int64_t *>(inputs[i]), summarizes[i]);
        break;
      case mindspore::kNumberTypeInt16:
        PrintData(static_cast<int16_t *>(inputs[i]), summarizes[i]);
        break;
      case mindspore::kNumberTypeInt8:
        PrintData(static_cast<int8_t *>(inputs[i]), summarizes[i]);
        break;
      case mindspore::kNumberTypeUInt32:
        PrintData(static_cast<uint32_t *>(inputs[i]), summarizes[i]);
        break;
      case mindspore::kNumberTypeUInt64:
        PrintData(static_cast<uint64_t *>(inputs[i]), summarizes[i]);
        break;
      case mindspore::kNumberTypeUInt16:
        PrintData(static_cast<uint16_t *>(inputs[i]), summarizes[i]);
        break;
      case mindspore::kNumberTypeUInt8:
        PrintData(static_cast<uint8_t *>(inputs[i]), summarizes[i]);
        break;
      case mindspore::kNumberTypeBool:
        PrintData(static_cast<bool *>(inputs[i]), summarizes[i]);
        break;
      default:
        printf("unsupported data type, typeid %d]\n", types[i]);
        break;
    }
  }
  return;
}

void AssertKernel(const bool *cond, void **inputs, int *summarizes, int *types, const size_t input_num,
                  const uint32_t device_id, cudaStream_t cuda_stream) {
  CalculateAssertKernel<<<1, 1, 0, cuda_stream>>>(cond, inputs, summarizes, types, input_num);
  return;
}
