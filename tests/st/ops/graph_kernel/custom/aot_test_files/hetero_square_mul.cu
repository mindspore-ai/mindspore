/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <cuda_fp16.h>
#define THREADS 1024

__global__ void CustomHSquareMulKernel(float *input1, half *input2, half *output, size_t size) {
  auto idx = blockIdx.x * THREADS + threadIdx.x;
  if (idx < size) {
    output[idx] = __float2half(input1[idx] * input1[idx] * __half2float(input2[idx]));
  }
}

extern "C" int CustomHSquareMul(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                void *stream, void *extra) {
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  if (nparam != 3) return 1;
  void *input1 = params[0];
  void *input2 = params[1];

  void *output = params[2];
  size_t size = 1;

  for (int i = 0; i < ndims[2]; i++) {
    size *= shapes[2][i];
  }
  int n = size / THREADS;

  if (strcmp(dtypes[0], "float32") != 0) {
    return 2;
  }
  if (strcmp(dtypes[1], "float16") != 0) {
    return 2;
  }
  if (strcmp(dtypes[2], "float16") != 0) {
    return 2;
  }

  CustomHSquareMulKernel<<<n + 1, THREADS, 0, custream>>>(static_cast<float *>(input1), static_cast<half *>(input2),
                                                          static_cast<half *>(output), size);
  return 0;
}
