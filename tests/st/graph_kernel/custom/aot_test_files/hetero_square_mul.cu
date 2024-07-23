/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
constexpr int THREADS = 1024;

#include <cuda_fp16.h>

__global__ void CustomHSquareMulKernel(float *input1, half *input2, half *output, size_t size) {
  auto idx = blockIdx.x * THREADS + threadIdx.x;
  if (idx < size) {
    output[idx] = __float2half(input1[idx] * input1[idx] * __half2float(input2[idx]));
  }
}

extern "C" int CustomHSquareMul(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                void *stream, void *extra) {
  cudaStream_t custream = static_cast<cudaStream_t>(stream);

  constexpr int OUTPUT_INDEX = 2;
  constexpr int TOTAL_PARAM_NUM = 3;

  // Users can add any check on their need. If check fails, user can return any value larger than 0 to safely exit.
  // Return value not equal to 0 will cause MindSpore to stop computing and safely exit.

  // This is to check if the num of parameters the same as what the user wants.
  // There are two inputs and one output, so the nparam should be 3.
  if (nparam != TOTAL_PARAM_NUM) {
    return 1;
  }

  // This is to check if the type of parameters the same as what the user wants.
  if (strcmp(dtypes[0], "float32") != 0) {
    return 2;
  }
  if (strcmp(dtypes[1], "float16") != 0) {
    return 2;
  }
  if (strcmp(dtypes[2], "float16") != 0) {
    return 2;
  }

  // input1's index is 0, input2's index is 1 and output's index is 2
  void *input1 = params[0];
  void *input2 = params[1];

  void *output = params[2];
  size_t size = 1;

  // Cumprod of output's shape to compute elements' num
  for (int i = 0; i < ndims[OUTPUT_INDEX]; i++) {
    size *= shapes[OUTPUT_INDEX][i];
  }
  int n = size / THREADS;

  // Do the computation
  CustomHSquareMulKernel<<<n + 1, THREADS, 0, custream>>>(static_cast<float *>(input1), static_cast<half *>(input2),
                                                          static_cast<half *>(output), size);

  // When return 0, MindSpore will continue to run if this kernel could launch successfully.
  return 0;
}
