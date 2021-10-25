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

#define THREADS 1024
__global__ void CustomReorganizeKernel(float *input1, int64_t *input2, float *output, size_t size) {
  auto idx = blockIdx.x * THREADS + threadIdx.x;
  if (idx < size) {
    output[idx] = input1[input2[idx]];
  }
}

extern "C" int CustomReorganize(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
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
  if (strcmp(dtypes[1], "int64") != 0) {
    return 2;
  }
  if (strcmp(dtypes[2], "float32") != 0) {
    return 2;
  }
  CustomReorganizeKernel<<<n + 1, THREADS, 0, custream>>>(static_cast<float *>(input1), static_cast<int64_t *>(input2),
                                                          static_cast<float *>(output), size);

  return 0;
}
