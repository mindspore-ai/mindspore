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
__global__ void CustomSquareBpropKernel(float *input1, float *input3, float *output, size_t size) {
  auto idx = blockIdx.x * THREADS + threadIdx.x;
  if (idx < size) {
    output[idx] = input1[idx] * input3[idx] * 2;
  }
}

extern "C" int CustomSquareBprop(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                 void *stream, void *extra) {
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  if (nparam != 4) return 1;
  void *input1 = params[0];
  void *input3 = params[2];
  void *output = params[3];

  size_t size = 1;

  for (int i = 0; i < ndims[3]; i++) {
    size *= shapes[3][i];
  }
  int n = size / THREADS;
  for (int i = 0; i < nparam; i++) {
    if (strcmp(dtypes[i], "float32") != 0) {
      return 2;
    }
  }

  CustomSquareBpropKernel<<<n + 1, THREADS, 0, custream>>>(static_cast<float *>(input1), static_cast<float *>(input3),
                                                           static_cast<float *>(output), size);
  return 0;
}
