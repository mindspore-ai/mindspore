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

__global__ void CustomAddMulDivBpropKernel(float *input1, float *input2, float *input3, float *input4, float *input5,
                                           float *output1, float *output2, size_t size) {
  auto idx = blockIdx.x * THREADS + threadIdx.x;
  if (idx < size) {
    output1[idx] = input3[idx] + input4[idx] * input2[idx] + input5[idx] / input2[idx];
    output2[idx] = input3[idx] + input4[idx] * input1[idx] - input5[idx] * input1[idx] / input2[idx] / input2[idx];
  }
}

extern "C" int CustomAddMulDivBprop(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                    void *stream, void *extra) {
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  if (nparam != 7) return 1;
  void *input1 = params[0];
  void *input2 = params[1];
  void *input3 = params[2];
  void *input4 = params[3];
  void *input5 = params[4];
  void *output1 = params[5];
  void *output2 = params[6];

  size_t size = 1;

  for (int i = 0; i < ndims[6]; i++) {
    size *= shapes[6][i];
  }
  int n = size / THREADS;
  for (int i = 0; i < nparam; i++) {
    if (strcmp(dtypes[i], "float32") != 0) {
      return 2;
    }
  }

  CustomAddMulDivBpropKernel<<<n + 1, THREADS, 0, custream>>>(
    static_cast<float *>(input1), static_cast<float *>(input2), static_cast<float *>(input3),
    static_cast<float *>(input4), static_cast<float *>(input5), static_cast<float *>(output1),
    static_cast<float *>(output2), size);

  return 0;
}
