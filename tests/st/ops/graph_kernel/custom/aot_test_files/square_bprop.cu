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

__global__ void CustomSquareBpropKernel(float *input1, float *input3, float *output, size_t size) {
  auto idx = blockIdx.x * THREADS + threadIdx.x;
  if (idx < size) {
    output[idx] = input1[idx] * input3[idx] * 2;
  }
}

extern "C" int CustomSquareBprop(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                 void *stream, void *extra) {
  cudaStream_t custream = static_cast<cudaStream_t>(stream);

  constexpr int OUTPUT_INDEX = 3;
  constexpr int TOTAL_PARAM_NUM = 4;

  // Users can add any check on their need. If check fails, user can return any value larger than 0 to safely exit.
  // Return value not equal to 0 will cause MindSpore to stop computing and safely exit.

  // This is to check if the num of parameters the same as what the user wants.
  // There are three inputs and one output, so the nparam should be 4.
  if (nparam != TOTAL_PARAM_NUM) {
    return 1;
  }

  // This is to check if the type of parameters the same as what the user wants.
  for (int i = 0; i < nparam; i++) {
    if (strcmp(dtypes[i], "float32") != 0) {
      return 2;
    }
  }

  // input1's index is 0, input2's index is 1, input3's index is 2 and output's index is 3
  void *input1 = params[0];
  void *input3 = params[2];
  void *output = params[3];

  size_t size = 1;

  // Cumprod of output's shape to compute elements'num
  for (int i = 0; i < ndims[OUTPUT_INDEX]; i++) {
    size *= shapes[OUTPUT_INDEX][i];
  }

  int n = size / THREADS;

  // Do the computation
  CustomSquareBpropKernel<<<n + 1, THREADS, 0, custream>>>(static_cast<float *>(input1), static_cast<float *>(input3),
                                                           static_cast<float *>(output), size);

  // When return 0, MindSpore will continue to run if this kernel could launch successfully.
  return 0;
}
