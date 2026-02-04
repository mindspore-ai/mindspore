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
#include <vector>
#include "custom_aot_extra.h"
class add_kernel : public AotKernelData {
 public:
  float scale;
  std::vector<float> paddings;
  void launch(float *input_1, float *input_2, float *output, float *workspace, int size) {
    for (int i = 0; i < size; i++) {
      workspace[i] = input_1[i] + input_2[i] * scale + paddings[0];
    }
    for (int i = 0; i < size; i++) {
      output[i] = workspace[i] + input_2[i] * scale + paddings[1];
    }
  }
};

extern "C" int CustomAddInit(int *ndims, int64_t **shapes, const char **dtypes, AotExtra *extra) {
  size_t workspace_size = 1;
  for (size_t i = 0; i < ndims[0]; i++) {
    workspace_size *= shapes[0][i];
  }

  std::vector<size_t> workspace = {workspace_size};

  extra->SetWorkSpace(workspace);

  add_kernel *kernel_ptr = new add_kernel;
  kernel_ptr->scale = extra->Attr<float>("scale");
  kernel_ptr->paddings = extra->Attr<std::vector<float>>("paddings");
  extra->SetKernelData(kernel_ptr);
  return 0;
}

extern "C" int CustomAdd(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                         void *extra_void) {
  constexpr int OUTPUT_INDEX = 2;

  float *input1 = static_cast<float *>(params[0]);
  float *input2 = static_cast<float *>(params[1]);
  float *output = static_cast<float *>(params[2]);
  float *tmp = static_cast<float *>(params[3]);
  int size = 1;

  AotExtra *extra = static_cast<AotExtra *>(extra_void);
  // Cumprod of output's shape to compute elements' num
  for (int i = 0; i < ndims[OUTPUT_INDEX]; i++) {
    size *= shapes[OUTPUT_INDEX][i];
  }

  auto kernel_ptr = static_cast<add_kernel *>(extra->KernelData());
  kernel_ptr->launch(input1, input2, output, tmp, size);
  return 0;
}
