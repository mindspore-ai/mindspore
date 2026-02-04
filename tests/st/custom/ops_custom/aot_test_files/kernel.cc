/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

class add_reduce_kernel_attr : public AotKernelData {
 public:
  int64_t axis;
  bool keep_dim;
};

extern "C" int CustomKernelInit(int *ndims, int64_t **shapes, const char **dtypes, AotExtra *extra) {
  size_t workspace_size = 1;
  for (size_t i = 0; i < ndims[0]; i++) {
    workspace_size *= shapes[0][i];
  }

  std::vector<size_t> workspace = {workspace_size * sizeof(float)};
  extra->SetWorkSpace(workspace);

  add_reduce_kernel_attr *kernel_data_ptr = new add_reduce_kernel_attr;
  kernel_data_ptr->axis = extra->Attr<int64_t>("axis");
  kernel_data_ptr->keep_dim = extra->Attr<bool>("keep_dim");
  extra->SetKernelData(kernel_data_ptr);
  return 0;
}

extern "C" std::vector<int64_t> CustomKernelInferShape(int *ndims, int64_t **shapes, AotExtra *extra) {
  const int64_t kDynRankSize = -2;

  if (shapes[0][0] == kDynRankSize) {
    return std::vector<int64_t>{shapes[0][0]};
  }
  int64_t axis = extra->Attr<int64_t>("axis");
  bool keep_dim = extra->Attr<bool>("keep_dim");
  if (keep_dim) {
    if (axis == 0) {
      return std::vector<int64_t>{1, shapes[0][1]};
    } else {
      return std::vector<int64_t>{shapes[0][0], 1};
    }
  } else {
    return std::vector<int64_t>{shapes[0][1 - axis]};
  }
}

extern "C" int CustomKernel(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                            void *extra_void) {
  float *input_1 = static_cast<float *>(params[0]);
  float *input_2 = static_cast<float *>(params[1]);
  float *output = static_cast<float *>(params[2]);
  float *tmp = static_cast<float *>(params[3]);

  // Add
  int in_size = 1;
  for (int i = 0; i < ndims[0]; i++) {
    in_size *= shapes[0][i];
  }

  for (int i = 0; i < in_size; i++) {
    tmp[i] = input_1[i] + input_2[i];
  }

  // ReduceSum
  AotExtra *extra = static_cast<AotExtra *>(extra_void);
  auto kernel_ptr = static_cast<add_reduce_kernel_attr *>(extra->KernelData());
  bool keep_dim = kernel_ptr->keep_dim;
  int64_t axis = kernel_ptr->axis;
  int64_t input_dim_1 = shapes[0][1];
  int size;
  if (keep_dim) {
    size = shapes[1][0] * shapes[1][1];
  } else {
    size = shapes[1][0];
  }
  int ext = shapes[0][axis];
  for (int i = 0; i < size; i++) {
    output[i] = 0;
    for (int j = 0; j < ext; j++) {
      int idx = input_dim_1 * (i * axis + j * (1 - axis)) + i * (1 - axis) + j * axis;
      output[i] = output[i] + tmp[idx];
    }
  }
  return 0;
}