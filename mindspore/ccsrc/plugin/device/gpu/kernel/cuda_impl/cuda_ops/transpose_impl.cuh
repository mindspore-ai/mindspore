/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#pragma once
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_TRANSPOSE_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_TRANSPOSE_IMPL_CUH_

#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

constexpr int kUnroll = 4;                   // Size of vector.
constexpr int kInfoDims = 78;                // Max TransposeInfoDevice length
constexpr int stride_ndims = 26;             // Max length of input_shape
constexpr int transpose_max_dimension = 26;  // Max dimension of input
constexpr int kDimSize = 26;

struct TransposeInfo {
  std::vector<int64_t> input_shape;
  std::vector<int32_t> perm;
};

struct TransposeInfoDevice {
  int32_t transpose_info_device[kInfoDims];
};

inline void ComputeInputStride(const std::vector<int64_t> &shape, int32_t *strides) {
  const int ndims = shape.size();
  int32_t stride = 1;
  for (int i = ndims - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= static_cast<int32_t>(shape[i]);
  }
}

inline void ComputeOutputStride(const std::vector<int64_t> &shape, const std::vector<int32_t> &perm, int32_t *strides) {
  const int ndims = shape.size();
  int32_t stride = 1;
  for (int i = ndims - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= static_cast<int32_t>(shape[perm[i]]);
  }
}

inline void SimplifyTranspose(const std::vector<int64_t> &input_shape, const std::vector<int32_t> &input_perm,
                              std::vector<int64_t> *new_shape, std::vector<int32_t> *new_perm) {
  auto input_shape_size = input_shape.size();
  std::vector<int64_t> combined_shape(input_shape_size, 0);
  std::vector<int32_t> new_perm_position(input_shape_size, -1);
  int32_t cur_dim = input_perm[0];
  new_perm_position[cur_dim] = 0;
  combined_shape[0] = input_shape[cur_dim];
  int dim_index = 0;
  for (size_t perm_index = 1; perm_index < input_shape_size; ++perm_index) {
    if (input_perm[perm_index] == cur_dim + 1) {
      cur_dim = input_perm[perm_index];
      combined_shape[dim_index] *= input_shape[cur_dim];
    } else {
      cur_dim = input_perm[perm_index];
      dim_index++;
      new_perm_position[cur_dim] = dim_index;
      combined_shape[dim_index] = input_shape[cur_dim];
    }
  }
  new_shape->resize(dim_index + 1);
  std::vector<int32_t> new_perm_temp(dim_index + 1, 0);
  new_perm->resize(dim_index + 1);
  dim_index = 0;

  for (size_t i = 0; i < new_perm_position.size(); ++i) {
    if (new_perm_position[i] >= 0) {
      int new_perm_index = new_perm_position[i];
      (*new_shape)[dim_index] = combined_shape[new_perm_index];
      new_perm_temp[dim_index] = new_perm_index;
      dim_index++;
    }
  }
  for (int i = 0; i < dim_index + 1; ++i) {
    auto ret = std::find_if(new_perm_temp.begin(), new_perm_temp.end(), [&](int x) { return x == i; });
    if (ret != new_perm_temp.end()) {
      (*new_perm)[i] = ret - new_perm_temp.begin();
    }
  }
}

template <typename T, bool need_simplify = true>
CUDA_LIB_EXPORT cudaError_t CalTranspose(const size_t size, const T *input, const TransposeInfo &info, T *output,
                                         cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_TRANSPOSE_IMPL_CUH_
