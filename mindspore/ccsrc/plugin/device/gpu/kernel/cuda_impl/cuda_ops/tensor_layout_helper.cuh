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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_TENSOR_LAYOUT_HELPER_CUH_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_TENSOR_LAYOUT_HELPER_CUH_

#include <cuda_runtime.h>
#include <climits>
#include <string>
#include <utility>
#include "ir/dtype/type_id.h"

#define MAX_TENSORINFO_DIMS 8

// CUDA kernel argument that defines tensor layout
struct TensorLayoutHelper {
  TensorLayoutHelper(const int shape[MAX_TENSORINFO_DIMS], int dim_size) {
    dim_size_ = dim_size;
    if (dim_size_ > MAX_TENSORINFO_DIMS) {
      printf("[ERROR] dim_size_ > MAX_TENSORINFO_DIMS(8).\n");
      exit(1);
    }

    for (int i = 0; i < dim_size_; ++i) {
      sizes_[i] = shape[i];
    }
    int64_t stride = 1;
    for (int i = dim_size_ - 1; i >= 0; i--) {
      strides_[i] = stride;
      stride *= shape[i];
    }
    shape_size_ = stride;
  }

  static std::pair<int64_t, int64_t> CollapseDimsInner(int *sizes, int64_t *strides, int dim_size, int exclude_dim) {
    int64_t stop_dim = (exclude_dim == -1) ? dim_size : exclude_dim;
    int64_t new_index = -1;
    int64_t old_index = 0;
    int64_t remapped_excluded_dim = -1;

    while (old_index < dim_size) {
      for (; old_index < stop_dim; ++old_index) {
        if (sizes[old_index] == 1) {
          continue;
        }

        ++new_index;
        sizes[new_index] = sizes[old_index];
        strides[new_index] = strides[old_index];
        ++old_index;
        break;
      }

      for (; old_index < stop_dim; ++old_index) {
        if (sizes[old_index] == 1) {
          continue;
        }

        if (strides[new_index] == sizes[old_index] * strides[old_index]) {
          sizes[new_index] *= sizes[old_index];
          strides[new_index] = strides[old_index];
        } else {
          ++new_index;
          sizes[new_index] = sizes[old_index];
          strides[new_index] = strides[old_index];
        }
      }

      if (old_index != dim_size) {
        ++new_index;
        sizes[new_index] = sizes[old_index];
        strides[new_index] = strides[old_index];
        remapped_excluded_dim = new_index;

        ++old_index;
        stop_dim = dim_size;
      }
    }

    if (new_index == -1 || (new_index == 0 && sizes[0] == 1)) {
      dim_size = 1;
      sizes[0] = 1;
      strides[0] = 1;
      return std::pair<int64_t, int64_t>(0, 1);
    }

    dim_size = new_index + 1;
    return std::pair<int64_t, int64_t>(remapped_excluded_dim, dim_size);
  }

  inline int CollapseDims(int exclude_dim = -1) {
    if (exclude_dim < 0) {
      exclude_dim += dim_size_;
    }
    if (exclude_dim >= dim_size_ || exclude_dim < 0) {
      printf("dim out of range of dim_size_.\n");
      exit(1);
    }
    auto result = CollapseDimsInner(sizes_, strides_, dim_size_, exclude_dim);
    dim_size_ = result.second;
    return result.first;
  }

  // Contiguous tensors of more than one dimension are collapsed down to one tensor
  inline bool IsContiguous() const { return (dim_size_ == 1 && strides_[0] == 1); }

  int sizes_[MAX_TENSORINFO_DIMS];
  int64_t strides_[MAX_TENSORINFO_DIMS];
  int dim_size_{0};
  int64_t shape_size_{0};
};

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_TENSOR_LAYOUT_HELPER_CUH_
