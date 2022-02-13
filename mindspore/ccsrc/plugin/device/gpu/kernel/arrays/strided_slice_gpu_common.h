/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_STRIDED_SLICE_GPU_COMMON_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_STRIDED_SLICE_GPU_COMMON_H_

#include <vector>
#include <algorithm>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
constexpr size_t MAX_DIMS = 8;
class StridedSliceGpuCommon {
 public:
  StridedSliceGpuCommon() : null_output_(false) {}
  ~StridedSliceGpuCommon() = default;

  void CollectInfo(const CNodePtr &kernel_node) {
    begin_ = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, kAttrBegin);
    end_ = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, kAttrEnd);
    strides_ = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, kAttrStrides);
    FillEmptyDims(kernel_node, &begin_, &end_, &strides_, &input_shape_);
    ParseStrideSliceMasks(kernel_node, &begin_, &end_, &strides_, input_shape_);
    FillOutputDim();
    null_output_ = IsNullOutput();
  }

 protected:
  void FillOutputDim() {
    for (size_t i = 0; i < MAX_DIMS; i++) {
      if (begin_[i] <= end_[i] && strides_[i] > 0) {
        output_shape_.push_back((end_[i] - 1 - begin_[i]) / strides_[i] + 1);
      } else if (begin_[i] > end_[i] && strides_[i] < 0) {
        output_shape_.push_back((end_[i] - begin_[i] + 1) / strides_[i] + 1);
      } else {
        output_shape_.push_back(0);
      }
    }
  }

  bool IsNullOutput() {
    for (size_t i = 0; i < MAX_DIMS; i++) {
      if (begin_[i] >= end_[i] && strides_[i] > 0) {
        return true;
      }
      if (begin_[i] < end_[i] && strides_[i] < 0) {
        return true;
      }
    }
    return false;
  }

  std::vector<int64_t> begin_;
  std::vector<int64_t> end_;
  std::vector<int64_t> strides_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  bool null_output_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_STRIDED_SLICE_GPU_COMMON_H_
