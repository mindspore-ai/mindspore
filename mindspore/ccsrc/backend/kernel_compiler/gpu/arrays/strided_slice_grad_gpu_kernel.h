/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_STRIDED_SLICE_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_STRIDED_SLICE_GRAD_GPU_KERNEL_H_

#include <vector>
#include <bitset>
#include <algorithm>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/slice_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t MAX_DIMS = 7;
template <typename T>
class StridedSliceGradGpuKernel : public GpuKernel {
 public:
  StridedSliceGradGpuKernel() : null_output_(false) {}
  ~StridedSliceGradGpuKernel() override = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *dy = GetDeviceAddress<T>(inputs, 0);
    T *dx = GetDeviceAddress<T>(outputs, 0);

    FillDeviceArray(outputs[0]->size / sizeof(T), dx, 0.f, reinterpret_cast<cudaStream_t>(stream_ptr));
    if (null_output_) {
      return true;
    }

    StridedSliceGrad(output_shape_, begin_, strides_, input_shape_, dy, dx, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    std::vector<int64_t> shapex = GetAttr<std::vector<int64_t>>(kernel_node, "shapex");
    for (auto x : shapex) {
      input_shape_.push_back(static_cast<size_t>(x));
    }
    if (input_shape_.size() > MAX_DIMS) {
      MS_LOG(ERROR) << "StridedSliceGrad support support dims less than " << input_shape_.size();
      return false;
    }

    FillEmptyDims(kernel_node);
    ParseMasks(kernel_node);
    FillOutputDim();
    null_output_ = IsNullOutput();
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    size_t size = sizeof(T);
    for (size_t i = 0; i < MAX_DIMS; i++) {
      size *= output_shape_[i];
    }
    input_size_list_.push_back(size);

    size_t size1 = sizeof(T);
    for (size_t i = 0; i < MAX_DIMS; i++) {
      size1 *= input_shape_[i];
    }
    output_size_list_.push_back(size1);
  }

 private:
  void FillEmptyDims(const CNodePtr &kernel_node) {
    begin_ = GetAttr<std::vector<int64_t>>(kernel_node, "begin");
    end_ = GetAttr<std::vector<int64_t>>(kernel_node, "end");
    strides_ = GetAttr<std::vector<int64_t>>(kernel_node, "strides");

    for (size_t i = 0; i < MAX_DIMS; i++) {
      if (i < begin_.size()) {
        int64_t dim = input_shape_[i];
        begin_[i] = std::min(begin_[i] < 0 ? std::max(begin_[i] + dim, static_cast<int64_t>(0)) : begin_[i], dim - 1);
      } else {
        begin_.push_back(0);
      }

      if (i < end_.size()) {
        int64_t dim = input_shape_[i];
        end_[i] = std::max(end_[i] < 0 ? end_[i] + dim : std::min(end_[i], dim), static_cast<int64_t>(-1));
      } else {
        end_.push_back(i < input_shape_.size() ? input_shape_[i] : 1);
      }

      if (i >= strides_.size()) {
        strides_.push_back(1);
      }

      if (i >= input_shape_.size()) {
        input_shape_.push_back(1);
      }
    }
  }

  void ParseMasks(const CNodePtr &kernel_node) {
    auto begin_mask_int = static_cast<int64_t>(GetAttr<int64_t>(kernel_node, "begin_mask"));
    auto begin_mask = Dec2Bin(begin_mask_int);
    for (size_t i = 0; i < begin_mask.size(); i++) {
      if (begin_mask[i]) {
        begin_[i] = 0;
      }
    }

    auto end_mask_int = static_cast<int64_t>(GetAttr<int64_t>(kernel_node, "end_mask"));
    auto end_mask = Dec2Bin(end_mask_int);
    for (size_t j = 0; j < end_mask.size(); j++) {
      if (end_mask[j]) {
        end_[j] = input_shape_[j];
      }
    }

    auto ellipsis_mask_int = static_cast<int64_t>(GetAttr<int64_t>(kernel_node, "ellipsis_mask"));
    auto ellipsis_mask = Dec2Bin(ellipsis_mask_int);
    for (size_t k = 0; k < ellipsis_mask.size(); k++) {
      if (ellipsis_mask[k]) {
        begin_[k] = 0;
        end_[k] = input_shape_[k];
        strides_[k] = 1;
      }
    }

    auto new_axis_mask_int = static_cast<int64_t>(GetAttr<int64_t>(kernel_node, "new_axis_mask"));
    auto new_axis_mask = Dec2Bin(new_axis_mask_int);
    for (size_t l = 0; l < new_axis_mask.size(); l++) {
      if (new_axis_mask[l]) {
        begin_[l] = 0;
        end_[l] = input_shape_[l];
        strides_[l] = 1;
      }
    }

    auto shrink_axis_mask_int = static_cast<int64_t>(GetAttr<int64_t>(kernel_node, "shrink_axis_mask"));
    auto shrink_axis_mask = Dec2Bin(shrink_axis_mask_int);
    for (size_t m = 0; m < shrink_axis_mask.size(); m++) {
      if (shrink_axis_mask[m]) {
        end_[m] = end_[m] > begin_[m] ? begin_[m] + 1 : begin_[m] - 1;
        strides_[m] = end_[m] > begin_[m] ? 1 : -1;
      }
    }
  }

  std::vector<bool> Dec2Bin(const int64_t &mask) {
    auto mask_str = std::bitset<MAX_DIMS>(mask).to_string();
    int64_t dim_idx = 0;
    std::vector<bool> result = {false, false, false, false};
    for (int64_t i = mask_str.size() - 1; i >= 0; i--) {
      if (mask_str[i] == '1') {
        result[dim_idx] = true;
      }
      dim_idx++;
    }
    return result;
  }

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

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_STRIDED_SLICE_GRAD_GPU_KERNEL_H_
