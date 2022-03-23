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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_STRIDED_SLICE_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_STRIDED_SLICE_GRAD_GPU_KERNEL_H_

#include <vector>
#include <bitset>
#include <algorithm>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/arrays/strided_slice_gpu_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/slice_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr int DynamicInputNum = 5;
template <typename T, typename S = int64_t>
class StridedSliceGradGpuKernelMod : public NativeGpuKernelMod, public StridedSliceGpuCommon {
 public:
  StridedSliceGradGpuKernelMod() = default;
  ~StridedSliceGradGpuKernelMod() override = default;

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
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num == DynamicInputNum) {
      is_dynamic_attr_ = true;
    }
    if (is_dynamic_attr_) {
      GetDynamicAttrIntValue(kernel_node, kShapexIndex_, &shapex_);
      GetDynamicAttrIntValue(kernel_node, kBeginIndex_, &begin_);
      GetDynamicAttrIntValue(kernel_node, kEndIndex_, &end_);
      GetDynamicAttrIntValue(kernel_node, kStrideIndex_, &strides_);
    } else {
      shapex_ = GetAttr<std::vector<int64_t>>(kernel_node, "shapex");
    }
    for (auto x : shapex_) {
      input_shape_.push_back(static_cast<size_t>(x));
    }
    if (input_shape_.size() > MAX_DIMS) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of input cannot be greater than " << MAX_DIMS
                        << ", but got " << input_shape_.size();
    }

    CollectInfo(kernel_node, is_dynamic_attr_);
    InitSizeLists();
    return true;
  }
  void ResetResource() noexcept override {
    ResetSizeLists();
    begin_.clear();
    end_.clear();
    strides_.clear();
    input_shape_.clear();
    output_shape_.clear();
    is_dynamic_attr_ = false;
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
  bool is_null_input_{false};
  bool is_dynamic_attr_{false};
  bool get_dynamic_attr_value_{false};
  static constexpr size_t kShapexIndex_{1};
  static constexpr size_t kBeginIndex_{2};
  static constexpr size_t kEndIndex_{3};
  static constexpr size_t kStrideIndex_{4};
  std::vector<int64_t> shapex_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_STRIDED_SLICE_GRAD_GPU_KERNEL_H_
