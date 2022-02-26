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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ARGMAXANDMINWITHVALUE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ARGMAXANDMINWITHVALUE_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/general_reduction_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T, typename S>
class ArgMaxAndMinWithValueGpuKernelMod : public NativeGpuKernelMod {
 public:
  ArgMaxAndMinWithValueGpuKernelMod() { ResetResource(); }
  ~ArgMaxAndMinWithValueGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input = GetDeviceAddress<T>(inputs, 0);
    T *output = GetDeviceAddress<T>(outputs, 1);
    S *index = GetDeviceAddress<S>(outputs, 0);
    CalGeneralReduction(small_, input, bound_, outerSize_, innerSize_, index, output,
                        reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    small_ = (kernel_name == "ArgMinWithValue") ? true : false;
    std::vector<size_t> shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto output_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 1);
    is_null_input_ =
      CHECK_SHAPE_NULL(shape, kernel_name, "input") || CHECK_SHAPE_NULL(output_shape, kernel_name, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    int64_t dims = SizeToLong(shape.size());
    int64_t axis = GetAttr<int64_t>(kernel_node, "axis");
    if (axis < -dims || axis >= dims) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the 'axis' should be in the range [-" << dims << "," << dims
                        << "), but got " << axis;
    }
    if (axis < 0) {
      axis += dims;
    }
    input_size_ = sizeof(T);
    for (auto x : shape) {
      input_size_ *= x;
    }
    output_size_ = sizeof(S);
    for (auto x : output_shape) {
      output_size_ *= x;
    }
    bound_ = static_cast<S>(shape[axis]);
    if (shape[axis] != static_cast<size_t>(bound_)) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the value of shape[axis] should be "
                        << static_cast<size_t>(bound_) << ", but got " << shape[axis];
    }
    outerSize_ = 1;
    for (int64_t i = axis - 1; i >= 0; i--) {
      outerSize_ *= shape[i];
    }
    innerSize_ = 1;
    for (int64_t i = axis + 1; i < dims; i++) {
      innerSize_ *= shape[i];
    }
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = 0;
    output_size_ = 0;
    bound_ = 0;
    outerSize_ = 0;
    innerSize_ = 0;
    is_null_input_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    output_size_list_.push_back(output_size_ / sizeof(S) * sizeof(T));
  }

 private:
  bool small_ = false;
  size_t input_size_;
  size_t output_size_;
  S bound_;
  size_t outerSize_;
  size_t innerSize_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ARGMAXANDMINWITHVALUE_GPU_KERNEL_H_
