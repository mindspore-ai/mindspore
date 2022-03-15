/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_UNARYOP_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_UNARYOP_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <functional>
#include <vector>
#include <string>
#include <memory>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/unary_helper.h"

namespace mindspore {
namespace kernel {
template <typename T>
class UnaryOpGpuKernelMod : public NativeGpuKernelMod {
 public:
  UnaryOpGpuKernelMod() { ResetResource(); }
  ~UnaryOpGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    std::vector<void *> input_addrs;
    std::vector<void *> output_addrs;
    std::vector<void *> work_addrs;
    for (size_t idx = 0; idx < inputs.size(); ++idx) {
      void *cur_ptr = reinterpret_cast<void *>(GetDeviceAddress<T>(inputs, idx));
      input_addrs.emplace_back(cur_ptr);
    }
    for (size_t idx = 0; idx < outputs.size(); ++idx) {
      void *cur_ptr = reinterpret_cast<void *>(GetDeviceAddress<T>(outputs, idx));
      output_addrs.emplace_back(cur_ptr);
    }
    int flag = helper_ptr_->Process(input_addrs, output_addrs, work_addrs, stream_ptr);
    if (flag != 0) {
      return false;
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    helper_ptr_ = std::make_unique<cukernel::UnaryHelperGpuKernel<T>>(kernel_name);
    helper_ptr_->ResetResource();
    std::vector<std::vector<size_t>> input_shapes;
    std::vector<std::vector<size_t>> output_shapes;
    auto input_shape = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 0);
    auto output_shape = AnfAlgo::GetOutputDeviceShapeAdaptively(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name, "input");
    if (is_null_input_) {
      input_size_list_.emplace_back(0);
      output_size_list_.emplace_back(0);
      return true;
    }
    input_shapes.emplace_back(input_shape);
    output_shapes.emplace_back(output_shape);
    int flag = helper_ptr_->CalMemSize(input_shapes, output_shapes);
    if (flag != 0) {
      return false;
    }
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_ = helper_ptr_->GetInputSizeList();
    output_size_list_ = helper_ptr_->GetOutputSizeList();
    workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  }

 private:
  std::unique_ptr<cukernel::UnaryHelperGpuKernel<T>> helper_ptr_ = nullptr;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_UNARYOP_GPU_KERNEL_H_
