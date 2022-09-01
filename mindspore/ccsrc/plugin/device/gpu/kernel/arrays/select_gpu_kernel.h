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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SELECT_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SELECT_GPU_KERNEL_H

#include <vector>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/select_impl.cuh"
#include "utils/ms_context.h"

namespace mindspore {
namespace kernel {
template <typename T>
class SelectGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  SelectGpuKernelMod() : input_size_(0), output_size_(0), is_null_input_(false), kernel_name_("Select") {}
  ~SelectGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    bool *input_cond = GetDeviceAddress<bool>(inputs, 0);
    T *input_x = GetDeviceAddress<T>(inputs, 1);
    T *input_y = GetDeviceAddress<T>(inputs, 2);
    T *output = GetDeviceAddress<T>(outputs, 0);
    CalSelect(output_size_ / sizeof(T), input_cond, input_x, input_y, output, device_id_,
              reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    device_id_ = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    (void)CheckParam(kernel_node);
    auto shape_signed = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    if (IsDynamic(shape_signed)) {
      return true;
    }
    auto shape = Convert2SizeTClipNeg(shape_signed);
    is_null_input_ = CHECK_SHAPE_NULL(shape, kernel_name_, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    input_size_ = sizeof(bool);
    output_size_ = sizeof(T);
    for (size_t x : shape) {
      input_size_ = input_size_ * x;
      output_size_ = output_size_ * x;
    }
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = 0;
    output_size_ = 0;
    is_null_input_ = false;
    kernel_name_ = "Select";
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(output_size_);
    input_size_list_.push_back(output_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  void CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 3) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 3, but got " << input_num;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
    }
  }

  size_t input_size_;
  size_t output_size_;
  bool is_null_input_;
  std::string kernel_name_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SELECT_GPU_KERNEL_H
