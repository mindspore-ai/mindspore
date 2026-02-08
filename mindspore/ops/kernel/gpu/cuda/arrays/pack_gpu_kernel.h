/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_PACK_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_PACK_GPU_KERNEL_H

#include <vector>
#include <string>
#include <memory>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/pack.cuh"
#include "mindspore/ops/infer/stack.h"
#include "ir/format_utils.h"

namespace mindspore {
namespace kernel {
template <typename T>
class PackFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  PackFwdGpuKernelMod()
      : axis_(0), input_num_(1), output_size_(0), dims_behind_axis_(1), inputs_host_(nullptr), kernel_name_("Pack") {}
  ~PackFwdGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    T *output = GetDeviceAddress<T>(outputs, 0);
    T **inputs_array = GetDeviceAddress<T *>(workspace, 0);
    for (size_t i = 0; i < inputs.size(); i++) {
      inputs_host_[i] = GetDeviceAddress<T>(inputs, i);
    }
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(inputs_array,  // NOLINT
                      inputs_host_.get(), sizeof(T *) * input_num_, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "Pack opt cudaMemcpyAsync inputs failed");
    auto status = PackKernel(output_size_, input_num_, dims_behind_axis_, inputs_array, output,
                             reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    int ret = KernelMod::Resize(inputs, outputs);
    if (ret != KRET_OK) {
      return ret;
    }
    axis_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("axis")));
    if (axis_ < 0) {
      auto input_shape = inputs.at(kIndex0)->GetShapeVector();
      axis_ += (SizeToInt(input_shape.size()) + 1);
    }
    auto origin_data_format = kOpFormat_DEFAULT;
    auto input_format = GetFormatFromEnumToStr(inputs[0]->format());
    axis_ = AxisTransform(origin_data_format, input_format, axis_);

    input_num_ = inputs.size();
    inputs_host_ = std::make_unique<T *[]>(input_num_);
    dims_behind_axis_ = 1;
    for (size_t i = 0; i < input_num_; i++) {
      size_t input_size = 1;
      auto input_shape = inputs.at(i)->GetShapeVector();
      for (size_t j = 0; j < input_shape.size(); j++) {
        input_size *= static_cast<size_t>(input_shape[j]);
        if (i == 0 && j >= IntToSize(axis_)) {
          dims_behind_axis_ *= static_cast<size_t>(input_shape[j]);
        }
      }
    }
    workspace_size_list_.push_back(sizeof(T *) * input_num_);

    auto output_shape = outputs.at(kIndex0)->GetShapeVector();
    output_size_ = 1;
    for (size_t i = 0; i < output_shape.size(); i++) {
      output_size_ *= static_cast<size_t>(output_shape[i]);
    }
    return KRET_OK;
  }
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    auto output_num = outputs.size();
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
    }
    return true;
  }

  void ResetResource() noexcept {
    axis_ = 0;
    input_num_ = 1;
    output_size_ = 0;
    dims_behind_axis_ = 1;
    inputs_host_ = nullptr;
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 private:
  int axis_;
  size_t input_num_;
  size_t output_size_;
  size_t dims_behind_axis_;
  std::unique_ptr<T *[]> inputs_host_;
  std::string kernel_name_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_PACK_GPU_KERNEL_H
