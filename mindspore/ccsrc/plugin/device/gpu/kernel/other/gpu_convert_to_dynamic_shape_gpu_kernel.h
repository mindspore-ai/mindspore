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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_OTHER_GPU_CONVERT_TO_DYNAMIC_SHAPE_GPU_KERNEL_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_OTHER_GPU_CONVERT_TO_DYNAMIC_SHAPE_GPU_KERNEL_H

#include <map>
#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class GpuConvertToDynamicShapeGpuKernelMod : public NativeGpuKernelMod {
 public:
  GpuConvertToDynamicShapeGpuKernelMod() { ResetResource(); }
  ~GpuConvertToDynamicShapeGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (input_shape_.size() == 0) {
      return true;
    }
    VARIABLE_NOT_USED(workspace);
    T *input_device_address = GetDeviceAddress<T>(inputs, 0);
    T *output_device_address = GetDeviceAddress<T>(outputs, 0);
    cuda_stream_ptr_ = stream_ptr;

    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemcpyAsync(output_device_address, input_device_address, input_size_ * sizeof(T), cudaMemcpyDeviceToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "Failed to copy gpu memory.");

    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) {
    MS_EXCEPTION_IF_NULL(base_operator);
    kernel_name_ = base_operator->GetPrim()->name();
    constexpr size_t input_num = 1;
    if (inputs.size() != input_num) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 1, but got " << inputs.size();
    }
    is_need_retrieve_output_shape_ = true;
    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) {
    int ret = KernelMod::Resize(base_operator, inputs, outputs);
    if (ret != KRET_OK && ret != KRET_UNKNOWN_OUT_SHAPE) {
      return ret;
    }
    input_shape_ = inputs[0]->GetShapeVector();
    outputs_[0]->SetShapeVector(input_shape_);
    input_size_ = 1;
    for (const auto &e : input_shape_) {
      input_size_ *= e;
    }
    InitSizeLists();
    return ret;
  }

  void ResetResource() noexcept {
    cuda_stream_ptr_ = nullptr;
    input_shape_.clear();
    input_size_ = 1;
    input_size_list_.clear();
    output_size_list_.clear();
  }

 protected:
  void InitSizeLists() {
    input_size_list_.clear();
    output_size_list_.clear();
    input_size_list_.push_back(input_size_ * sizeof(T));
    output_size_list_.push_back(input_size_ * sizeof(T));
  }

 private:
  void *cuda_stream_ptr_;
  ShapeVector input_shape_;
  int64_t input_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_OTHER_GPU_CONVERT_TO_DYNAMIC_SHAPE_GPU_KERNEL_H
