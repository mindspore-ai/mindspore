/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_MESHGRID_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_MESHGRID_GPU_KERNEL_H

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/oneslike_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/math/broadcast_gpu_kernel.h"
#include "mindspore/core/ops/meshgrid.h"

namespace mindspore {
namespace kernel {
template <typename T>
class MeshgridGpuKernelMod : public NativeGpuKernelMod {
 public:
  MeshgridGpuKernelMod() {}
  ~MeshgridGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    T *ones_device = GetDeviceAddress<T>(workspace, 0);
    CalOnesLike(output_size_, static_cast<T *>(nullptr), ones_device, reinterpret_cast<cudaStream_t>(cuda_stream));

    std::vector<size_t> broadcasted_ones_shape(MAX_DIMS, 1);
    for (size_t i = 0; i < output_shape_.size(); i++) {
      broadcasted_ones_shape[i] = output_shape_[i];
    }

    for (size_t i = 0; i < outputs.size(); i++) {
      T *input_device = GetDeviceAddress<T>(inputs, i);
      T *output_device = GetDeviceAddress<T>(outputs, i);
      std::vector<size_t> broadcasted_input_shape(MAX_DIMS, 1);
      broadcasted_input_shape[i] = input_shapes_[i];

      if (swap_indexing_ && i < 2) {
        std::swap(broadcasted_input_shape[0], broadcasted_input_shape[1]);
      }

      BroadcastArith(broadcasted_input_shape, broadcasted_ones_shape, output_shape_, BROADCAST_TYPE_MUL, input_device,
                     ones_device, output_device, reinterpret_cast<cudaStream_t>(cuda_stream));
    }

    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    auto kernel_ptr = std::dynamic_pointer_cast<ops::Meshgrid>(base_operator);
    MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);
    kernel_name_ = kernel_ptr->name();
    std::string indexing = kernel_ptr->get_indexing();
    if (indexing == "xy") {
      swap_indexing_ = true;
    } else if (indexing == "ij") {
      swap_indexing_ = false;
    } else {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the value of 'indexing' must be \"xy\" or \"ij\", but got "
                    << indexing;
      return false;
    }
    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override {
    int ret = KRET_OK;
    if ((ret = KernelMod::Resize(base_operator, inputs, outputs)) != 0) {
      return ret;
    }

    input_size_ = 1;
    input_count_ = static_cast<size_t>(input_size_list_.size());
    for (size_t i = 0; i < input_count_; i++) {
      auto input_shape = inputs[i]->GetShapeVector();
      if (input_shape.size() < 1) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of input[" << i << "] cannot be less than 1, "
                      << "but got " << input_shape.size();
        return KRET_RESIZE_FAILED;
      }
      size_t input_size = input_shape[0];
      input_shapes_.push_back(input_size);
      input_size_ *= input_size;
    }

    output_size_ = 1;
    output_count_ = static_cast<size_t>(output_size_list_.size());

    // inferred shape swaps output shape for us if needed
    auto shape_signed = outputs[kIndex0]->GetShapeVector();
    output_shape_ = Convert2SizeTClipNeg(shape_signed);
    is_null_input_ = CHECK_SHAPE_NULL(output_shape_, kernel_name_, "output");
    if (is_null_input_) {
      workspace_size_list_.push_back(output_size_ * sizeof(T));
      return KRET_OK;
    }

    if (output_count_ != input_count_) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the number of inputs and outputs must be the same, but got the number of inputs: "
                    << input_count_ << ", the number of outputs: " << output_count_;
      return KRET_RESIZE_FAILED;
    }

    for (size_t i = 0; i < output_shape_.size(); i++) {
      output_size_ *= output_shape_[i];
    }

    // need to pad output shape with ones for broadcast kernel
    int need_broadcast_size = MAX_DIMS - output_shape_.size();
    for (int i = 0; i < need_broadcast_size; i++) {
      output_shape_.push_back(1);
    }

    workspace_size_list_.push_back(output_size_ * sizeof(T));
    return KRET_OK;
  }

 private:
  std::vector<size_t> input_shapes_;
  std::vector<size_t> output_shape_;
  size_t input_size_;
  size_t input_count_;
  size_t output_size_;
  size_t output_count_;
  bool swap_indexing_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_MESHGRID_GPU_KERNEL_H
