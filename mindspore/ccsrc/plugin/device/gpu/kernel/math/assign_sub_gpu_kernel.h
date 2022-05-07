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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_ASSIGN_SUB_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_ASSIGN_SUB_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/assign_sub_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T>
class AssignSubFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  AssignSubFwdGpuKernelMod() : input_elements_(0) {}
  ~AssignSubFwdGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *input_addr2 = GetDeviceAddress<T>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    CalAssignSub(input_elements_, input_addr, input_addr2, output_addr, device_id_,
                 reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) {
    kernel_name_ = base_operator->name();

    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
    int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
    if (ret != 0) {
      return ret;
    }
    if (input_size_list_.size() != 2) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be equal 2.";
      return KRET_RESIZE_FAILED;
    }
    if (input_size_list_[0] != input_size_list_[1]) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "' input data size must be same.";
      return KRET_RESIZE_FAILED;
    }
    input_elements_ = input_size_list_[0] / sizeof(T);

    return KRET_OK;
  }

 private:
  size_t input_elements_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_ASSIGN_SUB_GPU_KERNEL_H_
