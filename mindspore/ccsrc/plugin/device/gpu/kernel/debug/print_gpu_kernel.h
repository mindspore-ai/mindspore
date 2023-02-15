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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DEBUG_PRINT_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DEBUG_PRINT_GPU_KERNEL_H_

#include <unordered_map>
#include <map>
#include <string>
#include <vector>
#include <tuple>
#include "ir/tensor.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "kernel/common_utils.h"

using mindspore::tensor::Tensor;

namespace mindspore {
namespace kernel {
class PrintGpuKernelMod : public NativeGpuKernelMod {
 public:
  PrintGpuKernelMod() {}
  ~PrintGpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
              const std::vector<AddressPtr> &outputs, void *cuda_stream) override;
  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  void InitDeviceData(const std::vector<AddressPtr> &inputs, std::vector<void *> *input_device_data);
  std::vector<int64_t> SetInputFlag(std::vector<int64_t> *string_pos, size_t input_tensor_num);
  std::string GetString(size_t tensor_index, size_t original_index, void *input_host_data);

 private:
  std::vector<std::string> string_value_;
  std::vector<int64_t> string_pos_;
  std::vector<int64_t> input_flag_;
  std::unordered_map<int64_t, int64_t> value_type_;
  // size_in_byte, typeid
  std::vector<std::tuple<size_t, TypeId>> input_info_;
  std::vector<std::vector<int64_t>> input_shape_;

  bool is_null_input_{false};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DEBUG_PRINT_GPU_KERNEL_H_
