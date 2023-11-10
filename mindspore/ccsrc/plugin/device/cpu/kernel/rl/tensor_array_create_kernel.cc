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
#include <memory>
#include "plugin/device/cpu/kernel/rl/tensor_array_create_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_tensor_array.h"
#include "runtime/device/tensor_array_manager.h"

namespace mindspore {
namespace kernel {
using mindspore::device::TensorArrayMgr;
using mindspore::device::cpu::CPUTensorArray;
using mindspore::device::cpu::CPUTensorArrayPtr;
TensorArrayCreateCpuKernelMod::TensorArrayCreateCpuKernelMod() : is_dynamic_(true), size_(0), type_(nullptr) {}

int TensorArrayCreateCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  shapes_ = GetValue<std::vector<int64_t>>(primitive_->GetAttr("element_shape"));
  type_ = GetValue<TypePtr>(primitive_->GetAttr("dtype"));
  size_ = GetValue<int64_t>(primitive_->GetAttr("size"));
  is_dynamic_ = GetValue<bool>(primitive_->GetAttr("dynamic_size"));
  name_ = GetValue<std::string>(primitive_->GetAttr("name"));
  output_size_list_.clear();
  output_size_list_.push_back(sizeof(int64_t));
  return KRET_OK;
}

bool TensorArrayCreateCpuKernelMod::Launch(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
                                           const std::vector<KernelTensor *> &outputs) {
  // Create a tensorarray, and generate an unique handle.
  int64_t tensor_array_handle = TensorArrayMgr::GetInstance().GetHandleCount();
  auto name = "CPUTensorArray_" + name_ + "_" + std::to_string(tensor_array_handle);
  CPUTensorArrayPtr tensor_array = std::make_shared<CPUTensorArray>(name, type_, shapes_);
  MS_EXCEPTION_IF_NULL(tensor_array);
  tensor_array->SetMaxSize(size_, is_dynamic_);
  auto out_addr = GetDeviceAddress<int64_t>(outputs, 0);
  MS_EXCEPTION_IF_NULL(out_addr);
  // Set handle to out_addr.
  out_addr[0] = tensor_array_handle;
  MS_LOG(DEBUG) << "Create handle id " << tensor_array_handle;
  // Put tensorarray to a saved map : map<handle, tensorarray> in tensorarray manager.
  TensorArrayMgr::GetInstance().AddTensorArray(tensor_array_handle, tensor_array);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorArray, TensorArrayCreateCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
