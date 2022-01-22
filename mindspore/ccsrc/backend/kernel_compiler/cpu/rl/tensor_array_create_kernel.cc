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
#include "backend/kernel_compiler/cpu/rl/tensor_array_create_kernel.h"
#include "backend/kernel_compiler/common_utils.h"
#include "runtime/device/cpu/cpu_tensor_array.h"
#include "runtime/device/tensor_array_manager.h"

namespace mindspore {
namespace kernel {
using mindspore::device::TensorArrayMgr;
using mindspore::device::cpu::CPUTensorArray;
using mindspore::device::cpu::CPUTensorArrayPtr;
TensorArrayCreateCpuKernelMod::TensorArrayCreateCpuKernelMod() : is_dynamic_(true), size_(0), type_(nullptr) {}

void TensorArrayCreateCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto shape = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "element_shape");
  for (auto i : shape) {
    shapes_.push_back(LongToSize(i));
  }
  type_ = AnfAlgo::GetNodeAttr<TypePtr>(kernel_node, "dtype");
  size_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "size");
  is_dynamic_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "dynamic_size");
  name_ = AnfAlgo::GetNodeAttr<std::string>(kernel_node, "name");
  output_size_list_.push_back(sizeof(int64_t));
}

bool TensorArrayCreateCpuKernelMod::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                           const std::vector<AddressPtr> &outputs) {
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
}  // namespace kernel
}  // namespace mindspore
