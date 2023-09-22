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

#include "plugin/device/cpu/kernel/rl/tensor_array_stack_kernel.h"
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_tensor_array.h"
#include "runtime/device/tensor_array_manager.h"

namespace mindspore {
namespace kernel {
using mindspore::device::TensorArrayMgr;
using mindspore::device::TensorArrayPtr;
TensorArrayStackCpuKernelMod::TensorArrayStackCpuKernelMod()
    : handle_(0), value_size_(0), ele_size_(0), type_(nullptr), is_dynamic_(true) {
  ResetResource();
}

bool TensorArrayStackCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  ResetResource();
  auto shape = GetValue<std::vector<int64_t>>(primitive_->GetAttr("element_shape"));
  auto max_element = GetValue<int64_t>(primitive_->GetAttr("max_element"));
  is_dynamic_ = GetValue<bool>(primitive_->GetAttr("is_dynamic_shape"));
  auto size = GetValue<int64_t>(primitive_->GetAttr("size"));
  for (auto i : shape) {
    shapes_.push_back(LongToSize(i));
  }
  type_ = GetValue<TypePtr>(primitive_->GetAttr("dtype"));
  ele_size_ = GetTypeByte(type_);
  for (auto i : shapes_) {
    ele_size_ *= i;
  }
  if (is_dynamic_) {
    value_size_ = ele_size_ * LongToSize(max_element);
  } else {
    value_size_ = ele_size_ * LongToSize(size);
  }
  return true;
}

int TensorArrayStackCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  output_size_list_.clear();
  output_size_list_.push_back(value_size_);
  return KRET_OK;
}

void TensorArrayStackCpuKernelMod::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                                            const std::vector<KernelTensor *> &outputs) {
  if (!is_dynamic_) {
    return;
  }
  TensorArrayPtr tensors_ = TensorArrayMgr::GetInstance().GetTensorArray(handle_);
  MS_EXCEPTION_IF_NULL(tensors_);
  size_t tensor_size = tensors_->GetValidSize();
  auto shape = shapes_;
  (void)shape.insert(shape.cbegin(), tensor_size);
  MS_LOG(DEBUG) << "After postexecute, the real shape of TensorArrayStack is " << shape;
  outputs[kIndex0]->SetShapeVector(Convert2Long(shape));
  outputs[kIndex0]->set_size(value_size_);
}

void TensorArrayStackCpuKernelMod::ResetResource() noexcept {
  handle_ = 0;
  value_size_ = 0;
  ele_size_ = 0;
  is_dynamic_ = true;
  shapes_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

bool TensorArrayStackCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &,
                                          const std::vector<KernelTensor *> &outputs) {
  auto handle_addr = GetDeviceAddress<int64_t>(inputs, 0);
  auto out_value = GetDeviceAddress<unsigned char>(outputs, 0);
  MS_EXCEPTION_IF_NULL(out_value);
  MS_EXCEPTION_IF_NULL(handle_addr);

  // Set out_value to zeros when TensorArray in static size.
  if (!is_dynamic_) {
    auto ret = memset_s(out_value, outputs[0]->size(), 0, value_size_);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset failed, errorno(" << ret << ")";
    }
  }
  handle_ = handle_addr[0];
  TensorArrayPtr tensors_ = TensorArrayMgr::GetInstance().GetTensorArray(handle_);
  MS_EXCEPTION_IF_NULL(tensors_);
  if (tensors_->GetValidSize() > tensors_->GetRealSize()) {
    MS_LOG(EXCEPTION) << "Invalid TensorArray size, maybe should Clear() TensorArray before next usage.";
  }
  for (size_t i = 0; i < tensors_->GetValidSize(); i++) {
    auto out_ele_size = ele_size_;
    auto src_addr = tensors_->GetTensorAddr(i);
    MS_EXCEPTION_IF_NULL(src_addr);
    auto ret = memcpy_s(out_value + ele_size_ * i, out_ele_size, src_addr, ele_size_);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy failed, errorno(" << ret << ")";
    }
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorArrayStack, TensorArrayStackCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
