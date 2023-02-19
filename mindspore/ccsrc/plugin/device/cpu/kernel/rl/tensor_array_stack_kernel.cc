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
#include "kernel/common_utils.h"
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

void TensorArrayStackCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  ResetResource();
  kernel_node_ = kernel_node;
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  auto shape = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "element_shape");
  auto max_element = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "max_element");
  is_dynamic_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "is_dynamic_shape");
  auto size = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "size");
  for (auto i : shape) {
    shapes_.push_back(LongToSize(i));
  }
  type_ = common::AnfAlgo::GetNodeAttr<TypePtr>(kernel_node, "dtype");
  ele_size_ = GetTypeByte(type_);
  for (auto i : shapes_) {
    ele_size_ *= i;
  }
  if (is_dynamic_) {
    value_size_ = ele_size_ * LongToSize(max_element);
  } else {
    value_size_ = ele_size_ * LongToSize(size);
  }
  is_need_retrieve_output_shape_ = true;
}

void TensorArrayStackCpuKernelMod::InitInputOutputSize(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  output_size_list_.clear();
  input_size_list_.clear();
  output_size_list_.push_back(value_size_);
  input_size_list_.push_back(sizeof(int64_t));
}

void TensorArrayStackCpuKernelMod::PostExecute() {
  TensorArrayPtr tensors_ = TensorArrayMgr::GetInstance().GetTensorArray(handle_);
  MS_EXCEPTION_IF_NULL(tensors_);
  size_t tensor_size = tensors_->GetValidSize();
  auto shape = shapes_;
  (void)shape.insert(shape.cbegin(), tensor_size);
  MS_LOG(DEBUG) << "After postexecute, the real shape of TensorArrayStack is " << shape;
  common::AnfAlgo::SetOutputInferTypeAndShape({type_->type_id()}, {Convert2Long(shape)}, kernel_node_.lock().get());
}

void TensorArrayStackCpuKernelMod::ResetResource() noexcept {
  handle_ = 0;
  value_size_ = 0;
  ele_size_ = 0;
  is_dynamic_ = true;
  shapes_.clear();
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

bool TensorArrayStackCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                          const std::vector<AddressPtr> &outputs) {
  auto handle_addr = GetDeviceAddress<int64_t>(inputs, 0);
  auto out_value = GetDeviceAddress<unsigned char>(outputs, 0);
  MS_EXCEPTION_IF_NULL(out_value);
  MS_EXCEPTION_IF_NULL(handle_addr);

  // Set out_value to zeros when TensorArray in static size.
  if (!is_dynamic_) {
    auto ret = memset_s(out_value, outputs[0]->size, 0, value_size_);
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
  if (is_dynamic_) {
    PostExecute();
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorArrayStack, TensorArrayStackCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
