/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/rl/tensor_array_stack_kernel.h"
#include <algorithm>
#include "backend/kernel_compiler/common_utils.h"
#include "runtime/device/cpu/cpu_tensor_array.h"
#include "runtime/device/tensor_array_manager.h"

namespace mindspore {
namespace kernel {
using mindspore::device::TensorArrayMgr;
using mindspore::device::TensorArrayPtr;
TensorArrayCPUStackKernel::TensorArrayCPUStackKernel() : handle_(0), value_size_(0), ele_size_(0), type_(nullptr) {
  ResetResource();
}

const std::vector<size_t> &TensorArrayCPUStackKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &TensorArrayCPUStackKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &TensorArrayCPUStackKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

void TensorArrayCPUStackKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_node_ = kernel_node;
  auto shape = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "element_shape");
  auto max_element = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "max_element");
  for (auto i : shape) {
    shapes_.push_back(LongToSize(i));
  }
  type_ = AnfAlgo::GetNodeAttr<TypePtr>(kernel_node, "dtype");
  ele_size_ = GetTypeByte(type_);
  for (auto i : shapes_) {
    ele_size_ *= i;
  }
  value_size_ = ele_size_ * LongToSize(max_element);
  output_size_list_.push_back(value_size_);
  input_size_list_.push_back(sizeof(int64_t));
}

void TensorArrayCPUStackKernel::PostExecute() {
  TensorArrayPtr tensors_ = TensorArrayMgr::GetInstance().GetTensorArray(handle_);
  MS_EXCEPTION_IF_NULL(tensors_);
  size_t tensor_size = tensors_->GetValidSize();
  auto shape = shapes_;
  (void)shape.insert(shape.begin(), tensor_size);
  MS_LOG(DEBUG) << "After postexecute, the real shape of TensorArrayStack is " << shape;
  AnfAlgo::SetOutputInferTypeAndShape({type_->type_id()}, {shape}, kernel_node_.lock().get());
}

void TensorArrayCPUStackKernel::ResetResource() noexcept {
  handle_ = 0;
  value_size_ = 0;
  ele_size_ = 0;
  shapes_.clear();
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

bool TensorArrayCPUStackKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                       const std::vector<AddressPtr> &outputs) {
  auto handle_addr = GetDeviceAddress<int64_t>(inputs, 0);
  auto out_value = GetDeviceAddress<unsigned char>(outputs, 0);
  MS_EXCEPTION_IF_NULL(out_value);
  MS_EXCEPTION_IF_NULL(handle_addr);
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
      MS_LOG(EXCEPTION) << "Memcpy failed, errorno(" << ret << ")";
    }
  }
  PostExecute();
  return true;
}
}  // namespace kernel
}  // namespace mindspore
