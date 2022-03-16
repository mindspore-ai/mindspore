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

#include "plugin/device/cpu/kernel/rl/tensor_array_read_kernel.h"
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_tensor_array.h"
#include "runtime/device/tensor_array_manager.h"

namespace mindspore {
namespace kernel {
using mindspore::device::TensorArrayMgr;
using mindspore::device::TensorArrayPtr;
TensorArrayReadCpuKernelMod::TensorArrayReadCpuKernelMod() : value_size_(0), type_(nullptr) {}

void TensorArrayReadCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  shapes_ = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "element_shape");
  type_ = common::AnfAlgo::GetNodeAttr<TypePtr>(kernel_node, "dtype");
  value_size_ = GetTypeByte(type_);
  for (auto i : shapes_) {
    value_size_ *= LongToSize(i);
  }
  input_size_list_.push_back(sizeof(int64_t));
  input_size_list_.push_back(sizeof(int64_t));
  output_size_list_.push_back(value_size_);
}

bool TensorArrayReadCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                         const std::vector<AddressPtr> &outputs) {
  auto handle_addr = GetDeviceAddress<int64_t>(inputs, 0);
  auto index = GetDeviceAddress<int64_t>(inputs, 1);
  auto out_value = GetDeviceAddress<unsigned char>(outputs, 0);
  MS_EXCEPTION_IF_NULL(handle_addr);
  MS_EXCEPTION_IF_NULL(index);
  MS_EXCEPTION_IF_NULL(out_value);
  int64_t index_host = index[0];
  TensorArrayPtr tensors_ = TensorArrayMgr::GetInstance().GetTensorArray(handle_addr[0]);
  MS_ERROR_IF_NULL(tensors_);
  if (!tensors_->CheckReadIndexLogical(index_host)) {
    MS_LOG(EXCEPTION) << "Invalid index " << index_host << " for read.";
  }
  auto value_addr = tensors_->Read(index_host);
  MS_LOG(DEBUG) << "Read value index:" << index_host;
  auto out_value_size = value_size_;
  auto ret = memcpy_s(out_value, out_value_size, value_addr->addr, value_size_);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "Memcpy failed, errorno(" << ret << ")";
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorArrayRead, TensorArrayReadCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
