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
#include "plugin/device/cpu/kernel/rl/tensor_array_write_kernel.h"
#include <memory>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_tensor_array.h"
#include "runtime/device/tensor_array_manager.h"
#include "plugin/device/cpu/hal/hardware/cpu_memory_pool.h"
namespace mindspore {
namespace kernel {
constexpr size_t kSecondInputIndex = 2;
using mindspore::device::TensorArrayMgr;
using mindspore::device::cpu::CPUTensorArray;
using mindspore::device::cpu::CPUTensorArrayPtr;
TensorArrayWriteCpuKernelMod::TensorArrayWriteCpuKernelMod() : value_size_(0), type_(kTypeUnknown) {}

void TensorArrayWriteCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  type_ = common::AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, kSecondInputIndex);
  shapes_ = AnfAlgo::GetInputDeviceShape(kernel_node, kSecondInputIndex);
  value_size_ = GetTypeByte(TypeIdToType(type_));
  for (auto i : shapes_) {
    value_size_ *= static_cast<size_t>(i);
  }
  input_size_list_.push_back(sizeof(int64_t));
  input_size_list_.push_back(sizeof(int64_t));
  input_size_list_.push_back(sizeof(value_size_));
  output_size_list_.push_back(sizeof(int64_t));
}

bool TensorArrayWriteCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                          const std::vector<AddressPtr> &) {
  auto handle_addr = GetDeviceAddress<int64_t>(inputs, 0);
  auto index = GetDeviceAddress<int64_t>(inputs, 1);
  auto value = GetDeviceAddress<unsigned char>(inputs, kSecondInputIndex);
  MS_EXCEPTION_IF_NULL(handle_addr);
  MS_EXCEPTION_IF_NULL(index);
  MS_EXCEPTION_IF_NULL(value);
  int64_t index_host = index[0];
  CPUTensorArrayPtr tensors_ =
    std::dynamic_pointer_cast<CPUTensorArray>(TensorArrayMgr::GetInstance().GetTensorArray(handle_addr[0]));
  MS_EXCEPTION_IF_NULL(tensors_);
  if (!tensors_->CheckValue(type_, shapes_)) {
    MS_LOG(EXCEPTION) << "Invalid input data for tensor array write op.";
  }
  // Manage the value : create/reuse a device memory, and copy the input value to it.
  AddressPtr dev_addr = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(dev_addr);
  if (tensors_->GetRealSize() > LongToSize(index_host)) {
    dev_addr->addr = tensors_->Read(index_host)->addr;
  } else {
    dev_addr->addr = mindspore::device::cpu::CPUMemoryPool::GetInstance().AllocTensorMem(value_size_);
    MS_LOG(DEBUG) << "Create tensor " << dev_addr->addr << ", size " << value_size_;
  }
  MS_EXCEPTION_IF_NULL(dev_addr->addr);
  dev_addr->size = value_size_;
  auto ret = memcpy_s(dev_addr->addr, dev_addr->size, value, value_size_);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "Memcpy failed, errorno(" << ret << ")";
  }
  if (tensors_->Write(index_host, dev_addr)) {
    MS_LOG(DEBUG) << "Write to tensorarry succeed, index " << index_host;
  } else {
    MS_LOG(EXCEPTION) << "Failed to write.";
  }
  return true;
}

std::vector<KernelAttr> TensorArrayWriteCpuKernelMod::support_list_ = {  // index int64
  KernelAttr()
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt64)
    .AddOutputAttr(kNumberTypeInt64),
  KernelAttr()
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt32)
    .AddOutputAttr(kNumberTypeInt64),
  KernelAttr()
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt16)
    .AddOutputAttr(kNumberTypeInt64),
  KernelAttr()
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeUInt64)
    .AddOutputAttr(kNumberTypeInt64),
  KernelAttr()
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeUInt32)
    .AddOutputAttr(kNumberTypeInt64),
  KernelAttr()
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeUInt16)
    .AddOutputAttr(kNumberTypeInt64),
  KernelAttr()
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeUInt8)
    .AddOutputAttr(kNumberTypeInt64),
  KernelAttr()
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeFloat32)
    .AddOutputAttr(kNumberTypeInt64),
  KernelAttr()
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeFloat16)
    .AddOutputAttr(kNumberTypeInt64),
  KernelAttr()
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeBool)
    .AddOutputAttr(kNumberTypeInt64),
  // index int32
  KernelAttr()
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt32)
    .AddInputAttr(kNumberTypeInt64)
    .AddOutputAttr(kNumberTypeInt64),
  KernelAttr()
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt32)
    .AddInputAttr(kNumberTypeInt32)
    .AddOutputAttr(kNumberTypeInt64),
  KernelAttr()
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt32)
    .AddInputAttr(kNumberTypeInt16)
    .AddOutputAttr(kNumberTypeInt64),
  KernelAttr()
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt32)
    .AddInputAttr(kNumberTypeUInt64)
    .AddOutputAttr(kNumberTypeInt64),
  KernelAttr()
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt32)
    .AddInputAttr(kNumberTypeUInt32)
    .AddOutputAttr(kNumberTypeInt64),
  KernelAttr()
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt32)
    .AddInputAttr(kNumberTypeUInt16)
    .AddOutputAttr(kNumberTypeInt64),
  KernelAttr()
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt32)
    .AddInputAttr(kNumberTypeUInt8)
    .AddOutputAttr(kNumberTypeInt64),
  KernelAttr()
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt32)
    .AddInputAttr(kNumberTypeFloat32)
    .AddOutputAttr(kNumberTypeInt64),
  KernelAttr()
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt32)
    .AddInputAttr(kNumberTypeFloat16)
    .AddOutputAttr(kNumberTypeInt64),
  KernelAttr()
    .AddInputAttr(kNumberTypeInt64)
    .AddInputAttr(kNumberTypeInt32)
    .AddInputAttr(kNumberTypeBool)
    .AddOutputAttr(kNumberTypeInt64)};

std::vector<KernelAttr> TensorArrayWriteCpuKernelMod::GetOpSupport() { return support_list_; }

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorArrayWrite, TensorArrayWriteCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
