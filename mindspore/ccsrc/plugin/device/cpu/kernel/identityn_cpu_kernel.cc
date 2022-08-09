/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/identityn_cpu_kernel.h"
#include <algorithm>
#include <ctime>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
bool IdentityNCpuKernelMod::CheckType(TypeId idx_type, size_t idx) {
  bool is_in = (support_types_.find(idx_type) != support_types_.end());
  if (!is_in) {
    std::ostringstream buffer;
    buffer << "For primitive[IdentityN], the input arguments x[" << idx << "] must be a type of {";
    for (auto type : support_types_) {
      buffer << TypeIdLabel(type) << " ";
    }
    buffer << "}, but get " << TypeIdLabel(idx_type) << ".";
    MS_EXCEPTION(TypeError) << buffer.str();
  }
  return true;
}

void IdentityNCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  in_num_ = common::AnfAlgo::GetInputTensorNum(kernel_node);
  out_num_ = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  if (in_num_ != out_num_) {
    MS_EXCEPTION(ValueError) << "For IdentityN, input tensor number should same to outputs but get [" << in_num_
                             << "] != [" << out_num_ << "].";
  }
  for (size_t idx = 0; idx < in_num_; ++idx) {
    auto idx_type = AnfAlgo::GetInputDeviceDataType(kernel_node, idx);
    (void)CheckType(idx_type, idx);
    auto out_type = AnfAlgo::GetOutputDeviceDataType(kernel_node, idx);
    if (idx_type != out_type) {
      MS_EXCEPTION(TypeError) << "For IdentityN, input tensor datatype should be same to output. But datatype ["
                              << TypeIdLabel(idx_type) << "] != [" << TypeIdLabel(out_type) << "].";
    }
  }
}

bool IdentityNCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), in_num_, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), out_num_, kernel_name_);
  for (size_t idx = 0; idx < in_num_; ++idx) {
    auto idx_in_addr = inputs[idx]->addr;
    size_t idx_in_size = inputs[idx]->size;
    auto idx_out_addr = outputs[idx]->addr;
    size_t idx_out_size = outputs[idx]->size;
    if (idx_in_addr == idx_out_addr) {
      continue;
    }
    if (idx_in_size > idx_out_size) {
      MS_EXCEPTION(ValueError)
        << "For IdentityN, output tensor memory size less than input tensor memory size. in memory size: ["
        << idx_in_size << "] out memory size: [" << idx_out_size << "].";
    }
    int cpret = memcpy_s(idx_out_addr, idx_out_size, idx_in_addr, idx_in_size);
    if (cpret != EOK) {
      MS_EXCEPTION(MemoryError) << "IdentityN memcpy_s to output failed.";
    }
  }
  return true;
}

std::vector<KernelAttr> IdentityNCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {};
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IdentityN, IdentityNCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
