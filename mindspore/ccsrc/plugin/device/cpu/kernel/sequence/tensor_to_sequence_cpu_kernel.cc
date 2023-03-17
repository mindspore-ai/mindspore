/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/sequence/tensor_to_sequence_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kTensorToTuple = "TensorToTuple";
constexpr auto kTensorToScalar = "TensorToScalar";
constexpr size_t kInputNum = 1;
constexpr size_t kOutputNum = 1;
}  // namespace

bool TensorToSeqCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (kernel_name_ != kernel_type_) {
    MS_LOG(EXCEPTION) << "Suppose to be " << kernel_type_ << " but got " << kernel_name_;
  }
  return true;
}

int TensorToSeqCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  return KRET_OK;
}

bool TensorToSeqCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                     const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  const auto input_addr = inputs[0]->addr;
  auto output_addr = outputs[0]->addr;
  auto input_size = inputs[0]->size;
  auto output_size = outputs[0]->size;
  if (input_size != output_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the size of 'input_x': {" << inputs[0]->size
                      << "} is not equal to the size of output: {" << outputs[0]->size << "}";
  }
  if (input_size != 0) {
    auto cp_ret = memcpy_s(output_addr, output_size, input_addr, input_size);
    if (cp_ret != EOK) {
      MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", memcpy error, errorno: " << cp_ret;
    }
  }
  return true;
}

std::vector<KernelAttr> TensorToSeqCpuKernelMod::sequence_list_ = {
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat32),
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat64),
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
};

std::vector<KernelAttr> TensorToSeqCpuKernelMod::scalar_list_ = {
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kObjectTypeNumber, kNumberTypeFloat32),
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kObjectTypeNumber, kNumberTypeFloat64),
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt32),
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
};

std::map<std::string, std::vector<KernelAttr>> TensorToSeqCpuKernelMod::kernel_attr_lists_ = {
  {kTensorToTuple, sequence_list_}, {kTensorToScalar, scalar_list_}};

std::vector<KernelAttr> TensorToSeqCpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_lists_.find(kernel_type_);
  if (iter == kernel_attr_lists_.end()) {
    MS_LOG(ERROR) << "For prim[" << kernel_type_ << "], it don't support.";
    return std::vector<KernelAttr>{};
  }
  return iter->second;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, TensorToTuple,
                                 []() { return std::make_shared<TensorToSeqCpuKernelMod>(kTensorToTuple); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, TensorToScalar,
                                 []() { return std::make_shared<TensorToSeqCpuKernelMod>(kTensorToScalar); });
}  // namespace kernel
}  // namespace mindspore
