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

#include "plugin/device/cpu/kernel/hshrink_cpu_kernel.h"
#include <algorithm>
#include "mindspore/core/ops/hshrink.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/nnacl/errorcode.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/activation_fp32.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kHShrinkInputsNum = 1;
constexpr size_t kHShrinkOutputsNum = 1;

const std::vector<KernelAttr> kernel_attr = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}};
}  // namespace

bool HShrinkCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.size() != kHShrinkInputsNum || outputs.size() != kHShrinkOutputsNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kHShrinkInputsNum << " and "
                  << kHShrinkOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  auto kernel_ptr = std::dynamic_pointer_cast<ops::HShrink>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "Cast HShrink ops failed!";
    return false;
  }
  lambd_ = kernel_ptr->get_lambd();

  auto input_type_id = inputs[0]->GetDtype();
  if (input_type_id != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "HShrink kernel does not support " << TypeIdToString(input_type_id);
    return false;
  }
  unit_size_ = sizeof(float);
  return true;
}

int HShrinkCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  input_elements_ = input_size_list_[0] / unit_size_;
  return static_cast<int>(KRET_OK);
}

bool HShrinkCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                 const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kHShrinkInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kHShrinkOutputsNum, kernel_name_);
  auto *input = reinterpret_cast<float *>(inputs[kIndex0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(input, false);
  auto *output = reinterpret_cast<float *>(outputs[kIndex0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(output, false);

  auto task = [input, output, this](size_t start, size_t end) {
    auto ret = HardShrink(input + start, SizeToInt(end - start), output + start, this->lambd_);
    if (ret != NNACL_OK) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', call NNACL HShrink function failed. Error code: " << ret;
      return false;
    }
    return true;
  };
  ParallelLaunchAutoSearch(task, input_elements_, this, &parallel_search_info_, pool_);
  return true;
}

std::vector<KernelAttr> HShrinkCpuKernelMod::GetOpSupport() { return kernel_attr; }

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, HShrink, HShrinkCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
