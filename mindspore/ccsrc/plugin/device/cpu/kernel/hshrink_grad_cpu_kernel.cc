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

#include "plugin/device/cpu/kernel/hshrink_grad_cpu_kernel.h"
#include <algorithm>
#include "mindspore/core/ops/grad/hshrink_grad.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/nnacl/fp32_grad/activation_grad_fp32.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kHShrinkGradInputsNum = 2;
constexpr size_t kHShrinkGradOutputsNum = 1;

const std::vector<KernelAttr> kernel_attr = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}};
}  // namespace

bool HShrinkGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.size() != kHShrinkGradInputsNum || outputs.size() != kHShrinkGradOutputsNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kHShrinkGradInputsNum << " and "
                  << kHShrinkGradOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  auto kernel_ptr = std::dynamic_pointer_cast<ops::HShrinkGrad>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "Cast HShrinkGrad ops failed!";
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

int HShrinkGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  input_elements_ = input_size_list_[0] / unit_size_;
  return static_cast<int>(KRET_OK);
}

bool HShrinkGradCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                     const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kHShrinkGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kHShrinkGradOutputsNum, kernel_name_);
  auto *dy = reinterpret_cast<float *>(inputs[kIndex0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(dy, false);
  auto *x = reinterpret_cast<float *>(inputs[kIndex1]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(x, false);
  auto *dx = reinterpret_cast<float *>(outputs[kIndex0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(dx, false);

  auto task = [dy, x, dx, this](size_t start, size_t end) {
    auto ret = HardShrinkGrad(dy + start, x + start, SizeToInt(end - start), dx + start, this->lambd_);
    if (ret != NNACL_OK) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', call NNACL HShrinkGrad function failed.";
      return false;
    }
    return true;
  };
  ParallelLaunchAutoSearch(task, input_elements_, this, &parallel_search_info_, pool_);
  return true;
}

std::vector<KernelAttr> HShrinkGradCpuKernelMod::GetOpSupport() { return kernel_attr; }

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, HShrinkGrad, HShrinkGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
