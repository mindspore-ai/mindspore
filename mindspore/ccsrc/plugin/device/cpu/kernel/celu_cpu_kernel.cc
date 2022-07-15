/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/celu_cpu_kernel.h"
#include <utility>
#include <algorithm>
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
namespace mindspore {
namespace kernel {
namespace {

const std::vector<KernelAttr> kernel_attr = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}};
}  // namespace

bool CeluCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::CeLU>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "cast Celu ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  alpha_ = kernel_ptr->get_alpha();

  auto input_type_id = inputs[0]->GetDtype();
  if (input_type_id != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "celu kernel does not support " << TypeIdToString(input_type_id);
    return false;
  }
  unit_size_ = sizeof(float);

  return true;
}

int CeluCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs,
                             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  if (input_size_list_.size() != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be equal 1.";
    return KRET_RESIZE_FAILED;
  }
  input_elements_ = input_size_list_[0] / unit_size_;
  return ret;
}

std::vector<KernelAttr> CeluCpuKernelMod::GetOpSupport() { return kernel_attr; }

bool CeluCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                              const std::vector<AddressPtr> &outputs) {
  auto in_data = static_cast<float *>(inputs[0]->addr);
  auto out_data = static_cast<float *>(outputs[0]->addr);

  auto task = [this, in_data, out_data](size_t start, size_t end) {
    auto src = in_data + start;
    auto dst = out_data + start;
    auto length = end - start;
    for (size_t i = 0; i < length; ++i) {
      dst[i] = src[i] > 0 ? src[i] : (expm1(src[i] / alpha_) * alpha_);
    }
  };
  ParallelLaunchAutoSearch(task, input_elements_, this, &parallel_search_info_, pool_);

  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CeLU, CeluCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
