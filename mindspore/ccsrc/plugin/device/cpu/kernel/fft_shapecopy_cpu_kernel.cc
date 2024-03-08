/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/fft_shapecopy_cpu_kernel.h"
#include <algorithm>
#include "ops/op_utils.h"
#include "kernel/kernel.h"
#include "utils/fft_helper.h"

namespace mindspore {
namespace kernel {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

bool FFTShapeCopyCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " valid cpu kernel does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int FFTShapeCopyCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  dout_shape_ = inputs[kIndex0]->GetShapeVector();
  shape_ = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  return KRET_OK;
}

template <typename T>
bool FFTShapeCopyCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                            const std::vector<kernel::KernelTensor *> &outputs) {
  auto *input_ptr = reinterpret_cast<T *>(inputs[kIndex0]->device_ptr());
  auto *output_ptr = reinterpret_cast<T *>(outputs[kIndex0]->device_ptr());

  auto ret = memset_s(output_ptr, outputs[kIndex0]->size(), 0, outputs[kIndex0]->size());
  if (ret != EOK) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', output buff memset failed. Error no: " << ret;
    return false;
  }

  ShapeCopy<T, T>(input_ptr, output_ptr, dout_shape_, shape_);
  return true;
}

#define ONE_DIM_CPU_REG(MS_Tin, MS_Tout, T)     \
  KernelAttr()                                  \
    .AddInputAttr(MS_Tin)           /* dout */  \
    .AddInputAttr(kNumberTypeInt64) /* shape */ \
    .AddOutputAttr(MS_Tout),                    \
    &FFTShapeCopyCpuKernelMod::LaunchKernel<T>

std::vector<std::pair<KernelAttr, FFTShapeCopyCpuKernelMod::FFTShapeCopyFunc>> FFTShapeCopyCpuKernelMod::func_list_ = {
  {ONE_DIM_CPU_REG(kNumberTypeFloat32, kNumberTypeFloat32, float)},
  {ONE_DIM_CPU_REG(kNumberTypeFloat64, kNumberTypeFloat64, double)},
  {ONE_DIM_CPU_REG(kNumberTypeComplex64, kNumberTypeComplex64, complex64)},
  {ONE_DIM_CPU_REG(kNumberTypeComplex128, kNumberTypeComplex128, complex128)}};

std::vector<KernelAttr> FFTShapeCopyCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, FFTShapeCopyFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FFTShapeCopy, FFTShapeCopyCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
