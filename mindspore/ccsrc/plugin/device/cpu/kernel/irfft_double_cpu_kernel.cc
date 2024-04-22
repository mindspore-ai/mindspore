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

#include "plugin/device/cpu/kernel/irfft_double_cpu_kernel.h"
#include <algorithm>
#include "ops/op_utils.h"
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr float kDoubleFactor = 2.0;
constexpr int kOnsideDivisor = 2;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

template <typename T>
bool PartialDouble(T *input, T *output, const std::vector<int64_t> &input_shape, int64_t n, int64_t dim) {
  int64_t input_nums = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int64_t>());
  int64_t start_pos{1};
  int64_t end_pos = start_pos + n - (n / kOnsideDivisor + 1);

  // compute original offsets for each axes
  std::vector<int64_t> offsets(input_shape.size(), 0);
  for (size_t j = 0; j < input_shape.size(); j++) {
    int64_t pos = SizeToLong(j);
    offsets[j] = std::accumulate(input_shape.begin() + pos + 1, input_shape.end(), 1, std::multiplies<>());
  }

  for (int64_t i = 0; i < input_nums; ++i) {
    std::vector<int64_t> index(input_shape.size(), 0);
    int64_t flat_index = i;
    // compute original coordinates
    for (size_t dim_index = 0; dim_index < offsets.size(); ++dim_index) {
      index[dim_index] = flat_index / offsets[dim_index];
      flat_index %= offsets[dim_index];
    }
    T ele_val = input[i];
    T factor(kDoubleFactor, 0);
    if (index[dim] >= start_pos && index[dim] < end_pos) {
      ele_val = factor * ele_val;
    }
    output[i] = ele_val;
  }
  return true;
}
}  // namespace

bool IRFFTDoubleCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " valid cpu kernel does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int IRFFTDoubleCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  tensor_shape_ = inputs[kIndex0]->GetShapeVector();
  x_rank_ = SizeToLong(tensor_shape_.size());

  auto n_opt = inputs[kIndex1]->GetOptionalValueWithCheck<int64_t>();
  if (!n_opt.has_value()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', param [n] must be given. ";
  } else {
    n_ = n_opt.value();
  }

  // Get or set attribute s and dims.
  dim_ = inputs[kIndex2]->GetValueWithCheck<int64_t>();
  dim_ = dim_ < 0 ? x_rank_ + dim_ : dim_;

  return KRET_OK;
}

template <typename T>
bool IRFFTDoubleCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                           const std::vector<kernel::KernelTensor *> &outputs) {
  auto *input_ptr = reinterpret_cast<T *>(inputs[kIndex0]->device_ptr());
  auto *output_ptr = reinterpret_cast<T *>(outputs[kIndex0]->device_ptr());

  auto ret = memset_s(output_ptr, outputs[kIndex0]->size(), 0, outputs[kIndex0]->size());
  if (ret != EOK) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', output buff memset failed. Error no: " << ret;
    return false;
  }

  PartialDouble<T>(input_ptr, output_ptr, tensor_shape_, n_, dim_);
  return true;
}

#define ONE_DIM_CPU_REG(MS_Tin, MS_Tout, T)            \
  KernelAttr()                                         \
    .AddInputAttr(MS_Tin)                   /* dout */ \
    .AddOptionalInputAttr(kNumberTypeInt64) /* n */    \
    .AddInputAttr(kNumberTypeInt64)         /* dim */  \
    .AddOutputAttr(MS_Tout),                           \
    &IRFFTDoubleCpuKernelMod::LaunchKernel<T>

std::vector<std::pair<KernelAttr, IRFFTDoubleCpuKernelMod::IRFFTDoubleFunc>> IRFFTDoubleCpuKernelMod::func_list_ = {
  //  {ONE_DIM_CPU_REG(kNumberTypeFloat32, kNumberTypeFloat32, float)},
  //  {ONE_DIM_CPU_REG(kNumberTypeFloat64, kNumberTypeFloat64, double)},
  {ONE_DIM_CPU_REG(kNumberTypeComplex64, kNumberTypeComplex64, complex64)},
  {ONE_DIM_CPU_REG(kNumberTypeComplex128, kNumberTypeComplex128, complex128)}};

std::vector<KernelAttr> IRFFTDoubleCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, IRFFTDoubleFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IRFFTDouble, IRFFTDoubleCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
