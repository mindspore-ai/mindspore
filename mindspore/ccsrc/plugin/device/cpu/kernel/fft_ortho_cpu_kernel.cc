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

#include "plugin/device/cpu/kernel/fft_ortho_cpu_kernel.h"
#include <algorithm>
#include <set>
#include <cmath>
#include "ops/op_utils.h"
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

template <typename T>
bool Orthogonalize(T *input, T *output, const std::vector<int64_t> &input_shape, const std::vector<int64_t> dims,
                   bool forward) {
  int64_t input_nums = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int64_t>());
  std::set<int64_t> ortho_dims(dims.begin(), dims.end());
  T normal_factor{2};
  T head_factor{4};

  // compute original offsets for each axes
  std::vector<int64_t> offsets(input_shape.size(), 0);
  for (size_t j = 0; j < input_shape.size(); j++) {
    int64_t pos = SizeToLong(j);
    offsets[j] = std::accumulate(input_shape.begin() + pos + 1, input_shape.end(), 1, std::multiplies<>());
  }

  for (int64_t i = 0; i < input_nums; ++i) {
    std::vector<int64_t> index(input_shape.size(), 0);
    int64_t flat_index = i;
    T ele_factor{1};
    // compute original coordinates
    for (size_t dim_index = 0; dim_index < offsets.size(); ++dim_index) {
      index[dim_index] = flat_index / offsets[dim_index];
      flat_index %= offsets[dim_index];
      if (ortho_dims.find(static_cast<int64_t>(dim_index)) != ortho_dims.end()) {
        ele_factor = index[dim_index] == 0 ? ele_factor * head_factor : ele_factor * normal_factor;
        ele_factor *= static_cast<T>(input_shape[dim_index]);
      }
    }
    T ele_val = input[i];
    if (forward) {
      ele_val /= std::sqrt(ele_factor);
    } else {
      ele_val *= std::sqrt(ele_factor);
    }

    output[i] = ele_val;
  }
  return true;
}
}  // namespace

bool FFTOrthoCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " valid cpu kernel does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int FFTOrthoCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  tensor_shape_ = inputs[kIndex0]->GetShapeVector();
  x_rank_ = SizeToLong(tensor_shape_.size());

  auto dim_opt = inputs[kIndex1]->GetOptionalValueWithCheck<std::vector<int64_t>>();
  if (dim_opt.has_value()) {
    dim_ = dim_opt.value();
    for (size_t i = 0; i < dim_.size(); i++) {
      dim_[i] = dim_[i] < 0 ? x_rank_ + dim_[i] : dim_[i];
    }
  }

  forward_ = inputs[kIndex2]->GetValueWithCheck<bool>();

  return KRET_OK;
}

template <typename T>
bool FFTOrthoCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                        const std::vector<kernel::KernelTensor *> &outputs) {
  auto *input_ptr = reinterpret_cast<T *>(inputs[kIndex0]->device_ptr());
  auto *output_ptr = reinterpret_cast<T *>(outputs[kIndex0]->device_ptr());

  auto ret = memset_s(output_ptr, outputs[kIndex0]->size(), 0, outputs[kIndex0]->size());
  if (ret != EOK) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', output buff memset failed. Error no: " << ret;
    return false;
  }

  Orthogonalize<T>(input_ptr, output_ptr, tensor_shape_, dim_, forward_);
  return true;
}

#define ONE_DIM_CPU_REG(MS_Tin, MS_Tout, T)                         \
  KernelAttr()                                                      \
    .AddInputAttr(MS_Tin)                             /* dout */    \
    .AddOptionalInputAttr(kNumberTypeInt64)           /* axes */    \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeBool) /* forward */ \
    .AddOutputAttr(MS_Tout),                                        \
    &FFTOrthoCpuKernelMod::LaunchKernel<T>

std::vector<std::pair<KernelAttr, FFTOrthoCpuKernelMod::FFTOrthoFunc>> FFTOrthoCpuKernelMod::func_list_ = {
  {ONE_DIM_CPU_REG(kNumberTypeFloat32, kNumberTypeFloat32, float)},
  {ONE_DIM_CPU_REG(kNumberTypeFloat64, kNumberTypeFloat64, double)}};
//  {ONE_DIM_CPU_REG(kNumberTypeComplex64, kNumberTypeComplex64, complex64)},
//  {ONE_DIM_CPU_REG(kNumberTypeComplex128, kNumberTypeComplex128, complex128)}};

std::vector<KernelAttr> FFTOrthoCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, FFTOrthoFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FFTOrtho, FFTOrthoCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
