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

#include "kernel/cpu/mkldnn/addn_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>

#include "kernel/cpu/nnacl/fp32/add_fp32.h"
#include "kernel/cpu/nnacl/errorcode.h"
#include "include/utils/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

constexpr size_t kAddNInputsMinNum = 2;
constexpr size_t kAddNOutputsNum = 1;

void AddInt(const int *in_0, const int *in_1, int *out, int start, int end) {
  int ret = ElementAddInt(in_0 + start, in_1 + start, out + start, end - start);
  if (ret != NNACL_OK) {
    MS_LOG(EXCEPTION) << "For 'AddN', AddInt failed.";
  }
}

void AddFloat(const float *in_0, const float *in_1, float *out, int start, int end) {
  int ret = ElementAdd(in_0 + start, in_1 + start, out + start, end - start);
  if (ret != NNACL_OK) {
    MS_LOG(EXCEPTION) << "For 'AddN', AddFloat failed.";
  }
}

template <typename T>
void AddT(const T *in0, const T *in1, T *out, int start, int end) {
  for (int index = start; index < end; index++) {
    out[index] = in0[index] + in1[index];
  }
}
}  // namespace

bool AddNCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  input_num_ = inputs.size();
  if (input_num_ < kAddNInputsMinNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input numbers can not less " << kAddNInputsMinNum << ", but got "
                  << input_num_;
    return false;
  }
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kAddNOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

template <typename T>
bool AddNCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                    const std::vector<kernel::KernelTensor *> &,
                                    const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num_, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kAddNOutputsNum, kernel_name_);
  std::function<void(const T *, const T *, T *, int, int)> comput_func;
  if constexpr (std::is_same<T, float>::value) {
    comput_func = AddFloat;
  } else if constexpr (std::is_same<T, int>::value) {
    comput_func = AddInt;
  } else {
    comput_func = AddT<T>;
  }

  size_t elements_num = outputs[0]->size() / sizeof(T);
  const auto input_0 = GetDeviceAddress<T>(inputs, kIndex0);
  const auto input_1 = GetDeviceAddress<T>(inputs, kIndex1);
  auto output = GetDeviceAddress<T>(outputs, kIndex0);
  auto task_0 = std::bind(comput_func, input_0, input_1, output, std::placeholders::_1, std::placeholders::_2);
  ParallelLaunchAutoSearch(task_0, elements_num, this, &parallel_search_info_);
  for (size_t index = 2; index < input_num_; ++index) {
    const auto input = GetDeviceAddress<T>(inputs, index);
    auto task = std::bind(comput_func, input, output, output, std::placeholders::_1, std::placeholders::_2);
    ParallelLaunchAutoSearch(task, elements_num, this, &parallel_search_info_);
  }
  return true;
}

std::vector<std::pair<KernelAttr, AddNCpuKernelMod::AddNFunc>> AddNCpuKernelMod::func_list_ = {
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &AddNCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &AddNCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &AddNCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &AddNCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &AddNCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
   &AddNCpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
   &AddNCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
   &AddNCpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &AddNCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &AddNCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &AddNCpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
   &AddNCpuKernelMod::LaunchKernel<complex64>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
   &AddNCpuKernelMod::LaunchKernel<complex128>}};

std::vector<KernelAttr> AddNCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, AddNFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, AddN, AddNCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
