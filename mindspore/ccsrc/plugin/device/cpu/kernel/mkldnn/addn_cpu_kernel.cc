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

#include "plugin/device/cpu/kernel/mkldnn/addn_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/add_fp32.h"
#include "plugin/device/cpu/kernel/nnacl/errorcode.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"

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
    MS_LOG(EXCEPTION) << "Add failed.";
  }
}

template <typename T>
void AddT(const T *in0, const T *in1, T *out, int start, int end) {
  for (int index = start; index < end; index++) {
    out[index] = in0[index] + in1[index];
  }
}
}  // namespace

bool AddNCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();
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
  dtype_ = inputs[kIndex0]->GetDtype();
  return true;
}

int AddNCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs,
                             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto src0_shape = inputs[kIndex0]->GetDeviceShapeAdaptively();
  auto src1_shape = inputs[kIndex1]->GetDeviceShapeAdaptively();
  auto dst_shape = outputs[kIndex0]->GetDeviceShapeAdaptively();
  dnnl::memory::desc src0_mem_desc = GetDefaultMemDesc(src0_shape);
  dnnl::memory::desc src1_mem_desc = GetDefaultMemDesc(src1_shape);
  dnnl::memory::desc dst_mem_desc = GetDefaultMemDesc(dst_shape);
  auto desc = CreateDesc<dnnl::binary::desc>(dnnl::algorithm::binary_add, src0_mem_desc, src1_mem_desc, dst_mem_desc);
  auto prim_desc = CreateDesc<dnnl::binary::primitive_desc>(desc, engine_);
  primitive_ = CreatePrimitive<dnnl::binary>(prim_desc);
  AddArgument(DNNL_ARG_SRC_0, src0_mem_desc);
  AddArgument(DNNL_ARG_SRC_1, src1_mem_desc);
  AddArgument(DNNL_ARG_DST, dst_mem_desc);
  return KRET_OK;
}

template <typename T>
bool AddNCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num_, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kAddNOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat32) {
    SetArgumentHandle(DNNL_ARG_SRC_0, inputs[0]->addr);
    SetArgumentHandle(DNNL_ARG_SRC_1, inputs[1]->addr);
    SetArgumentHandle(DNNL_ARG_DST, outputs[0]->addr);
    ExecutePrimitive();
    for (size_t index = 2; index < input_num_; ++index) {
      SetArgumentHandle(DNNL_ARG_SRC_0, outputs[0]->addr);
      SetArgumentHandle(DNNL_ARG_SRC_1, inputs[index]->addr);
      SetArgumentHandle(DNNL_ARG_DST, outputs[0]->addr);
      ExecutePrimitive();
    }
  } else if (dtype_ == kNumberTypeInt32) {
    size_t elements_num = outputs[0]->size / sizeof(int);
    const auto input_0 = reinterpret_cast<int *>(inputs[0]->addr);
    const auto input_1 = reinterpret_cast<int *>(inputs[1]->addr);
    auto output = reinterpret_cast<int *>(outputs[0]->addr);
    auto task_0 = std::bind(AddInt, input_0, input_1, output, std::placeholders::_1, std::placeholders::_2);
    ParallelLaunchAutoSearch(task_0, elements_num, this, &parallel_search_info_);
    for (size_t index = 2; index < input_num_; ++index) {
      const auto input = reinterpret_cast<int *>(inputs[index]->addr);
      auto task = std::bind(AddInt, input, output, output, std::placeholders::_1, std::placeholders::_2);
      ParallelLaunchAutoSearch(task, elements_num, this, &parallel_search_info_);
    }
  } else {
    size_t elements_num = outputs[0]->size / sizeof(T);
    const auto input_0 = reinterpret_cast<T *>(inputs[0]->addr);
    const auto input_1 = reinterpret_cast<T *>(inputs[1]->addr);
    auto output = reinterpret_cast<T *>(outputs[0]->addr);
    auto task_0 = std::bind(AddT<T>, input_0, input_1, output, std::placeholders::_1, std::placeholders::_2);
    ParallelLaunchAutoSearch(task_0, elements_num, this, &parallel_search_info_);
    for (size_t index = 2; index < input_num_; ++index) {
      const auto input = reinterpret_cast<T *>(inputs[index]->addr);
      auto task = std::bind(AddT<T>, input, output, output, std::placeholders::_1, std::placeholders::_2);
      ParallelLaunchAutoSearch(task, elements_num, this, &parallel_search_info_);
    }
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
