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

#include "plugin/device/cpu/kernel/gather_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "nnacl/gather_parameter.h"
#include "nnacl/base/gather_base.h"
#include "include/common/thread_pool.h"
#include "mindspore/core/ops/gather.h"

namespace mindspore {
namespace kernel {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
constexpr size_t kGatherInputsNum = 3;
constexpr size_t kGatherOutputsNum = 1;
constexpr size_t kGatherInputParamsMaxDim = 7;
}  // namespace
bool GatherCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Gather>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  batch_dims_ = kernel_ptr->get_batch_dims();

  size_t input_num = inputs.size();
  if (input_num != kGatherInputsNum) {
    MS_LOG(EXCEPTION) << "Argument number is " << input_num << ", but GatherCPUKernel needs 2.";
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  input_type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  indices_type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).dtype);
  axis_type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex2).dtype);
  return true;
}

int GatherCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  ResetResource();
  input_shape_ = inputs[kIndexZero]->GetShapeVector();
  indices_shape_ = inputs[kIndexOne]->GetShapeVector();
  output_shape_ = outputs[kIndexZero]->GetShapeVector();
  if (IsDynamic(input_shape_) || IsDynamic(indices_shape_) || IsDynamic(output_shape_)) {
    return KRET_UNKNOWN_SHAPE;
  }
  if (batch_dims_ < 0) {
    batch_dims_ += SizeToLong(indices_shape_.size());
  }
  is_null_input_ = input_shape_.empty() || indices_shape_.empty() || output_shape_.empty();
  if (is_null_input_) {
    InitSizeLists();
    return KRET_OK;
  }
  if (input_shape_.size() > kGatherInputParamsMaxDim) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'input_params' should be "
                      << kGatherInputParamsMaxDim << "D or lower, but got " << input_shape_.size() << ".";
  }
  InitSizeLists();
  return KRET_OK;
}

template <typename T, typename S>
bool GatherCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGatherOutputsNum, kernel_name_);
  const auto *input_tensor = reinterpret_cast<int8_t *>(inputs[0]->addr);
  const auto *indices_data = reinterpret_cast<int32_t *>(inputs[1]->addr);
  auto *output_addr = reinterpret_cast<int8_t *>(outputs[0]->addr);
  S axis_temp = static_cast<S *>(inputs[kIndex2]->addr)[0];
  axis_ = static_cast<int64_t>(axis_temp);

  int dims = SizeToInt(input_shape_.size());
  if (axis_ < -dims || axis_ >= dims) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'axis' should be in the range [-" << dims << ", " << dims
                  << "), but got " << axis_ << ".";
    return false;
  } else if (axis_ < 0) {
    axis_ = axis_ + dims;
  }

  size_t batch_size = 1;
  size_t outer_size = 1;
  size_t indices_element_size = 1;
  size_t inner_size = 1;
  auto axis = LongToSize(axis_);
  auto batch_dims = LongToSize(batch_dims_);
  for (size_t i = 0; i < batch_dims; i++) {
    batch_size *= LongToSize(input_shape_.at(i));
  }
  for (size_t i = batch_dims; i < axis; ++i) {
    outer_size *= LongToSize(input_shape_.at(i));
  }
  for (size_t i = axis + 1; i < input_shape_.size(); ++i) {
    inner_size *= LongToSize(input_shape_.at(i));
  }
  for (size_t i = batch_dims; i < indices_shape_.size(); i++) {
    indices_element_size *= LongToSize(indices_shape_.at(i));
  }

  auto limit = LongToSize(input_shape_.at(axis));
  size_t byte_inner_size = inner_size * sizeof(T);
  size_t byte_out_stride = indices_element_size * byte_inner_size;
  for (size_t i = 0; i < batch_size; i++) {
    auto output_ptr = output_addr + i * outer_size * byte_out_stride;
    auto input_ptr = input_tensor + i * outer_size * byte_inner_size * limit;
    auto indice_ptr = indices_data + i * indices_element_size;

    auto task = [&](size_t start, size_t end) {
      int count = SizeToInt(end - start);
      const int8_t *in = input_ptr + start * limit * byte_inner_size;
      int8_t *out = output_ptr + start * byte_out_stride;
      int ret = Gather(in, count, byte_inner_size, limit, indice_ptr, indices_element_size, out, byte_out_stride);
      if (ret != 0) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', error_code[" << ret << "]";
      }
    };
    ParallelLaunchAutoSearch(task, outer_size, this, &parallel_search_info_);
  }

  return true;
}

#define GATHER_CPU_REG(MS_T, MS_S, MS_G, T, S)                                               \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_S).AddInputAttr(MS_G).AddOutputAttr(MS_T), \
    &GatherCpuKernelMod::LaunchKernel<T, S>

std::vector<std::pair<KernelAttr, GatherCpuKernelMod::GatherFunc>> GatherCpuKernelMod::func_list_ = {
  {GATHER_CPU_REG(kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeInt32, uint8_t, int32_t)},
  {GATHER_CPU_REG(kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeInt64, uint8_t, int64_t)},
  {GATHER_CPU_REG(kNumberTypeUInt16, kNumberTypeInt32, kNumberTypeInt32, uint16_t, int32_t)},
  {GATHER_CPU_REG(kNumberTypeUInt16, kNumberTypeInt32, kNumberTypeInt64, uint16_t, int64_t)},
  {GATHER_CPU_REG(kNumberTypeUInt32, kNumberTypeInt32, kNumberTypeInt32, uint32_t, int32_t)},
  {GATHER_CPU_REG(kNumberTypeUInt32, kNumberTypeInt32, kNumberTypeInt64, uint32_t, int64_t)},
  {GATHER_CPU_REG(kNumberTypeUInt64, kNumberTypeInt32, kNumberTypeInt32, uint64_t, int32_t)},
  {GATHER_CPU_REG(kNumberTypeUInt64, kNumberTypeInt32, kNumberTypeInt64, uint64_t, int64_t)},
  {GATHER_CPU_REG(kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt32, int8_t, int32_t)},
  {GATHER_CPU_REG(kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt64, int8_t, int64_t)},
  {GATHER_CPU_REG(kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt32, int16_t, int32_t)},
  {GATHER_CPU_REG(kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64, int16_t, int64_t)},
  {GATHER_CPU_REG(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int32_t, int32_t)},
  {GATHER_CPU_REG(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt64, int32_t, int64_t)},
  {GATHER_CPU_REG(kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32, int64_t, int32_t)},
  {GATHER_CPU_REG(kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt64, int64_t, int64_t)},
  {GATHER_CPU_REG(kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeInt32, float16, int32_t)},
  {GATHER_CPU_REG(kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeInt64, float16, int64_t)},
  {GATHER_CPU_REG(kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeInt32, float, int32_t)},
  {GATHER_CPU_REG(kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeInt64, float, int64_t)},
  {GATHER_CPU_REG(kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeInt32, double, int32_t)},
  {GATHER_CPU_REG(kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeInt64, double, int64_t)},
  {GATHER_CPU_REG(kNumberTypeBool, kNumberTypeInt32, kNumberTypeInt32, bool, int32_t)},
  {GATHER_CPU_REG(kNumberTypeBool, kNumberTypeInt32, kNumberTypeInt64, bool, int64_t)},
  {GATHER_CPU_REG(kNumberTypeComplex64, kNumberTypeInt32, kNumberTypeInt32, complex64, int32_t)},
  {GATHER_CPU_REG(kNumberTypeComplex64, kNumberTypeInt32, kNumberTypeInt64, complex64, int64_t)},
  {GATHER_CPU_REG(kNumberTypeComplex128, kNumberTypeInt32, kNumberTypeInt32, complex128, int32_t)},
  {GATHER_CPU_REG(kNumberTypeComplex128, kNumberTypeInt32, kNumberTypeInt64, complex128, int64_t)}};

std::vector<KernelAttr> GatherCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, GatherFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Gather, GatherCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
