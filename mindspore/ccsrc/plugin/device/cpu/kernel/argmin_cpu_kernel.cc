/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/kernel/argmin_cpu_kernel.h"
#include <string>
#include <algorithm>
#include "mindspore/core/ops/arg_min.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kArgMinInputsNum = 1;
constexpr size_t kArgMinOutputsNum = 1;
constexpr char kKernelName[] = "Argmin";

int64_t get_element_num(const std::vector<int64_t> &shape) {
  int64_t size = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }
  return size;
}
}  // namespace

template <typename T, typename S>
bool check_validation(const std::vector<int64_t> &shape, const int64_t num_before_axis, const int64_t num_after_axis,
                      const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kArgMinInputsNum, kKernelName);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kArgMinOutputsNum, kKernelName);
  int64_t inputs_size = get_element_num(shape) * static_cast<int64_t>(sizeof(T));
  int64_t output_num = num_before_axis * num_after_axis;
  int64_t output_size = output_num * static_cast<int64_t>(sizeof(S));
  if (static_cast<int64_t>(inputs[0]->size) != inputs_size) {
    MS_LOG(EXCEPTION) << "For '" << kKernelName << "', the memory size of 'input_x' must be equal to " << inputs_size
                      << ", but got the memory size is " << inputs[0]->size;
  }
  if (static_cast<int64_t>(outputs[0]->size) != output_size) {
    MS_LOG(EXCEPTION) << "For '" << kKernelName << "', the memory size of output must be equal to " << output_size
                      << ", but got the memory size is " << outputs[0]->size;
  }
  return true;
}

template <typename T, typename S>
bool ArgminCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &workspace,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  (void)check_validation<T, S>(shape_, num_before_axis_, num_after_axis_, inputs, outputs);

  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output = reinterpret_cast<S *>(outputs[0]->addr);

  auto task = [&](size_t start, size_t end) {
    auto num_after_axis = LongToSize(num_after_axis_);
    auto dim_axis = LongToSize(dim_axis_);
    for (size_t pos = start; pos < end; pos++) {
      size_t i = pos / num_after_axis;
      size_t j = pos % num_after_axis;
      size_t src_index_j = i * dim_axis * num_after_axis + j;

      T min_value = input[src_index_j];
      S min_index = 0;
      for (size_t k = 0; k < dim_axis; k++) {
        auto src_index_k = k * num_after_axis + src_index_j;
        if (input[src_index_k] < min_value) {
          min_value = input[src_index_k];
          min_index = static_cast<S>(k);
        }
      }
      auto dst_index = i * num_after_axis + j;
      output[dst_index] = min_index;
    }
  };
  ParallelLaunchAutoSearch(task, LongToSize(num_before_axis_ * num_after_axis_), this, &parallel_search_info_);

  return true;
}
bool ArgminCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ArgMin>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast ArgMin ops failed!";
    return false;
  }
  if (inputs.size() < 1) {
    MS_LOG(ERROR) << "Argmin input size can not less than 1!";
    return false;
  }

  kernel_name_ = kernel_ptr->name();
  axis_ = kernel_ptr->get_axis();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int ArgminCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = 0;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost)) != 0) {
    return ret;
  }
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kArgMinInputsNum, kernel_name_);
  shape_ = inputs[0]->GetShapeVector();
  size_t shape_len = shape_.size();
  if (shape_len == 0) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "', the dimension of 'input_x' must be at least 1, but got 0.";
    return KRET_RESIZE_FAILED;
  }
  if (CHECK_SHAPE_NULL(shape_, kernel_name_, "input")) {
    return KRET_RESIZE_FAILED;
  }
  int64_t dim = static_cast<int64_t>(shape_len);
  if (axis_ >= dim || axis_ < -dim) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the value of 'axis' must be in the range of [" << -dim
                             << "," << dim << "), but got 'axis': " << axis_ << ".";
  }
  axis_ += SizeToLong(shape_len);
  if (axis_ < 0) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "', the 'axis' must be in range [0, " << (shape_len - 1)
                    << "], but got " << axis_;
    return KRET_RESIZE_FAILED;
  }
  axis_ = axis_ % SizeToLong(shape_len);
  num_before_axis_ = 1;
  num_after_axis_ = 1;
  for (size_t index = 0; index < shape_len; index++) {
    if (SizeToLong(index) < axis_) {
      num_before_axis_ *= shape_[index];
    } else if (SizeToLong(index) > axis_) {
      num_after_axis_ *= shape_[index];
    }
  }
  dim_axis_ = shape_[LongToSize(axis_)];
  return KRET_OK;
}

std::vector<std::pair<KernelAttr, ArgminCpuKernelMod::ArgminFunc>> ArgminCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
   &ArgminCpuKernelMod::LaunchKernel<int8_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
   &ArgminCpuKernelMod::LaunchKernel<int16_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &ArgminCpuKernelMod::LaunchKernel<int32_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
   &ArgminCpuKernelMod::LaunchKernel<int64_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
   &ArgminCpuKernelMod::LaunchKernel<uint8_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32),
   &ArgminCpuKernelMod::LaunchKernel<uint16_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32),
   &ArgminCpuKernelMod::LaunchKernel<uint32_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32),
   &ArgminCpuKernelMod::LaunchKernel<uint64_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
   &ArgminCpuKernelMod::LaunchKernel<float16, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
   &ArgminCpuKernelMod::LaunchKernel<float, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
   &ArgminCpuKernelMod::LaunchKernel<double, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
   &ArgminCpuKernelMod::LaunchKernel<int8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
   &ArgminCpuKernelMod::LaunchKernel<int16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   &ArgminCpuKernelMod::LaunchKernel<int32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &ArgminCpuKernelMod::LaunchKernel<int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
   &ArgminCpuKernelMod::LaunchKernel<uint8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64),
   &ArgminCpuKernelMod::LaunchKernel<uint16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64),
   &ArgminCpuKernelMod::LaunchKernel<uint32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64),
   &ArgminCpuKernelMod::LaunchKernel<uint64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
   &ArgminCpuKernelMod::LaunchKernel<float16, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
   &ArgminCpuKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
   &ArgminCpuKernelMod::LaunchKernel<double, int64_t>}};

std::vector<KernelAttr> ArgminCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ArgminFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Argmin, ArgminCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
