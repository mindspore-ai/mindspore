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

#include "plugin/device/cpu/kernel/reversev2_cpu_kernel.h"
#include <map>
#include <algorithm>
#include <utility>
#include "Eigen/Core"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/nnacl/errorcode.h"
#include "utils/tensor_iterator.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kReverseV2InputsNum = 1;
constexpr size_t kReverseV2OutputsNum = 1;
constexpr int64_t kInputDim = 9;

std::vector<int64_t> idx2coord(int idx, const std::vector<int64_t> &accum_dim) {
  std::vector<int64_t> coord(accum_dim.size());
  for (size_t i = 0; i < coord.size(); ++i) {
    coord[i] = idx / accum_dim[i];
    idx -= coord[i] * accum_dim[i];
  }
  return coord;
}

inline int64_t calc_target_idx(const std::vector<int64_t> &coord, const std::unordered_set<int64_t> &dims,
                               const std::vector<int64_t> &shape, const std::vector<int64_t> &accum_dim) {
  int64_t idx = 0;
  for (size_t i = 0; i < coord.size(); ++i) {
    if (dims.count(i) != 0) {
      idx += accum_dim[i] * (shape[i] - coord[i] - 1);
    } else {
      idx += accum_dim[i] * coord[i];
    }
  }
  return idx;
}
}  // namespace

bool ReverseV2CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For ReverseV2, ReverseV2 type should be uint8_t, uint16_t, int8_t, int16_t, "
                         "int32_t, int64_t, float16, float, double, complex64, complex128, but got data type: "
                      << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int ReverseV2CpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[kIndex0]->GetShapeVector();
  input_dims_ = SizeToLong(input_shape_.size());
  if (base_operator->HasAttr("axis")) {
    auto axis = GetValue<std::vector<int64_t>>(base_operator->GetAttr("axis"));
    (void)std::transform(axis.begin(), axis.end(), std::inserter(axis_, axis_.begin()),
                         [input_dims = input_dims_](int64_t x) { return x >= 0 ? x : input_dims + x; });
  }
  axis_dims_ = SizeToLong(axis_.size());
  if (input_dims_ >= kInputDim) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input should less than " << kInputDim
                      << ", but got " << input_dims_;
  }
  return KRET_OK;
}

template <typename T>
bool ReverseV2CpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kReverseV2InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kReverseV2OutputsNum, kernel_name_);

  auto input_data = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_data = reinterpret_cast<T *>(outputs[0]->addr);
  int64_t num_element = 1;
  for (int64_t i = 0; i < input_dims_; ++i) {
    num_element *= input_shape_[i];
  }
  if (axis_dims_ == 0) {
    std::copy(input_data, input_data + num_element, output_data);
    return true;
  }

  std::vector<int64_t> accum_dim(input_shape_.size());
  accum_dim.back() = 1;
  for (size_t i = input_shape_.size() - 1; i > 0; --i) {
    accum_dim[i - 1] = accum_dim[i] * input_shape_[i];
  }

  auto sharder_reverse = [&](int64_t start, int64_t end) {
    std::vector<int64_t> cur_coord = idx2coord(start, accum_dim);
    auto coord_iter = TensorIterator(input_shape_, cur_coord);
    for (int i = start; i < end; ++i) {
      auto target_idx = calc_target_idx(*coord_iter, axis_, input_shape_, accum_dim);
      output_data[target_idx] = input_data[i];
      ++coord_iter;
    }
  };

  CPUKernelUtils::ParallelForAutoSearch(sharder_reverse, num_element, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, ReverseV2CpuKernelMod::ReverseV2Func>> ReverseV2CpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
   &ReverseV2CpuKernelMod::LaunchKernel<bool>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &ReverseV2CpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
   &ReverseV2CpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
   &ReverseV2CpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
   &ReverseV2CpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &ReverseV2CpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &ReverseV2CpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &ReverseV2CpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &ReverseV2CpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &ReverseV2CpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &ReverseV2CpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &ReverseV2CpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
   &ReverseV2CpuKernelMod::LaunchKernel<complex64>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
   &ReverseV2CpuKernelMod::LaunchKernel<complex128>}};

std::vector<KernelAttr> ReverseV2CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ReverseV2Func> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ReverseV2, ReverseV2CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
