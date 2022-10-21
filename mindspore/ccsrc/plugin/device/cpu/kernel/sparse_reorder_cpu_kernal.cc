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

#include "plugin/device/cpu/kernel/sparse_reorder_cpu_kernal.h"
#include <algorithm>
#include <string>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kIndicesShapeSize = 2;
constexpr size_t kSparseReorderInputsNum = 3;
constexpr size_t kSparseReorderOutputsNum = 2;
}  // namespace

bool SparseReorderCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "SparseReorder does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int SparseReorderCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto indices_shape = inputs.at(kIndex0)->GetShapeVector();
  if (indices_shape.size() != kIndicesShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'indices' must be " << kIndicesShapeSize
                      << "-D, but got " << indices_shape.size() << "-D";
  }
  auto values_shape = inputs.at(kIndex1)->GetShapeVector();
  if (values_shape.size() != 1 || values_shape[0] != indices_shape[0]) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', it requires 'values' must be a 1-D Tensor and the first dimension length "
                         "must be equal to the first dimension length of 'indices', but got 'values' shape: "
                      << Vector2Str(values_shape) << " and 'indices' shape: " << Vector2Str(indices_shape);
  }
  auto shape_shape = inputs.at(kIndex2)->GetShapeVector();
  if (shape_shape.size() != 1 || shape_shape[0] != indices_shape[1]) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', it requires 'shape' must be a 1-D Tensor and the first dimension length "
                         "must be equal to the second dimension length of 'indices', but got 'shape' shape: "
                      << Vector2Str(shape_shape) << " and 'indices' shape: " << Vector2Str(indices_shape);
  }

  indices_shape_ = Convert2SizeT(indices_shape);
  return KRET_OK;
}

template <typename I, typename T>
bool SparseReorderCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                             const std::vector<kernel::AddressPtr> & /* workspace */,
                                             const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseReorderInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseReorderOutputsNum, kernel_name_);

  if (outputs[0]->size == 0) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "', indices memory size must be greater than 0, but got 0.";
    return true;
  }
  if (outputs[1]->size == 0) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "', values memory size must be greater than 0, but got 0.";
    return true;
  }
  auto ret_indices = memset_s(outputs[0]->addr, outputs[0]->size, 0, outputs[0]->size);
  if (ret_indices != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset y_indices failed. Error no: " << ret_indices;
  }
  auto ret_values = memset_s(outputs[1]->addr, outputs[1]->size, 0, outputs[1]->size);
  if (ret_values != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset y_values failed. Error no: " << ret_values;
  }

  SparseGradient<I, T> sparse_tensor_;
  const auto *indices_addr = reinterpret_cast<I *>(inputs[0]->addr);
  const auto *values_addr = reinterpret_cast<T *>(inputs[1]->addr);
  const auto *shape_addr = reinterpret_cast<I *>(inputs[2]->addr);
  auto *y_indices_addr = reinterpret_cast<I *>(outputs[0]->addr);
  auto *y_values_addr = reinterpret_cast<T *>(outputs[1]->addr);
  const size_t indices_length = inputs[0]->size / sizeof(I);
  const size_t values_length = inputs[1]->size / sizeof(T);
  size_t rank = indices_shape_[1];
  for (size_t i = 0; i < indices_shape_[0]; ++i) {
    if (i >= values_length) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the index of 'values' out of bounds.";
    }
    y_values_addr[i] = values_addr[i];
    for (size_t j = 0; j < rank; j++) {
      if (i * rank + j >= indices_length) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the index of 'indices' out of bounds.";
      }
      int index = indices_addr[i * rank + j];
      if (index >= (shape_addr[j]) || index < 0) {
        string outboundary_error = "For '" + kernel_name_ + "', shape=(";
        for (size_t k = 0; k < rank; ++k) {
          outboundary_error += std::to_string(shape_addr[k]);
          if (k != rank - 1) {
            outboundary_error += ", ";
          }
        }
        outboundary_error +=
          "), the index range on the " + std::to_string(j) + "-th dimension is [0, " + std::to_string(shape_addr[j]);
        outboundary_error += "), but the obtained index value is " + std::to_string(index) + ".";
        MS_EXCEPTION(ValueError) << outboundary_error;
      }
      y_indices_addr[i * rank + j] = indices_addr[i * rank + j];
    }
  }
  sparse_tensor_.indices_ = y_indices_addr;
  sparse_tensor_.value_ = y_values_addr;
  sparse_tensor_.dims_size_ = rank;
  std::vector<int64_t> reorder(indices_shape_[0]);
  std::iota(reorder.begin(), reorder.end(), 0);
  // Sort to get order of indices
  std::sort(reorder.begin(), reorder.end(), sparse_tensor_);
  std::vector<size_t> permutation(reorder.size());
  for (std::size_t n = 0; n < reorder.size(); ++n) {
    permutation[reorder[n]] = n;
  }
  for (std::size_t n = 0; n + 1 < permutation.size(); ++n) {
    while (n != permutation[n]) {
      std::size_t r = permutation[n];
      std::swap_ranges(y_indices_addr + n * indices_shape_[1], y_indices_addr + (n + 1) * indices_shape_[1],
                       y_indices_addr + r * indices_shape_[1]);
      std::swap(y_values_addr[n], y_values_addr[r]);
      std::swap(permutation[n], permutation[r]);
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, SparseReorderCpuKernelMod::SparseReorderFunc>> SparseReorderCpuKernelMod::func_list_ =
  {{KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeBool)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeBool),
    &SparseReorderCpuKernelMod::LaunchKernel<int64_t, bool>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt8)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt8),
    &SparseReorderCpuKernelMod::LaunchKernel<int64_t, int8_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt16)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt16),
    &SparseReorderCpuKernelMod::LaunchKernel<int64_t, int16_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt32),
    &SparseReorderCpuKernelMod::LaunchKernel<int64_t, int32_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64),
    &SparseReorderCpuKernelMod::LaunchKernel<int64_t, int64_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeUInt8)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeUInt8),
    &SparseReorderCpuKernelMod::LaunchKernel<int64_t, uint8_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeUInt16)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeUInt16),
    &SparseReorderCpuKernelMod::LaunchKernel<int64_t, uint16_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat16),
    &SparseReorderCpuKernelMod::LaunchKernel<int64_t, float16>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat32),
    &SparseReorderCpuKernelMod::LaunchKernel<int64_t, float>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeFloat64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat64),
    &SparseReorderCpuKernelMod::LaunchKernel<int64_t, double>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeComplex64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeComplex64),
    &SparseReorderCpuKernelMod::LaunchKernel<int64_t, float_complex>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeComplex128)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeComplex128),
    &SparseReorderCpuKernelMod::LaunchKernel<int64_t, double_complex>}};

std::vector<KernelAttr> SparseReorderCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseReorderFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseReorder, SparseReorderCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
