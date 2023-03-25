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

#include "plugin/device/cpu/kernel/gathernd_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
#define MAX_INT ((static_cast<unsigned int>(-1)) >> 1)

constexpr auto kGatherNd = "GatherNd";
constexpr size_t kGatherNdInputsNum = 2;
constexpr size_t kGatherNdOutputsNum = 1;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

bool GatherNdCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();
  dtype_ = inputs[0]->GetDtype();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, GatherNdFunc> &pair) { return pair.first; });
  auto [is_match, index] = MatchKernelAttr(kernel_attr, support_list);
  if (!is_match) {
    MS_LOG(EXCEPTION) << "GatherNd does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}
int GatherNdCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  indices_shapes_.clear();
  dims_.clear();
  batch_indices_.clear();

  input_shapes_ = inputs[0]->GetShapeVector();
  indices_shapes_ = inputs[1]->GetShapeVector();
  // make a scalar to tensor whose shape is (1,)
  if (indices_shapes_.size() == 0) {
    indices_shapes_.emplace_back(1);
  }
  // Reshape()
  size_t dim_of_indices = 1;
  for (size_t i = 0; i < indices_shapes_.size() - IntToSize(1); ++i) {
    dim_of_indices *= LongToSize(indices_shapes_[i]);
  }

  size_t dim_after_indices = 1;
  size_t dim_indices_last = LongToSize(indices_shapes_[indices_shapes_.size() - IntToSize(1)]);
  for (size_t i = dim_indices_last; i < input_shapes_.size(); i++) {
    dim_after_indices *= LongToSize(input_shapes_[i]);
  }

  (void)dims_.emplace_back(dim_of_indices);
  (void)dims_.emplace_back(dim_after_indices);
  (void)dims_.emplace_back(dim_indices_last);
  batch_indices_.resize(dim_indices_last, 0);

  if (dim_indices_last > 0) {
    batch_indices_[dim_indices_last - 1] = dims_[1];
  }

  for (int i = static_cast<int>(dim_indices_last) - 1; i > 0; --i) {
    batch_indices_[i - 1] = batch_indices_[i] * LongToInt(input_shapes_[i]);
  }
  return ret;
}

template <typename S, typename T>
bool GatherNdCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kGatherNdInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGatherNdOutputsNum, kernel_name_);
  const auto *input_addr = static_cast<T *>(inputs[0]->addr);
  const auto *indices_addr = static_cast<S *>(inputs[1]->addr);
  auto output_addr = static_cast<T *>(outputs[0]->addr);

  size_t output_dim0 = dims_[0];
  size_t output_dim1 = dims_[1];
  size_t indices_dim1 = dims_[2];

  size_t num = output_dim0 * output_dim1;

  for (size_t write_index = 0; write_index < num; write_index++) {
    size_t i = write_index / output_dim1 % output_dim0;
    size_t j = write_index % output_dim1;

    int read_index = 0;
    for (size_t k = 0; k < indices_dim1; k++) {
      size_t ind = indices_dim1 * i + k;
      int indices_i = indices_addr[ind];
      if (indices_i >= input_shapes_[k] || indices_i < 0) {
        std::vector<S> error_indice(indices_dim1);
        auto ret = memcpy_s(error_indice.data(), sizeof(S) * indices_dim1, indices_addr + indices_dim1 * i,
                            sizeof(S) * indices_dim1);
        if (ret != EOK) {
          MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy_s failed, Error no: " << ret;
        }
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the indices[" << i << "]: " << error_indice
                          << ", does not index into input_shape: " << input_shapes_ << ".";
      }
      read_index += indices_i * batch_indices_[k];
    }
    read_index += SizeToInt(j);
    output_addr[write_index] = input_addr[read_index];
  }
  return true;
}

std::vector<std::pair<KernelAttr, GatherNdCpuKernelMod::GatherNdFunc>> GatherNdCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
   &GatherNdCpuKernelMod::LaunchKernel<int32_t, int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
   &GatherNdCpuKernelMod::LaunchKernel<int32_t, int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &GatherNdCpuKernelMod::LaunchKernel<int32_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   &GatherNdCpuKernelMod::LaunchKernel<int32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
   &GatherNdCpuKernelMod::LaunchKernel<int32_t, uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
   &GatherNdCpuKernelMod::LaunchKernel<int32_t, uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
   &GatherNdCpuKernelMod::LaunchKernel<int32_t, uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
   &GatherNdCpuKernelMod::LaunchKernel<int32_t, uint64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
   &GatherNdCpuKernelMod::LaunchKernel<int32_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
   &GatherNdCpuKernelMod::LaunchKernel<int32_t, double>},
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
   &GatherNdCpuKernelMod::LaunchKernel<int32_t, bool>},
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
   &GatherNdCpuKernelMod::LaunchKernel<int64_t, bool>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex64),
   &GatherNdCpuKernelMod::LaunchKernel<int32_t, complex64>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex128),
   &GatherNdCpuKernelMod::LaunchKernel<int32_t, complex128>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
   &GatherNdCpuKernelMod::LaunchKernel<int64_t, int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
   &GatherNdCpuKernelMod::LaunchKernel<int64_t, int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
   &GatherNdCpuKernelMod::LaunchKernel<int64_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &GatherNdCpuKernelMod::LaunchKernel<int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
   &GatherNdCpuKernelMod::LaunchKernel<int64_t, uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
   &GatherNdCpuKernelMod::LaunchKernel<int64_t, uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
   &GatherNdCpuKernelMod::LaunchKernel<int64_t, uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
   &GatherNdCpuKernelMod::LaunchKernel<int64_t, uint64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
   &GatherNdCpuKernelMod::LaunchKernel<int64_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
   &GatherNdCpuKernelMod::LaunchKernel<int64_t, double>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex64),
   &GatherNdCpuKernelMod::LaunchKernel<int64_t, complex64>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex128),
   &GatherNdCpuKernelMod::LaunchKernel<int64_t, complex128>}};

std::vector<KernelAttr> GatherNdCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, GatherNdFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, GatherNd, GatherNdCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
