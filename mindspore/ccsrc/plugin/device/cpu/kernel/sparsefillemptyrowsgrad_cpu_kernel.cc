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

#include "plugin/device/cpu/kernel/sparsefillemptyrowsgrad_cpu_kernel.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <vector>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseFillEmptyRowsGradInputsNum = 2;
constexpr size_t kSparseFillEmptyRowsGradOutputsNum = 2;
constexpr size_t kReverseIndexMapSizeNum = 1;
constexpr size_t kGradValuesSizeNum = 1;

const uint32_t kOutput_y_values = 0;
const uint32_t kOutput_y_default_value = 1;
constexpr char kKernelName[] = "SparseFillEmptyRows";

#define SPARSE_FILL_EMPTY_ROWS_GRAD_COMPUTE_CASE(DTYPE, TYPE) \
  case (DTYPE): {                                             \
    ret = LaunchKernel<TYPE>(inputs, outputs);                \
    break;                                                    \
  }
}  // namespace

bool SparseFillEmptyRowsGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);

  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseFillEmptyRowsGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseFillEmptyRowsGradOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match.first) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
  }

  output_y_values_type_ = inputs[kIndex0]->GetDtype();
  output_y_default_value_type_ = inputs[kIndex1]->GetDtype();

  return true;
}

int SparseFillEmptyRowsGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                const std::vector<KernelTensorPtr> &inputs,
                                                const std::vector<KernelTensorPtr> &outputs,
                                                const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  reverse_index_map_shape_ = inputs[kIndex0]->GetShapeVector();
  grad_values_shape_ = inputs[kIndex1]->GetShapeVector();
  if (reverse_index_map_shape_.size() != kReverseIndexMapSizeNum &&
      reverse_index_map_shape_[0] > grad_values_shape_[0]) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', it requires 'reverse_index_map' must be a 1-D Tensor and the first dimension length "
                         "must be smalll or equal to the first dimension length of 'values' "
                      << Vector2Str(reverse_index_map_shape_);
  }
  if (grad_values_shape_.size() != kGradValuesSizeNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it requires 'grad_values' must be a 1-D Tensor "
                      << Vector2Str(grad_values_shape_);
  }

  return KRET_OK;
}

template <typename T>
bool SparseFillEmptyRowsGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                       const std::vector<kernel::AddressPtr> &outputs) {
  auto reverse_index_map_ptr = reinterpret_cast<int64_t *>(inputs[0]->addr);
  auto grad_values_ptr = reinterpret_cast<T *>(inputs[1]->addr);
  const int64_t N = reverse_index_map_shape_[0];
  const int64_t N_full = grad_values_shape_[0];
  auto y_values_ptr = reinterpret_cast<T *>(outputs[kOutput_y_values]->addr);

  auto ret1 = memset_s(y_values_ptr, N * sizeof(T), 0, N * sizeof(T));
  if (ret1 != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset y_values failed!";
  }
  std::vector<bool> visited(N_full, false);
  auto y_default_value = reinterpret_cast<T *>(outputs[kOutput_y_default_value]->addr);
  *y_default_value = static_cast<T>(0);
  size_t output_size = outputs[0]->size / sizeof(T);
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      size_t reverse_index = reverse_index_map_ptr[i];
      if (static_cast<int64_t>(reverse_index) < 0 && reverse_index >= end) {
        MS_LOG(EXCEPTION) << "For '" << kKernelName << "', Elements in reverse index must be in [0, [" << end
                          << "]) but got [" << reverse_index << "] ";
      }
      y_values_ptr[i] = grad_values_ptr[reverse_index];
      visited[reverse_index] = true;
    }
  };
  ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
  for (int64_t j = 0; j < N_full; ++j) {
    if (!visited[j]) {
      *y_default_value += grad_values_ptr[j];
    }
  }

  return true;
}

std::vector<KernelAttr> SparseFillEmptyRowsGradCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt8)
                                                   .AddOutputAttr(kNumberTypeInt8)
                                                   .AddOutputAttr(kNumberTypeInt8),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeUInt8)
                                                   .AddOutputAttr(kNumberTypeUInt8)
                                                   .AddOutputAttr(kNumberTypeUInt8),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt16)
                                                   .AddOutputAttr(kNumberTypeInt16)
                                                   .AddOutputAttr(kNumberTypeInt16),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeUInt16)
                                                   .AddOutputAttr(kNumberTypeUInt16)
                                                   .AddOutputAttr(kNumberTypeUInt16),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeUInt32)
                                                   .AddOutputAttr(kNumberTypeUInt32)
                                                   .AddOutputAttr(kNumberTypeUInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeUInt64)
                                                   .AddOutputAttr(kNumberTypeUInt64)
                                                   .AddOutputAttr(kNumberTypeUInt64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddOutputAttr(kNumberTypeFloat16)
                                                   .AddOutputAttr(kNumberTypeFloat16),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeBool)
                                                   .AddOutputAttr(kNumberTypeBool)
                                                   .AddOutputAttr(kNumberTypeBool),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddOutputAttr(kNumberTypeFloat64)
                                                   .AddOutputAttr(kNumberTypeFloat64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeComplex64)
                                                   .AddOutputAttr(kNumberTypeComplex64)
                                                   .AddOutputAttr(kNumberTypeComplex64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeComplex128)
                                                   .AddOutputAttr(kNumberTypeComplex128)
                                                   .AddOutputAttr(kNumberTypeComplex128)};
  return support_list;
}

bool SparseFillEmptyRowsGradCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &workspace,
                                                 const std::vector<AddressPtr> &outputs) {
  bool ret = false;
  auto data_type = output_y_default_value_type_;
  switch (data_type) {
    SPARSE_FILL_EMPTY_ROWS_GRAD_COMPUTE_CASE(kNumberTypeInt8, int8_t)
    SPARSE_FILL_EMPTY_ROWS_GRAD_COMPUTE_CASE(kNumberTypeUInt8, uint8_t)
    SPARSE_FILL_EMPTY_ROWS_GRAD_COMPUTE_CASE(kNumberTypeInt16, int16_t)
    SPARSE_FILL_EMPTY_ROWS_GRAD_COMPUTE_CASE(kNumberTypeUInt16, uint16_t)
    SPARSE_FILL_EMPTY_ROWS_GRAD_COMPUTE_CASE(kNumberTypeInt32, int32_t)
    SPARSE_FILL_EMPTY_ROWS_GRAD_COMPUTE_CASE(kNumberTypeInt64, int64_t)
    SPARSE_FILL_EMPTY_ROWS_GRAD_COMPUTE_CASE(kNumberTypeUInt32, uint32_t)
    SPARSE_FILL_EMPTY_ROWS_GRAD_COMPUTE_CASE(kNumberTypeUInt64, uint64_t)
    SPARSE_FILL_EMPTY_ROWS_GRAD_COMPUTE_CASE(kNumberTypeFloat16, Eigen::half)
    SPARSE_FILL_EMPTY_ROWS_GRAD_COMPUTE_CASE(kNumberTypeFloat32, float)
    SPARSE_FILL_EMPTY_ROWS_GRAD_COMPUTE_CASE(kNumberTypeBool, bool)
    SPARSE_FILL_EMPTY_ROWS_GRAD_COMPUTE_CASE(kNumberTypeFloat64, double)
    SPARSE_FILL_EMPTY_ROWS_GRAD_COMPUTE_CASE(kNumberTypeComplex64, std::complex<float>)
    SPARSE_FILL_EMPTY_ROWS_GRAD_COMPUTE_CASE(kNumberTypeComplex128, std::complex<double>)
    default:
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', Unsupported input data type: " << data_type << ".";
  }
  return ret;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseFillEmptyRowsGrad, SparseFillEmptyRowsGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
