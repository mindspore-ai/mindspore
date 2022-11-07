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

const uint32_t kInput_reverse_index_map = 0;
const uint32_t kInput_grad_values = 1;

const uint32_t kOutput_y_values = 0;
const uint32_t kOutput_y_default_value = 1;
constexpr char kKernelName[] = "SparseFillEmptyRows";

#define EIGEN_SHAPE_CAST(INPUT) static_cast<Eigen::DenseIndex>(AnfAlgo::GetInputDeviceShape(node_ptr, INPUT)[0])

#define SPARSE_FILL_EMPTY_ROWS_GRAD_COMPUTE_CASE(DTYPE, TYPE) \
  case (DTYPE): {                                             \
    ret = LaunchKernel<TYPE>(inputs, outputs);                \
    break;                                                    \
  }
}  // namespace

void SparseFillEmptyRowsGradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  node_ptr = kernel_node;
  MS_EXCEPTION_IF_NULL(node_ptr);
  output_y_values_type_ = AnfAlgo::GetOutputDeviceDataType(kernel_node, 0);
  output_y_default_value_type_ = AnfAlgo::GetOutputDeviceDataType(kernel_node, 1);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(node_ptr);
  CHECK_KERNEL_INPUTS_NUM(input_num, kSparseFillEmptyRowsGradInputsNum, kernel_name_);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(node_ptr);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kSparseFillEmptyRowsGradOutputsNum, kernel_name_);
  const auto reverse_index_map_shape = AnfAlgo::GetInputDeviceShape(node_ptr, 0);
  const auto grad_values_shape = AnfAlgo::GetInputDeviceShape(node_ptr, 1);
  if (reverse_index_map_shape.size() != kReverseIndexMapSizeNum && reverse_index_map_shape[0] > grad_values_shape[0]) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', it requires 'reverse_index_map' must be a 1-D Tensor and the first dimension length "
                         "must be smalll or equal to the first dimension length of 'values' "
                      << Vector2Str(reverse_index_map_shape);
  }
  if (grad_values_shape.size() != kGradValuesSizeNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it requires 'grad_values' must be a 1-D Tensor "
                      << Vector2Str(grad_values_shape);
  }
}

template <typename T>
bool SparseFillEmptyRowsGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseFillEmptyRowsGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseFillEmptyRowsGradOutputsNum, kernel_name_);
  auto reverse_index_map_ptr = reinterpret_cast<int64_t *>(inputs[0]->addr);
  auto grad_values_ptr = reinterpret_cast<T *>(inputs[1]->addr);
  const auto reverse_index_map_shape = AnfAlgo::GetInputDeviceShape(node_ptr, 0);
  const auto grad_values_shape = AnfAlgo::GetInputDeviceShape(node_ptr, 1);
  const int64_t N = reverse_index_map_shape[0];
  const int64_t N_full = grad_values_shape[0];
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
  ShapeVector output_y_values_shape;
  ShapeVector output_y_default_value_shape = {};
  output_y_values_shape.push_back(N);
  common::AnfAlgo::SetOutputInferTypeAndShape({output_y_values_type_, output_y_default_value_type_},
                                              {output_y_values_shape, output_y_default_value_shape},
                                              cnode_ptr_.lock().get());
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
  auto data_type = AnfAlgo::GetInputDeviceDataType(node_ptr, kInput_grad_values);
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
