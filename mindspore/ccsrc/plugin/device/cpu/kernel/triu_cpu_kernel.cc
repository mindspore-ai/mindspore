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

#include <Eigen/Dense>
#include <algorithm>
#include "plugin/device/cpu/kernel/triu_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kTriuInputsNum = 1;
constexpr size_t kTriuOutputsNum = 1;
constexpr size_t kDim = 2;
}  // namespace

bool TriuCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  input_dtype_ = inputs.at(kIndex0)->GetDtype();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Triu>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);
  diagonal_ = kernel_ptr->get_diagonal();
  return true;
}

int TriuCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs,
                             const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs.at(kIndex0)->GetShapeVector();
  input_dims_ = input_shape_.size();
  if (input_dims_ < kDim) {
    MS_LOG(EXCEPTION)
      << "For Triu, the input tensor's rank must be at least 2 for 'Triu' Op, but input tensor's rank is "
      << input_dims_ << ".";
  }
  return KRET_OK;
}

template <typename T>
bool TriuCpuKernelMod::TriuCompute(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kTriuInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kTriuOutputsNum, kernel_name_);

  auto input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);

  size_t input_size = 1;
  for (size_t i = 0; i < input_dims_; ++i) {
    input_size *= input_shape_[i];
  }

  using MatrixMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

  auto matrix_width = input_shape_[input_dims_ - 2];
  auto matrix_height = input_shape_[input_dims_ - 1];
  auto matrix_size = matrix_width * matrix_height;
  auto matrixs_num = input_size / matrix_size;

  for (int64_t k = 0; k < SizeToLong(matrixs_num); ++k) {
    MatrixMap input(input_addr + k * SizeToLong(matrix_size), matrix_width, matrix_height);
    MatrixMap output(output_addr + k * SizeToLong(matrix_size), matrix_width, matrix_height);
    output = input.template triangularView<Eigen::Upper>();
    if (diagonal_ < 0) {
      for (int64_t j = 0; j < SizeToLong(matrix_height); j++) {
        for (int64_t i = j + 1; i <= j - diagonal_ && i < SizeToLong(matrix_width); i++) {
          output(i, j) = input(i, j);
        }
      }
    } else {
      for (int64_t i = 0; i < SizeToLong(matrix_width); i++) {
        for (int64_t j = i; j < i + diagonal_ && j < SizeToLong(matrix_height); j++) {
          output(i, j) = static_cast<T>(0.0);
        }
      }
    }
  }
  return true;
}

bool TriuCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                              const std::vector<kernel::AddressPtr> &outputs) {
  switch (input_dtype_) {
    case kNumberTypeUInt8:
      return TriuCompute<uint8_t>(inputs, outputs);
    case kNumberTypeUInt16:
      return TriuCompute<uint16_t>(inputs, outputs);
    case kNumberTypeUInt32:
      return TriuCompute<uint32_t>(inputs, outputs);
    case kNumberTypeUInt64:
      return TriuCompute<uint64_t>(inputs, outputs);
    case kNumberTypeInt8:
      return TriuCompute<int8_t>(inputs, outputs);
    case kNumberTypeInt16:
      return TriuCompute<int16_t>(inputs, outputs);
    case kNumberTypeInt32:
      return TriuCompute<int32_t>(inputs, outputs);
    case kNumberTypeInt64:
      return TriuCompute<int64_t>(inputs, outputs);
    case kNumberTypeFloat16:
      return TriuCompute<float16>(inputs, outputs);
    case kNumberTypeFloat32:
      return TriuCompute<float>(inputs, outputs);
    case kNumberTypeFloat64:
      return TriuCompute<double>(inputs, outputs);
    case kNumberTypeBool:
      return TriuCompute<bool>(inputs, outputs);
    default:
      MS_LOG(ERROR) << "Unsupported data type.";
  }
  return true;
}

std::vector<KernelAttr> TriuCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
    KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
    KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
    KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Triu, TriuCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
