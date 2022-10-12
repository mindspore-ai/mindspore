/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/tril_cpu_kernel.h"

#include <algorithm>
#include "Eigen/Core"

#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kTrilInputsNum = 1;
constexpr size_t kTrilOutputsNum = 1;
constexpr size_t kDim = 2;
}  // namespace

void TrilCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_shape_ = Convert2SizeTClipNeg(common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0));
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);

  input_dims_ = input_shape_.size();
  if (input_dims_ < kDim) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'x' must be at least 1-D, but got "
                      << input_dims_ << "-D.";
  }
  if (common::AnfAlgo::HasNodeAttr("diagonal", kernel_node)) {
    diagonal_ = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "diagonal");
  }
}

bool TrilCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                              const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kTrilInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kTrilOutputsNum, kernel_name_);

  switch (dtype_) {
    case (kNumberTypeUInt8):
      LaunchKernel<uint8_t>(inputs, outputs);
      break;
    case (kNumberTypeUInt16):
      LaunchKernel<uint16_t>(inputs, outputs);
      break;
    case (kNumberTypeUInt32):
      LaunchKernel<uint32_t>(inputs, outputs);
      break;
    case (kNumberTypeUInt64):
      LaunchKernel<uint64_t>(inputs, outputs);
      break;
    case (kNumberTypeInt8):
      LaunchKernel<int8_t>(inputs, outputs);
      break;
    case (kNumberTypeInt16):
      LaunchKernel<int16_t>(inputs, outputs);
      break;
    case (kNumberTypeInt32):
      LaunchKernel<int32_t>(inputs, outputs);
      break;
    case (kNumberTypeInt64):
      LaunchKernel<int64_t>(inputs, outputs);
      break;
    case (kNumberTypeFloat16):
      LaunchKernel<float16>(inputs, outputs);
      break;
    case (kNumberTypeFloat32):
      LaunchKernel<float>(inputs, outputs);
      break;
    case (kNumberTypeFloat64):
      LaunchKernel<double>(inputs, outputs);
      break;
    case (kNumberTypeBool):
      LaunchKernel<bool>(inputs, outputs);
      break;
    default:
      MS_LOG(EXCEPTION) << "the datatype of the input not support, support datatype: "
                           "uint8, uint16, uint32, uint64, int8, int16, int32, int64, "
                           "float16, float32, float64, bool.";
  }
  return true;
}

template <typename T>
void TrilCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
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
    output = input.template triangularView<Eigen::Lower>();
    if (diagonal_ > 0) {
      for (int64_t i = 0; i < SizeToLong(matrix_width); i++) {
        for (int64_t j = i + 1; j <= i + SizeToLong(diagonal_) && j < SizeToLong(matrix_height); j++) {
          output(i, j) = input(i, j);
        }
      }
    } else {
      for (int64_t j = 0; j < SizeToLong(matrix_height); j++) {
        for (int64_t i = j; i < j - SizeToLong(diagonal_) && i < SizeToLong(matrix_width); i++) {
          output(i, j) = static_cast<T>(0.0);
        }
      }
    }
  }
}

std::vector<KernelAttr> TrilCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
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
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Tril, TrilCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
