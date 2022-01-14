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

#include "backend/kernel_compiler/cpu/lstsq_cpu_kernel.h"
#include <Eigen/Dense>
#include "runtime/device/cpu/cpu_device_address.h"
#include "backend/kernel_compiler/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kLstsqInputsNum = 2;
constexpr size_t kLstsqOutputsNum = 1;
constexpr size_t kXDimNum = 2;
constexpr size_t kADimNum_1 = 1;
constexpr size_t kADimNum_2 = 2;
}  // namespace

void LstsqCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  input_0_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  input_1_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  if (input_0_shape_.size() != kXDimNum) {
    MS_LOG(EXCEPTION) << "The input x tensor's rank must be 2 for 'Lstsq' Op, but x tensor's rank is "
                      << input_0_shape_.size();
  }
  if (input_1_shape_.size() != kADimNum_2 && input_1_shape_.size() != kADimNum_1) {
    MS_LOG(EXCEPTION) << "The input a tensor's rank must be 2 or 1 for 'Lstsq' Op, but a tensor's rank is "
                      << input_1_shape_.size();
  }
  if (input_0_shape_[0] != input_1_shape_[0]) {
    MS_LOG(EXCEPTION) << "The length of x_dim[0]: " << input_0_shape_[0]
                      << " is not equal to the length of a_dims[0]: " << input_1_shape_[0] << ".";
  }
  dtype_0_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  dtype_1_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 1);
  if (dtype_0_ != dtype_1_) {
    MS_LOG(EXCEPTION) << "For Lstsq input's dtypes are not the same.";
  }
}

bool LstsqCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                               const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kLstsqInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kLstsqOutputsNum, kernel_name_);
  if (dtype_0_ == kNumberTypeFloat16) {
    LaunchKernel<float, float16>(inputs, outputs);
  } else if (dtype_0_ == kNumberTypeFloat32) {
    LaunchKernel<float, float>(inputs, outputs);
  } else if (dtype_0_ == kNumberTypeFloat64) {
    LaunchKernel<double, double>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported input data type.";
  }
  return true;
}

template <typename T1, typename T2>
void LstsqCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  auto input_0_addr = reinterpret_cast<T2 *>(inputs[0]->addr);
  auto input_1_addr = reinterpret_cast<T2 *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<T2 *>(outputs[0]->addr);
  size_t m = input_0_shape_[0];
  size_t n = input_0_shape_[1];
  size_t k = 0;
  if (input_1_shape_.size() == kADimNum_1) {
    k = 1;
  } else {
    k = input_1_shape_[1];
  }

  typedef Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MartixXd;
  MartixXd A(m, n);
  MartixXd B(m, k);
  for (size_t i = 0; i < m * n; i++) {
    A.data()[i] = static_cast<T1>(input_0_addr[i]);
  }
  for (size_t i = 0; i < m * k; i++) {
    B.data()[i] = static_cast<T1>(input_1_addr[i]);
  }
  MartixXd result;
  if (m >= n) {
    result = A.colPivHouseholderQr().solve(B);
  } else {
    MartixXd A_Transpose = A.transpose();
    MartixXd temp = A * A_Transpose;
    MartixXd tempI = temp.inverse();
    MartixXd x = A_Transpose * tempI;
    MartixXd output = x * B;
    result = output;
  }
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < k; j++) {
      *(output_addr + i * k + j) = static_cast<T2>(result(i, j));
    }
}
}  // namespace kernel
}  // namespace mindspore
