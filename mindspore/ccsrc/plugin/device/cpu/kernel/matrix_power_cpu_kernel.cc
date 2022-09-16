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

#include "Eigen/Core"
#include "Eigen/LU"

#include "plugin/device/cpu/kernel/matrix_power_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputSize = 1;
constexpr size_t kOutputSize = 1;
static constexpr int kNumber2 = 2;
}  // namespace

void MatrixPowerCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  node_wpt_ = kernel_node;
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
}

bool MatrixPowerCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> & /* workspace */,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputSize, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputSize, kernel_name_);
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Data type is " << TypeIdLabel(dtype_) << " which is not supported.";
  }
  return true;
}

template <typename T>
void MatrixPowerCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  auto node_ = node_wpt_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "node_wpt_ is expired.";
  }
  T *x_addr = reinterpret_cast<T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(x_addr);
  T *y_addr = reinterpret_cast<T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(y_addr);
  auto shape = Convert2SizeT(common::AnfAlgo::GetPrevNodeOutputInferShape(node_, 0));
  size_t batch = shape[0];
  size_t dim = shape[1];
  size_t matrix_size = dim * dim;
  std::vector<std::vector<float>> temp_x(batch, std::vector<float>(matrix_size));
  std::vector<std::vector<float>> temp_y(batch, std::vector<float>(matrix_size));
  power_ = common::AnfAlgo::GetNodeAttr<int64_t>(node_, "n");
  for (size_t i = 0; i < batch; i++) {
    int64_t n = power_;
    for (size_t j = 0; j < matrix_size; j++) {
      temp_x[i][j] = static_cast<float>(x_addr[i * matrix_size + j]);
    }
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> eigen_input(temp_x[i].data(), dim, dim);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> eigen_output(temp_y[i].data(), dim, dim);
    if (power_ < 0) {
      n = -n;
      Eigen::FullPivLU<Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>> lu(eigen_input);
      if (!(lu.isInvertible())) {
        MS_EXCEPTION(ValueError) << "For MatrixPower, the " << i << "-th matrix is singular"
                                 << ", but got n is negative.";
      }
      eigen_input = lu.inverse();
    }
    eigen_output.setIdentity();
    while (n > 0) {
      if (n % kNumber2 == 1) {
        eigen_output = eigen_output * eigen_input;
      }
      n = n / kNumber2;
      eigen_input = eigen_input * eigen_input;
    }
    for (size_t j = 0; j < matrix_size; j++) {
      y_addr[i * matrix_size + j] = static_cast<T>(temp_y[i][j]);
    }
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MatrixPower, MatrixPowerCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
