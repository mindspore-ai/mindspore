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
#include "plugin/device/cpu/kernel/cholesky_inverse_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kDimNum = 2;
}
template <typename T>
void CholeskyInverseCpuKernelMod<T>::InitKernel(const CNodePtr &kernel_node) {
  node_wpt_ = kernel_node;
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  auto x_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kOutputNum, kernel_name_);
  if (x_shape.size() != kDimNum) {
    MS_EXCEPTION(ValueError) << "The dimension of x must be equal to 2, while got x_dim: " << x_shape.size() << ".";
  }
  if (x_shape[x_shape.size() - 1] != x_shape[x_shape.size() - kDimNum]) {
    MS_EXCEPTION(ValueError) << "For CholeskyInverse"
                             << " input cholesky_inverse should be square matrix "
                             << "while row is " << x_shape[x_shape.size() - kDimNum] << ", col is "
                             << x_shape[x_shape.size() - 1];
  }
}

template <typename T>
bool CholeskyInverseCpuKernelMod<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  auto input_x0 = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_y = reinterpret_cast<T *>(outputs[0]->addr);
  auto inputShape = AnfAlgo::GetInputDeviceShape(node_wpt_, 0);
  int64_t n = SizeToLong(inputShape[0]);
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXd;
  Eigen::Map<MatrixXd> A(input_x0, n, n);
  MatrixXd result;
  auto upper = AnfAlgo::GetNodeAttr<bool>(node_wpt_, "upper");
  if (upper) {
    result = (A.transpose() * A).inverse();
  } else {
    result = (A * A.transpose()).inverse();
  }
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = 0; j < n; j++) {
      *(output_y + i * n + j) = result(i, j);
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
