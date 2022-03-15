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
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kDimNum = 2;
using CholeskyInverseFunc = std::function<bool(const CNodePtr &, const std::vector<kernel::AddressPtr> &,
                                               const std::vector<kernel::AddressPtr> &)>;
template <typename T>
bool CholeskyInverseKernelFunc(const CNodePtr &node_wpt, const std::vector<kernel::AddressPtr> &inputs,
                               const std::vector<kernel::AddressPtr> &outputs) {
  auto input_x0 = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_y = reinterpret_cast<T *>(outputs[0]->addr);
  auto inputShape = AnfAlgo::GetInputDeviceShape(node_wpt, 0);
  int64_t n = SizeToLong(inputShape[0]);
  using MatrixXd = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<MatrixXd> A(input_x0, n, n);
  MatrixXd result;
  auto upper = common::AnfAlgo::GetNodeAttr<bool>(node_wpt, "upper");
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

static std::vector<std::pair<KernelAttr, CholeskyInverseFunc>> kernel_attr_list = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CholeskyInverseKernelFunc<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CholeskyInverseKernelFunc<double>}};
}  // namespace

void CholeskyInverseCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  node_wpt_ = kernel_node;
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  auto x_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
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

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "Cholesky inverse valid cpu kernel does not support this kernel data type: " << kernel_attr;
  }

  kernel_func_ = std::bind(kernel_attr_list[index].second, node_wpt_, std::placeholders::_1, std::placeholders::_2);
}

std::vector<KernelAttr> CholeskyInverseCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  std::transform(kernel_attr_list.begin(), kernel_attr_list.end(), std::back_inserter(support_list),
                 [](const std::pair<KernelAttr, CholeskyInverseFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CholeskyInverse, CholeskyInverseCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
