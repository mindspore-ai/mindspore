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
#include "mindspore/core/ops/cholesky_inverse_.h"

namespace mindspore {
namespace kernel {
bool CholeskyInverseCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  constexpr size_t kInputNum = 1;
  constexpr size_t kOutputNum = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  auto cholesky_inverse_ptr = std::dynamic_pointer_cast<ops::CholeskyInverse>(base_operator);
  MS_ERROR_IF_NULL(cholesky_inverse_ptr);
  is_upper_ = cholesky_inverse_ptr->get_upper();
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int CholeskyInverseCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto x_shape = LongVecToSizeVec(inputs[kIndex0]->GetShapeVector());
  input_dim_0_ = x_shape[0];
  return KRET_OK;
}

template <typename T>
bool CholeskyInverseCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &,
                                               const std::vector<kernel::AddressPtr> &outputs) {
  auto input_x0 = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_y = reinterpret_cast<T *>(outputs[0]->addr);
  using MatrixXd = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<MatrixXd> A(input_x0, input_dim_0_, input_dim_0_);
  MatrixXd result;
  if (is_upper_) {
    result = (A.transpose() * A).inverse();
  } else {
    result = (A * A.transpose()).inverse();
  }
  for (int64_t i = 0; i < input_dim_0_; i++) {
    for (int64_t j = 0; j < input_dim_0_; j++) {
      *(output_y + i * input_dim_0_ + j) = result(i, j);
    }
  }
  return true;
}

const std::vector<std::pair<KernelAttr, CholeskyInverseCpuKernelMod::KernelRunFunc>>
  &CholeskyInverseCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, CholeskyInverseCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CholeskyInverseCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CholeskyInverseCpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CholeskyInverse, CholeskyInverseCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
