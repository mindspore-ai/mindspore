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

#include "mindspore/core/ops/matrix_power.h"

#include <functional>
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

bool MatrixPowerCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  dtype_ = inputs[kIndex0]->GetDtype();
  auto op_prim = std::dynamic_pointer_cast<ops::MatrixPower>(base_operator);
  MS_ERROR_IF_NULL(op_prim);
  power_ = op_prim->get_exponent();
  return true;
}

int MatrixPowerCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  output_shape_ = outputs[kIndex0]->GetShapeVector();
  return KRET_OK;
}

bool MatrixPowerCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> & /* workspace */,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputSize, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputSize, kernel_name_);
  if (dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt8) {
    LaunchKernel<uint8_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt8) {
    LaunchKernel<int8_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt16) {
    LaunchKernel<int16_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt32) {
    LaunchKernel<int32_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else {
    MS_LOG(ERROR) << "Data type is " << TypeIdLabel(dtype_) << " which is not supported.";
    return false;
  }
  return true;
}

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
void MatrixPowerCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  T *x_addr = reinterpret_cast<T *>(inputs[0]->addr);
  T *y_addr = reinterpret_cast<T *>(outputs[0]->addr);
  size_t batch = std::accumulate(output_shape_.begin(), output_shape_.end() - 2, 1, std::multiplies<int64_t>());
  size_t dim = output_shape_.back();

  std::function<void(size_t, size_t)> task;
  if constexpr (!std::is_integral_v<T>) {
    task = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        int64_t n = power_;
        size_t offset = i * dim * dim;
        Matrix<T> eigen_input = Eigen::Map<Matrix<T>>(x_addr + offset, dim, dim);
        if (n < 0) {
          n = -n;
          Eigen::FullPivLU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> LU(eigen_input);
          if (!(LU.isInvertible())) {
            MS_EXCEPTION(ValueError) << "For MatrixPower, negative power can not apply to singular matrix.";
          }
          eigen_input = LU.inverse();
        }
        Eigen::Map<Matrix<T>> eigen_output(y_addr + offset, dim, dim);
        (void)eigen_output.setIdentity();
        while (n > 0) {
          if (n % kNumber2 == 1) {
            eigen_output = eigen_output * eigen_input;
          }
          n = n / kNumber2;
          eigen_input = eigen_input * eigen_input;
        }
      }
    };
  } else {
    task = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        int64_t n = power_;
        size_t offset = i * dim * dim;
        Matrix<T> eigen_input = Eigen::Map<Matrix<T>>(x_addr + offset, dim, dim);
        if (n < 0) {
          MS_EXCEPTION(ValueError) << "For MatrixPower, n < 0 is not supported for input of integer type.";
        }
        Eigen::Map<Matrix<T>> eigen_output(y_addr + offset, dim, dim);
        (void)eigen_output.setIdentity();
        while (n > 0) {
          if (n % kNumber2 == 1) {
            eigen_output = eigen_output * eigen_input;
          }
          n = n / kNumber2;
          eigen_input = eigen_input * eigen_input;
        }
      }
    };
  }

  ParallelLaunchAutoSearch(task, batch, this, &parallel_search_info_);
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MatrixPower, MatrixPowerCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
