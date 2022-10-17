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

#include "plugin/device/cpu/kernel/matrix_determinant_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "Eigen/Core"
#include "Eigen/LU"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputOutputNumber = 1;
static constexpr int kNumber0 = 0;
static constexpr int kNumber1 = 1;
static constexpr int kNumber2 = 2;
}  // namespace

bool MatrixDeterminantCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (inputs.size() != kInputOutputNumber) {
    MS_LOG(ERROR) << "For " << kernel_name_ << ", the inputs number must be 1, but got " << inputs.size();
    return false;
  }
  if (outputs.size() != kInputOutputNumber) {
    MS_LOG(ERROR) << "For " << kernel_name_ << ", the inputs number must be 1, but got " << outputs.size();
    return false;
  }
  dtype_ = inputs[kIndex0]->GetDtype();
  return true;
}

int MatrixDeterminantCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  input_ = inputs[kIndex0]->GetShapeVector();
  if (input_.size() < kNumber2) {
    MS_LOG(ERROR) << "input must be at least rank 2, but got " << input_.size();
    return KRET_RESIZE_FAILED;
  }
  if (input_[input_.size() - kNumber1] != input_[input_.size() - kNumber2]) {
    MS_LOG(ERROR) << "The last two dimensions of Input x must be equal.";
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

bool MatrixDeterminantCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> & /* workspace */,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat32) {
    LaunchMatrixDeterminant<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchMatrixDeterminant<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeComplex64) {
    LaunchMatrixDeterminant<std::complex<float>>(inputs, outputs);
  } else if (dtype_ == kNumberTypeComplex128) {
    LaunchMatrixDeterminant<std::complex<double>>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "MatrixDeterminant kernel data type " << TypeIdLabel(dtype_) << " not support.";
  }
  return true;
}

template <typename T>
void MatrixDeterminantCpuKernelMod::LaunchMatrixDeterminant(const std::vector<AddressPtr> &inputs,
                                                            const std::vector<AddressPtr> &outputs) {
  T *input = reinterpret_cast<T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(input);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(output);

  size_t m = LongToSize(input_[input_.size() - 1]);
  int64_t n = 1;
  for (size_t i = kNumber0; i < input_.size() - kNumber2; i++) {
    n *= input_[i];
  }
  auto task = [this, &m, input, output](size_t start, size_t end) {
    for (size_t k = start; k < end; k++) {
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eMatrix(m, m);
      for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
          eMatrix(i, j) = *(input + k * m * m + i * m + j);
        }
      }
      // use eigen to calculate determinant
      T result = eMatrix.determinant();
      *(output + k) = result;
    }
  };
  CPUKernelUtils::ParallelFor(task, LongToSize(n));
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MatrixDeterminant, MatrixDeterminantCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
