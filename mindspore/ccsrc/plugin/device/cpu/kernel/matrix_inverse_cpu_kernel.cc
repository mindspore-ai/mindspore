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

#include "plugin/device/cpu/kernel/matrix_inverse_cpu_kernel.h"
#include <map>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "Eigen/Core"
#include "Eigen/LU"
#include "mindspore/core/ops/matrix_inverse.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputSize = 1;
constexpr size_t kOutputSize = 1;
static constexpr int kNumber1 = 1;
static constexpr int kNumber2 = 2;
constexpr size_t kParallelDataNums = 1 * 1024;
}  // namespace

bool MatrixInverseCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::MatrixInverse>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast " << kernel_name_ << "  ops failed!";
    return false;
  }
  dtype_ = inputs[kIndex0]->GetDtype();
  adjoint_ = kernel_ptr->get_adjoint();
  return true;
}

bool MatrixInverseCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> & /* workspace */,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat32) {
    LaunchMatrixInverse<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchMatrixInverse<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeComplex64) {
    LaunchMatrixInverse<std::complex<float>>(inputs, outputs);
  } else if (dtype_ == kNumberTypeComplex128) {
    LaunchMatrixInverse<std::complex<double>>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "MatrixInverse kernel data type " << TypeIdLabel(dtype_) << " not support.";
  }
  return true;
}

int MatrixInverseCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputSize, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputSize, kernel_name_);
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  // Judge whether the input shape matches
  input_shape_ = inputs[kIndex0]->GetShapeVector();
  if (input_shape_.size() < kNumber2) {
    MS_LOG(EXCEPTION) << "Input x must be at least rank 2.";
  }
  if (input_shape_[input_shape_.size() - kNumber1] != input_shape_[input_shape_.size() - kNumber2]) {
    MS_LOG(EXCEPTION) << "The last two dimensions of Input x must be equal.";
  }
  return KRET_OK;
}

template <typename T>
void MatrixInverseCpuKernelMod::LaunchMatrixInverse(const std::vector<AddressPtr> &inputs,
                                                    const std::vector<AddressPtr> &outputs) {
  T *input_ptr = reinterpret_cast<T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(input_ptr);
  T *output_ptr = reinterpret_cast<T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(output_ptr);

  auto last_dimsize = LongToSize(input_shape_[input_shape_.size() - 1]);
  // Output length
  size_t input_num = 1;
  for (size_t i = 0; i < input_shape_.size(); i++) {
    input_num *= input_shape_[i];
  }
  auto matrix_size = last_dimsize * last_dimsize;
  // Number of matrices
  auto matrix_num = input_num / matrix_size;
  // Store two-dimensional array of data for slicing

  std::vector<std::vector<T>> temp(matrix_num, std::vector<T>(matrix_size));
  for (size_t i = 0; i < matrix_num; i++) {
    for (size_t j = 0; j < matrix_size; j++) {
      temp[i][j] = *(input_ptr + i * matrix_size + j);
    }
  }
  auto one_size = sizeof(*input_ptr);

  if ((one_size * input_num) <= kParallelDataNums) {
    for (size_t i = 0; i < matrix_num; i++) {
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> eigen_input(temp[i].data(), last_dimsize,
                                                                               last_dimsize);
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> eigen_output(output_ptr + i * matrix_size,
                                                                                last_dimsize, last_dimsize);
      if (adjoint_) {
        eigen_input = eigen_input.adjoint().eval();
      }
      Eigen::FullPivLU<Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> lu(eigen_input);
      eigen_output = lu.inverse();
    }
  } else {
    auto task = [this, &last_dimsize, &matrix_size, &temp, output_ptr](size_t start, size_t end) {
      for (auto i = start; i < end; i++) {
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> eigen_input(temp[i].data(), last_dimsize,
                                                                                 last_dimsize);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> eigen_output(output_ptr + i * matrix_size,
                                                                                  last_dimsize, last_dimsize);
        if (adjoint_) {
          eigen_input = eigen_input.adjoint().eval();
        }
        Eigen::FullPivLU<Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> lu(eigen_input);
        eigen_output = lu.inverse();
      }
    };
    CPUKernelUtils::ParallelFor(task, matrix_num);
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MatrixInverse, MatrixInverseCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
