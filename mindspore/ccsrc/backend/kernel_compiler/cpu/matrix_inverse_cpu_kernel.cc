/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/matrix_inverse_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "Eigen/Core"
#include "Eigen/LU"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputSize = 1;
constexpr size_t kOutputSize = 1;
static constexpr int kNumber1 = 1;
static constexpr int kNumber2 = 2;
constexpr size_t kParallelDataNums = 1 * 1024;
}  // namespace

void MatrixInverseCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  node_wpt_ = kernel_node;
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
}

bool MatrixInverseCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> & /* workspace */,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputSize, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputSize, kernel_name_);

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

template <typename T>
void MatrixInverseCPUKernel::LaunchMatrixInverse(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &outputs) {
  auto node_ = node_wpt_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "node_wpt_ is expired.";
  }
  T *input_ptr = reinterpret_cast<T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(input_ptr);
  T *output_ptr = reinterpret_cast<T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(output_ptr);
  // Judge whether the input shape matches
  auto shape = AnfAlgo::GetPrevNodeOutputInferShape(node_, 0);
  if (shape.size() < kNumber2) {
    MS_LOG(EXCEPTION) << "Input x must be at least rank 2.";
  }
  if (shape[shape.size() - kNumber1] != shape[shape.size() - kNumber2]) {
    MS_LOG(EXCEPTION) << "The last two dimensions of Input x should be equal.";
  }
  auto last_dimsize = shape[shape.size() - 1];
  // Output length
  size_t input_num = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    input_num *= shape[i];
  }
  size_t matrix_size = last_dimsize * last_dimsize;
  // Number of matrices
  size_t matrix_num = input_num / matrix_size;
  // Store two-dimensional array of data for slicing
  std::vector<std::vector<T>> temp(matrix_num, std::vector<T>(matrix_size));
  for (size_t i = 0; i < matrix_num; i++) {
    for (size_t j = 0; j < matrix_size; j++) {
      temp[i][j] = *(input_ptr + i * matrix_size + j);
    }
  }
  // Gets the value of the property adjoint
  adjoint_ = AnfAlgo::GetNodeAttr<bool>(node_, "adjoint");
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
}  // namespace kernel
}  // namespace mindspore
