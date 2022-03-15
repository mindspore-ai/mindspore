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
constexpr size_t kInputSize = 1;
constexpr size_t kOutputSize = 1;
static constexpr int kNumber0 = 0;
static constexpr int kNumber1 = 1;
static constexpr int kNumber2 = 2;
}  // namespace

void MatrixDeterminantCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  node_wpt_ = kernel_node;
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kInputSize, kernel_name_);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kOutputSize, kernel_name_);
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
  auto node_ = node_wpt_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "node_wpt_ is expired.";
  }
  T *input = reinterpret_cast<T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(input);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(output);
  // Check if it's a square array
  auto dims = common::AnfAlgo::GetPrevNodeOutputInferShape(node_, 0);
  if (dims.size() < kNumber2) {
    MS_LOG(EXCEPTION) << "Input x must be at least rank 2.";
  }
  if (dims[dims.size() - kNumber1] != dims[dims.size() - kNumber2]) {
    MS_LOG(EXCEPTION) << "The last two dimensions of Input x should be equal.";
  }
  size_t m = dims[dims.size() - 1];
  size_t n = 1;
  for (size_t i = kNumber0; i < dims.size() - kNumber2; i++) {
    n *= dims[i];
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
  CPUKernelUtils::ParallelFor(task, n);
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MatrixDeterminant, MatrixDeterminantCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
