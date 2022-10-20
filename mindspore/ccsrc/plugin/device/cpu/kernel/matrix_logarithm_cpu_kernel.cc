/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/matrix_logarithm_cpu_kernel.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <complex>
#include <cmath>
#include <chrono>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputSize = 1;
constexpr size_t kOutputSize = 1;
static constexpr int kNumber0 = 0;
static constexpr int kNumber1 = 1;
static constexpr int kNumber2 = 2;
constexpr size_t kParallelDataNums = 2 * 1024;
constexpr int64_t kParallelDataNumMid = 16 * 1024;
}  // namespace

void MatrixLogarithmCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  node_wpt_ = kernel_node;
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kInputSize, kernel_name_);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kOutputSize, kernel_name_);
  auto shape_x = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto shape_y = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
  size_t shape_size_x = shape_x.size();
  if (shape_size_x < kNumber2) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", the input 'x' must be at least rank 2.";
  }
  if (shape_x[shape_size_x - kNumber2] != shape_x[shape_size_x - kNumber1]) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", the last two dimensions of input 'x' must be equal.";
  }
  for (size_t i = kNumber0; i < shape_size_x; i++) {
    if (shape_y[i] != shape_x[i]) {
      MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", the output 'y' and the input 'x' dimension " << i
                        << " must be equal.";
    }
  }
}

bool MatrixLogarithmCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> & /* workspace */,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeComplex64) {
    LaunchMatrixLogarithm<std::complex<float>>(inputs, outputs);
  } else if (dtype_ == kNumberTypeComplex128) {
    LaunchMatrixLogarithm<std::complex<double>>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For MatrixLogarithm, data type " << TypeIdLabel(dtype_) << " not support.";
  }
  return true;
}

template <typename T>
void MatrixLogarithmCpuKernelMod::LaunchMatrixLogarithm(const std::vector<AddressPtr> &inputs,
                                                        const std::vector<AddressPtr> &outputs) {
  auto node_ = node_wpt_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "For MatrixLogarithm, node_wpt_ is expired.";
  }
  auto input_x = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_y = reinterpret_cast<T *>(outputs[0]->addr);
  auto shape_x = common::AnfAlgo::GetPrevNodeOutputInferShape(node_, 0);
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXd;
  size_t shape_size = shape_x.size();
  auto m = shape_x[shape_size - 1];
  size_t size_mm = m * m;
  if (size_mm > 0) {
    size_t input_num = 1;
    for (size_t i = 0; i < shape_x.size(); i++) {
      input_num *= shape_x[i];
    }
    size_t matrix_num = input_num / size_mm;
    size_t data_size = input_num * sizeof(T);
    auto task = [this, &m, input_x, output_y](size_t start, size_t end) {
      for (size_t l = start; l < end; l++) {
        Eigen::Map<MatrixXd> matrix_x(input_x + l * m * m, m, m);
        Eigen::Map<MatrixXd> matrix_output(output_y + l * m * m, m, m);
        if (matrix_x.size() > 0) {
          matrix_output = matrix_x.log();
        }
      }
    };
    if (data_size <= kParallelDataNums) {
      task(0, matrix_num);
    } else {
      CPUKernelUtils::ParallelFor(task, matrix_num);
    }
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MatrixLogarithm, MatrixLogarithmCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
