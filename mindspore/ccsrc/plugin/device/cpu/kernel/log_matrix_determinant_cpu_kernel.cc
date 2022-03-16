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

#include "plugin/device/cpu/kernel/log_matrix_determinant_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "Eigen/LU"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputSize = 1;
constexpr size_t kOutputSize = 2;
constexpr int64_t kParallelDataNums = 8 * 1024;
static constexpr int kNumber0 = 0;
static constexpr int kNumber1 = 1;
static constexpr int kNumber2 = 2;
}  // namespace

void LogMatrixDeterminantCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  node_wpt_ = kernel_node;
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kInputSize, kernel_name_);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kOutputSize, kernel_name_);
  auto shape_x = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto shape_sign = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
  auto shape_y = common::AnfAlgo::GetOutputInferShape(kernel_node, 1);
  size_t shape_size_x = shape_x.size();
  size_t shape_size_sign = shape_sign.size();
  size_t shape_size_y = shape_y.size();
  if (shape_size_x < kNumber2) {
    MS_LOG(EXCEPTION) << "Input x must be at least rank 2.";
  }
  if (shape_x[shape_size_x - kNumber1] < kNumber1) {
    MS_LOG(EXCEPTION) << "Input x last dimension must be at least 1.";
  }
  if (shape_x[shape_size_x - kNumber2] != shape_x[shape_size_x - kNumber1]) {
    MS_LOG(EXCEPTION) << "The last two dimensions of Input x should be equal.";
  }
  if (shape_size_sign != shape_size_x - kNumber2) {
    MS_LOG(EXCEPTION) << "Output sign must be rank [" << shape_size_x - kNumber2 << "], got [" << shape_size_sign
                      << "].";
  }
  if (shape_size_y != shape_size_x - kNumber2) {
    MS_LOG(EXCEPTION) << "Output y must be rank [" << shape_size_x - kNumber2 << "], got [" << shape_size_y << "].";
  }
  for (size_t i = kNumber0; i < shape_size_x - kNumber2; i++) {
    if (shape_sign[i] != shape_x[i]) {
      MS_LOG(EXCEPTION) << "Output sign and Input x dimension " << i << " must be equal.";
    }
    if (shape_y[i] != shape_x[i]) {
      MS_LOG(EXCEPTION) << "Output y and Input x dimension " << i << " must be equal.";
    }
  }
}

bool LogMatrixDeterminantCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> & /* workspace */,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat32) {
    LaunchLogMatrixDeterminant<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchLogMatrixDeterminant<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeComplex64) {
    LaunchLogMatrixDeterminant<std::complex<float>>(inputs, outputs);
  } else if (dtype_ == kNumberTypeComplex128) {
    LaunchLogMatrixDeterminant<std::complex<double>>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "LogMatrixDeterminant kernel data type " << TypeIdLabel(dtype_) << " not support.";
  }
  return true;
}

template <typename T>
void LogMatrixDeterminantCpuKernelMod::LaunchLogMatrixDeterminant(const std::vector<AddressPtr> &inputs,
                                                                  const std::vector<AddressPtr> &outputs) {
  auto node_ = node_wpt_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "node_wpt_ is expired.";
  }
  auto input_x = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_sign = reinterpret_cast<T *>(outputs[0]->addr);
  auto output_y = reinterpret_cast<T *>(outputs[1]->addr);

  auto shape_x = common::AnfAlgo::GetPrevNodeOutputInferShape(node_, 0);
  size_t shape_size = shape_x.size();
  size_t m = shape_x[shape_size - 1];
  size_t size_mm = m * m;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MartixXd;
  using RealT = typename Eigen::NumTraits<T>::Real;
  if (size_mm > 0) {
    size_t input_num = 1;
    for (size_t i = 0; i < shape_x.size(); i++) {
      input_num *= shape_x[i];
    }
    size_t matrix_num = input_num / size_mm;
    size_t data_size = input_num * sizeof(T);

    if (data_size <= kParallelDataNums) {
      for (size_t i = 0; i < matrix_num; i++) {
        RealT log_abs_det = 0;
        T sign = 1;
        Eigen::Map<MartixXd> martix_x(input_x + i * m * m, m, m);
        if (martix_x.size() > 0) {
          Eigen::PartialPivLU<MartixXd> lu(martix_x);
          MartixXd LU = lu.matrixLU();
          sign = lu.permutationP().determinant();
          auto diag = LU.diagonal().array().eval();
          auto abs_diag = diag.cwiseAbs().eval();
          auto abs_diag_inverse = abs_diag.cwiseInverse();
          log_abs_det += abs_diag.log().sum();
          sign *= (diag * abs_diag_inverse).prod();
        }
        if (!Eigen::numext::isfinite(log_abs_det)) {
          sign = 0;
          log_abs_det = log_abs_det > 0 ? -std::log(RealT(0)) : std::log(RealT(0));
        }
        *(output_sign + i) = sign;
        *(output_y + i) = log_abs_det;
      }
    } else {
      auto task = [this, &m, input_x, output_sign, output_y](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
          RealT log_abs_det = 0;
          T sign = 1;
          Eigen::Map<MartixXd> martix_x(input_x + i * m * m, m, m);
          if (martix_x.size() > 0) {
            Eigen::PartialPivLU<MartixXd> lu(martix_x);
            MartixXd LU = lu.matrixLU();
            sign = lu.permutationP().determinant();
            auto diag = LU.diagonal().array().eval();
            auto abs_diag = diag.cwiseAbs().eval();
            auto abs_diag_inverse = abs_diag.cwiseInverse();
            log_abs_det += abs_diag.log().sum();
            sign *= (diag * abs_diag_inverse).prod();
          }
          if (!Eigen::numext::isfinite(log_abs_det)) {
            sign = 0;
            log_abs_det = log_abs_det > 0 ? -std::log(RealT(0)) : std::log(RealT(0));
          }
          *(output_sign + i) = sign;
          *(output_y + i) = log_abs_det;
        }
      };
      CPUKernelUtils::ParallelFor(task, matrix_num);
    }
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LogMatrixDeterminant, LogMatrixDeterminantCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
