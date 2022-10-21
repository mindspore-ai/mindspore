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

bool LogMatrixDeterminantCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputSize, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputSize, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  return true;
}

int LogMatrixDeterminantCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs,
                                             const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  dtype_ = inputs[kIndex0]->GetDtype();
  shape_x_ = inputs[kIndex0]->GetShapeVector();
  auto shape_sign = outputs[kIndex0]->GetShapeVector();
  auto shape_y = outputs[kIndex1]->GetShapeVector();
  size_t shape_size_x = shape_x_.size();
  size_t shape_size_sign = shape_sign.size();
  size_t shape_size_y = shape_y.size();
  if (shape_size_x < kNumber2) {
    MS_LOG(ERROR) << "Input x must be at least rank 2.";
    return KRET_RESIZE_FAILED;
  }
  if (shape_x_[shape_size_x - kNumber1] < kNumber1) {
    MS_LOG(ERROR) << "Input x last dimension must be at least 1.";
    return KRET_RESIZE_FAILED;
  }
  if (shape_x_[shape_size_x - kNumber2] != shape_x_[shape_size_x - kNumber1]) {
    MS_LOG(ERROR) << "The last two dimensions of Input x must be equal.";
    return KRET_RESIZE_FAILED;
  }
  if (shape_size_sign != shape_size_x - kNumber2) {
    MS_LOG(ERROR) << "Output sign must be rank [" << shape_size_x - kNumber2 << "], got [" << shape_size_sign << "].";
    return KRET_RESIZE_FAILED;
  }
  if (shape_size_y != shape_size_x - kNumber2) {
    MS_LOG(ERROR) << "Output y must be rank [" << shape_size_x - kNumber2 << "], got [" << shape_size_y << "].";
    return KRET_RESIZE_FAILED;
  }
  for (size_t i = kNumber0; i < shape_size_x - kNumber2; i++) {
    if (shape_sign[i] != shape_x_[i]) {
      MS_LOG(ERROR) << "Output sign and Input x dimension " << i << " must be equal.";
      return KRET_RESIZE_FAILED;
    }
    if (shape_y[i] != shape_x_[i]) {
      MS_LOG(ERROR) << "Output y and Input x dimension " << i << " must be equal.";
      return KRET_RESIZE_FAILED;
    }
  }
  return KRET_OK;
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
  auto input_x = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_sign = reinterpret_cast<T *>(outputs[0]->addr);
  auto output_y = reinterpret_cast<T *>(outputs[1]->addr);

  size_t shape_size = shape_x_.size();
  size_t m = shape_x_[shape_size - 1];
  size_t size_mm = m * m;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MartixXd;
  using RealT = typename Eigen::NumTraits<T>::Real;
  if (size_mm > 0) {
    size_t input_num = 1;
    for (size_t i = 0; i < shape_x_.size(); i++) {
      input_num *= shape_x_[i];
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
