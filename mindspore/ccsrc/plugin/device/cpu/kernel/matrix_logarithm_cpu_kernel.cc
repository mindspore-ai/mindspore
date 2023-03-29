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
bool MatrixLogarithmCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputSize, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputSize, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int MatrixLogarithmCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto shape_x = inputs.at(kIndex0)->GetShapeVector();
  auto shape_y = outputs.at(kIndex0)->GetShapeVector();
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
  shape_x_ = shape_x;
  return KRET_OK;
}

std::vector<std::pair<KernelAttr, MatrixLogarithmCpuKernelMod::MatrixLogarithmLaunchFunc>>
  MatrixLogarithmCpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &MatrixLogarithmCpuKernelMod::LaunchMatrixLogarithm<std::complex<float>>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &MatrixLogarithmCpuKernelMod::LaunchMatrixLogarithm<std::complex<double>>}};

std::vector<KernelAttr> MatrixLogarithmCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MatrixLogarithmCpuKernelMod::MatrixLogarithmLaunchFunc> &pair) {
                         return pair.first;
                       });
  return support_list;
}

template <typename T>
void MatrixLogarithmCpuKernelMod::LaunchMatrixLogarithm(const std::vector<AddressPtr> &inputs,
                                                        const std::vector<AddressPtr> &outputs) {
  auto input_x = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_y = reinterpret_cast<T *>(outputs[0]->addr);
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXd;
  size_t shape_size = shape_x_.size();
  auto m = IntToSize(shape_x_[shape_size - 1]);
  size_t size_mm = m * m;
  if (size_mm > 0) {
    size_t input_num = 1;
    for (size_t i = 0; i < shape_x_.size(); i++) {
      input_num *= IntToSize(shape_x_[i]);
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
