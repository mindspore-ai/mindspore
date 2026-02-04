/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/eigen/svd_cpu_kernel.h"
#include <algorithm>
#include <memory>
#include <map>
#include "kernel/cpu/eigen/svd_cpu_kernel_function.h"
#include "kernel/cpu/eigen/eigen_common_utils.h"

#include "include/utils/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace svd_cpu {
namespace {
using float_complex = std::complex<float>;
using double_complex = std::complex<double>;
const size_t kSvdInputsNum = 3;
const size_t kSvdOutputsNum = 3;
}  // namespace

bool SvdCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSvdInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSvdOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this data type: " << kernel_attr;
    return false;
  }

  kernel_func_ = func_list_[index].second;
  return true;
}

bool SvdCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                             const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(kernel_func_);
  return kernel_func_(this, inputs, workspace, outputs);
}

int SvdCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  int ret = NativeCpuKernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  auto in_shape = inputs[kIndex0]->GetShapeVector();
  full_matrices_ = inputs[kIndex1]->GetValueWithCheck<bool>();
  compute_uv_ = inputs[kIndex2]->GetValueWithCheck<bool>();
  if (IsShapeNone(in_shape)) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", the input can not be a empty tensor, but got: " << in_shape;
  }

  std::vector<size_t> input_shape = std::vector<size_t>(inputs.at(kIndex0)->GetDeviceShapeVector().begin(),
                                                        inputs.at(kIndex0)->GetDeviceShapeVector().end());
  size_t dim = input_shape.size();
  if (dim < kDim2) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_
                      << ", input dimension must be greater than or equal to 2, but got: " << dim;
  }

  num_of_rows_ = input_shape[dim - kDim2];
  num_of_cols_ = input_shape[dim - kDim1];
  batch_size_ = 1;
  for (size_t i = 0; i < dim - kDim2; i++) {
    batch_size_ = batch_size_ * input_shape[i];
  }
  return ret;
}

template <typename T>
bool SvdCpuKernelMod::LaunchKernelFloat(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs) {
  auto *input_a = GetDeviceAddress<T>(inputs, kIndex0);
  auto *output_s = GetDeviceAddress<T>(outputs, kIndex0);
  auto *output_u = GetDeviceAddress<T>(outputs, kIndex1);
  auto *output_v = GetDeviceAddress<T>(outputs, kIndex2);

  std::map<bool, std::pair<int, int>> optionMap{{true, {Eigen::ComputeFullU, Eigen::ComputeFullV}},
                                                {false, {Eigen::ComputeThinU, Eigen::ComputeThinV}}};
  std::function<Eigen::BDCSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor>>(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor> &, unsigned int)>
    svd_func = SVDFloat<T>;
  std::function<void(std::size_t, std::size_t)> task;

  if (compute_uv_) {
    task = [&](int64_t start, int64_t end) {
      for (; start < end; ++start) {
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor>> matrix(
          input_a + start * num_of_rows_ * num_of_cols_, num_of_rows_, num_of_cols_);
        auto svd = svd_func(matrix, static_cast<int>(optionMap[full_matrices_].first) |
                                      static_cast<int>(optionMap[full_matrices_].second));

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor> s = svd.singularValues();
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor> u = svd.matrixU();
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor> v = svd.matrixV();

        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor>>(output_s + start * s.rows() * s.cols(),
                                                                               s.rows(), s.cols()) = s;
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor>>(output_u + start * u.rows() * u.cols(),
                                                                               u.rows(), u.cols()) = u;
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor>>(output_v + start * v.rows() * v.cols(),
                                                                               v.rows(), v.cols()) = v;
      }
    };
  } else {
    task = [&](int64_t start, int64_t end) {
      for (; start < end; ++start) {
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor>> matrix(
          input_a + start * num_of_rows_ * num_of_cols_, num_of_rows_, num_of_cols_);
        auto svd = svd_func(matrix, static_cast<int>(optionMap[full_matrices_].first) |
                                      static_cast<int>(optionMap[full_matrices_].second));

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor> s = svd.singularValues();
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor>>(output_s + start * s.rows() * s.cols(),
                                                                               s.rows(), s.cols()) = s;
      }
    };
  }
  ParallelLaunchAutoSearch(task, batch_size_, this, &parallel_search_info_);

  // Set 0 for u and v if not compute u and v.
  if (!compute_uv_) {
    MS_EXCEPTION_IF_NULL(output_u);
    MS_EXCEPTION_IF_NULL(output_v);
    output_u[0] = static_cast<T>(0);
    output_v[0] = static_cast<T>(0);
  }
  return true;
}

template <typename T>
bool SvdCpuKernelMod::LaunchKernelComplex(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &workspace,
                                          const std::vector<KernelTensor *> &outputs) {
  auto *input_a = GetDeviceAddress<T>(inputs, kIndex0);
  auto *output_s = GetDeviceAddress<T>(outputs, kIndex0);
  auto *output_u = GetDeviceAddress<T>(outputs, kIndex1);
  auto *output_v = GetDeviceAddress<T>(outputs, kIndex2);

  std::map<bool, std::pair<int, int>> optionMap{{true, {Eigen::ComputeFullU, Eigen::ComputeFullV}},
                                                {false, {Eigen::ComputeThinU, Eigen::ComputeThinV}}};
  std::function<Eigen::BDCSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor>>(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor> &, unsigned int)>
    svd_func = SVDComplex<T>;
  std::function<void(std::size_t, std::size_t)> task;

  if (compute_uv_) {
    task = [&](int64_t start, int64_t end) {
      for (; start < end; ++start) {
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor>> matrix(
          input_a + start * num_of_rows_ * num_of_cols_, num_of_rows_, num_of_cols_);
        auto svd = svd_func(matrix, static_cast<int>(optionMap[full_matrices_].first) |
                                      static_cast<int>(optionMap[full_matrices_].second));

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor> s = svd.singularValues();
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor> u = svd.matrixU();
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor> v = svd.matrixV();

        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor>>(output_s + start * s.rows() * s.cols(),
                                                                               s.rows(), s.cols()) = s;
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor>>(output_u + start * u.rows() * u.cols(),
                                                                               u.rows(), u.cols()) = u;
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor>>(output_v + start * v.rows() * v.cols(),
                                                                               v.rows(), v.cols()) = v;
      }
    };
  } else {
    task = [&](int64_t start, int64_t end) {
      for (; start < end; ++start) {
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor>> matrix(
          input_a + start * num_of_rows_ * num_of_cols_, num_of_rows_, num_of_cols_);
        auto svd = svd_func(matrix, static_cast<int>(optionMap[full_matrices_].first) |
                                      static_cast<int>(optionMap[full_matrices_].second));

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor> s = svd.singularValues();
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor>>(output_s + start * s.rows() * s.cols(),
                                                                               s.rows(), s.cols()) = s;
      }
    };
  }
  ParallelLaunchAutoSearch(task, batch_size_, this, &parallel_search_info_);

  // Set 0 for u and v if not compute u and v.
  if (!compute_uv_) {
    MS_EXCEPTION_IF_NULL(output_u);
    MS_EXCEPTION_IF_NULL(output_v);
    output_u[0] = static_cast<T>(0);
    output_v[0] = static_cast<T>(0);
  }
  return true;
}

std::vector<KernelAttr> SvdCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SvdCpuKernelMod::SvdFunc> &pair) { return pair.first; });
  return support_list;
}

std::vector<std::pair<KernelAttr, SvdCpuKernelMod::SvdFunc>> SvdCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &SvdCpuKernelMod::LaunchKernelFloat<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &SvdCpuKernelMod::LaunchKernelFloat<double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeComplex64)
     .AddOutputAttr(kNumberTypeComplex64)
     .AddOutputAttr(kNumberTypeComplex64),
   &SvdCpuKernelMod::LaunchKernelComplex<float_complex>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeComplex128)
     .AddOutputAttr(kNumberTypeComplex128)
     .AddOutputAttr(kNumberTypeComplex128),
   &SvdCpuKernelMod::LaunchKernelComplex<double_complex>}};

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Svd, SvdCpuKernelMod);
}  // namespace svd_cpu
}  // namespace kernel
}  // namespace mindspore
