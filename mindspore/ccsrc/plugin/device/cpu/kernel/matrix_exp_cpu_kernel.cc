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

#include "plugin/device/cpu/kernel/matrix_exp_cpu_kernel.h"
#include "mindspore/core/ops/matrix_exp.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr uint32_t kMatrixExpInputsNum = 1;
constexpr uint32_t kMatrixExpOutputsNum = 1;
constexpr uint32_t kIndexTwo = 2;

// Coefficients for computing taylor approximant of order 8.
constexpr double sqrt_177 = 0.1330413469565007072504e+2, x3 = 2. / 3.;
constexpr double x1 = x3 * ((1. + sqrt_177) / 88.), x2 = x3 * ((1. + sqrt_177) / 352.);
constexpr double x4 = (-271. + 29. * sqrt_177) / (315. * x3), x5 = (-11. + 11. * sqrt_177) / (1260. * x3);
constexpr double x6 = (-99. + 11. * sqrt_177) / (5040. * x3), x7 = (89. - sqrt_177) / (5040. * x3);
constexpr double y2 = (857. - 58. * sqrt_177) / 630.;
const std::vector<std::vector<double>> b18 = {
  {0., -1.00365581030144618291e-01, -8.02924648241156932449e-03, -8.92138498045729985177e-04, 0.},
  {0., 3.97849749499645077844e-01, 1.36783778460411720168e+00, 4.98289622525382669416e-01, -6.37898194594723280150e-04},
  {-1.09676396052962061844e+01, 1.68015813878906206114e+00, 5.71779846478865511061e-02, -6.98210122488052056106e-03,
   3.34975017086070470649e-05},
  {-9.04316832390810593223e-02, -6.76404519071381882256e-02, 6.75961301770459654925e-02, 2.95552570429315521194e-02,
   -1.39180257516060693404e-05},
  {0., 0., -9.23364619367118555360e-02, -1.69364939002081722752e-02, -1.40086798182036094347e-05}};
}  // namespace

bool MatrixExpCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::MatrixExp>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast MatrixExp ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  data_type_ = inputs.at(kIndex0)->GetDtype();
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int MatrixExpCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &others) {
  int ret = 0;
  if ((ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs, others)) != 0) {
    return ret;
  }
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_), LongToSize);
  return 0;
}

template <typename Derived>
void MatrixExpCpuKernelMod::MTaylorApproximantLow(const Eigen::MatrixBase<Derived> &A,
                                                  const Eigen::MatrixBase<Derived> &I, int order,
                                                  Eigen::MatrixBase<Derived> *matrix_y) const {
  constexpr int exp_order_1 = 1, exp_order_2 = 2, exp_order_4 = 4;
  auto A2 = A * A;
  if (order == exp_order_1) {
    *matrix_y = I + A;
  } else if (order == exp_order_2) {
    constexpr int A2_divisor = 2;
    *matrix_y = I + A + A2 / A2_divisor;
  } else if (order == exp_order_4) {
    constexpr int I_divisor = 2, A_divisor = 6, A2_divisor = 24;
    *matrix_y = I + A + A2 * (I / I_divisor + A / A_divisor + A2 / A2_divisor);
  } else {
    auto A4 = A2 * (x1 * A + x2 * A2);
    auto A8 = (x3 * A2 + A4) * (x4 * I + x5 * A + x6 * A2 + x7 * A4);
    *matrix_y = I + A + y2 * A2 + A8;
  }
}
template <typename Derived, typename Derived1>
void MatrixExpCpuKernelMod::MTaylorApproximantHigh(const Eigen::MatrixBase<Derived1> &A_scaled,
                                                   const Eigen::MatrixBase<Derived> &I,
                                                   Eigen::MatrixBase<Derived> *matrix_y) const {
  auto A2 = A_scaled * A_scaled;
  auto A3 = A_scaled * A2;
  auto A6 = A3 * A3;
  auto q31 = b18[0][0] * I + b18[0][1] * A_scaled + b18[0][2] * A2 + b18[0][3] * A3 + b18[0][4] * A6;
  auto q61 = b18[1][0] * I + b18[1][1] * A_scaled + b18[1][2] * A2 + b18[1][3] * A3 + b18[1][4] * A6;
  auto q62 = b18[2][0] * I + b18[2][1] * A_scaled + b18[2][2] * A2 + b18[2][3] * A3 + b18[2][4] * A6;
  auto q63 = b18[3][0] * I + b18[3][1] * A_scaled + b18[3][2] * A2 + b18[3][3] * A3 + b18[3][4] * A6;
  auto q64 = b18[4][0] * I + b18[4][1] * A_scaled + b18[4][2] * A2 + b18[4][3] * A3 + b18[4][4] * A6;
  auto q91 = q31 * q64 + q63;
  *matrix_y = q61 + (q62 + q91) * q91;
}

template <typename Derived>
void MatrixExpCpuKernelMod::MexpImpl(const Eigen::MatrixBase<Derived> &A, const Eigen::MatrixBase<Derived> &I,
                                     Eigen::MatrixBase<Derived> *matrix_y) const {
  const auto norm = A.cwiseAbs().colwise().sum().maxCoeff();
  std::vector<int> m_vals = {1, 2, 4, 8};
  std::vector<double> thetas;
  if (data_type_ == kNumberTypeFloat16 || data_type_ == kNumberTypeFloat || data_type_ == kNumberTypeComplex64) {
    thetas = {1.192092800768788e-07, 5.978858893805233e-04, 5.116619363445086e-02, 5.800524627688768e-01,
              3.010066362817634e+00};
  } else {
    thetas = {2.220446049250313e-16, 2.580956802971767e-08, 3.397168839976962e-04, 4.991228871115323e-02,
              1.090863719290036e+00};
  }
  for (size_t i = 0; i < thetas.size() - 1; i++) {
    if (norm <= thetas[i]) {
      MTaylorApproximantLow(A, I, m_vals[i], matrix_y);
      return;
    }
  }
  int64_t s = ceil(log2(norm / thetas.back())) > 0 ? ceil(log2(norm / thetas.back())) : 0;
  const auto pow2s = pow(2, s);
  const auto A_scaled = A / pow2s;
  MTaylorApproximantHigh(A_scaled, I, matrix_y);
  for (int k = 0; k < s; k++) {
    *matrix_y *= (*matrix_y);
  }
}

template <typename T>
bool MatrixExpCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &workspace,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMatrixExpInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMatrixExpOutputsNum, kernel_name_);
  auto input_x = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_y = reinterpret_cast<T *>(outputs[0]->addr);
  int64_t m = SizeToLong(*(input_shape_.end() - 1));
  int64_t size_mm = m * m;
  MatrixXd<T> I(m, m);
  Eigen::Map<MatrixXd<T>> map_I(I.data(), m, m);
  (void)I.setIdentity();
  int64_t total = SizeToLong(inputs[0]->size / sizeof(T));
  int64_t matrix_num = total / size_mm;
  auto task = [this, &input_x, &output_y, &map_I, m](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      Eigen::Map<MatrixXd<T>> matrix_x(input_x + i * m * m, m, m);
      Eigen::Map<MatrixXd<T>> matrix_y(output_y + i * m * m, m, m);
      if (matrix_x.size() > 0) {
        MexpImpl(matrix_x, map_I, &matrix_y);
      }
    }
  };
  ParallelLaunchAutoSearch(task, matrix_num, this, &parallel_search_info_);
  return true;
}

const std::vector<std::pair<KernelAttr, MatrixExpCpuKernelMod::KernelRunFunc>> &MatrixExpCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, MatrixExpCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &MatrixExpCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &MatrixExpCpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &MatrixExpCpuKernelMod::LaunchKernel<std::complex<float>>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &MatrixExpCpuKernelMod::LaunchKernel<std::complex<double>>}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MatrixExp, MatrixExpCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
