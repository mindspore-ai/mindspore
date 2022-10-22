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

#include <cmath>
#include <map>
#include <string>
#include <limits>
#include "plugin/device/cpu/kernel/mvlgamma_grad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/grad/mvlgamma_grad.h"

namespace mindspore {
namespace kernel {
namespace {
/**
 * Coefficients for the Lanczos approximation of the gamma function. The
 * coefficients are uniquely determined by the choice of g and n (kLanczosGamma
 * and kLanczosCoefficients.size() + 1).
 * */
static constexpr double kLanczosGamma = 7;  // aka g
static constexpr double kBaseLanczosCoeff = 0.99999999999980993227684700473478;
static constexpr std::array<double, 8> kLanczosCoefficients = {
  676.520368121885098567009190444019, -1259.13921672240287047156078755283,
  771.3234287776530788486528258894,   -176.61502916214059906584551354,
  12.507343278686904814458936853,     -0.13857109526572011689554707,
  9.984369578019570859563e-6,         1.50563273514931155834e-7};
double log_lanczos_gamma_plus_one_half = std::log(kLanczosGamma + 0.5);
constexpr double PI = 3.14159265358979323846264338327950288;
constexpr int64_t kInputsNum = 2;
constexpr int64_t kOutputsNum = 1;
}  // namespace

bool MvlgammaGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  // get kernel attr
  auto kernel_ptr = std::dynamic_pointer_cast<ops::MvlgammaGrad>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  attr_p_ = kernel_ptr->get_p();
  if (attr_p_ < 1) {
    MS_LOG(ERROR) << "For " << kernel_name_ << ", the attr 'p' has to be greater than or equal to 1.";
    return false;
  }
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int MvlgammaGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[kIndex0]->GetDeviceShapeAdaptively();
  input_tensor_size_ = static_cast<int64_t>(SizeOf(input_shape_));
  return KRET_OK;
}

/* Compute the Digamma function using Lanczos' approximation from "A Precision
 * Approximation of the Gamma Function". SIAM Journal on Numerical Analysis
 * series B. Vol. 1:
 * digamma(z + 1) = log(t(z)) + A'(z) / A(z) - kLanczosGamma / t(z)
 * t(z) = z + kLanczosGamma + 1/2
 * A(z) = kBaseLanczosCoeff + sigma(k = 1, n, kLanczosCoefficients[i] / (z + k))
 * A'(z) = sigma(k = 1, n, kLanczosCoefficients[i] / (z + k) / (z + k))
 */
template <typename T>
T MvlgammaGradCpuKernelMod::Digamma(const T &input) const {
  /* If the input is less than 0.5 use Euler's reflection formula:
   * digamma(x) = digamma(1 - x) - pi * cot(pi * x)
   */
  bool need_to_reflect = (input < 0.5);
  T reflected_input = static_cast<T>(need_to_reflect ? -input : input - 1);

  T num = 0;
  T denom = kBaseLanczosCoeff;

  for (size_t i = 0, end = kLanczosCoefficients.size(); i < end; ++i) {
    num -= static_cast<T>(kLanczosCoefficients[i] / ((reflected_input + i + 1) * (reflected_input + i + 1)));
    denom += static_cast<T>(kLanczosCoefficients[i] / (reflected_input + i + 1));
  }

  /* To improve accuracy on platforms with less-precise log implementations,
   * compute log(lanczos_gamma_plus_one_half) at compile time and use log1p on
   * the device.
   * log(t) = log(kLanczosGamma + 0.5 + z)
   *        = log(kLanczosGamma + 0.5) + log1p(z / (kLanczosGamma + 0.5))
   */

  T gamma_plus_onehalf_plus_z = static_cast<T>(kLanczosGamma + 0.5 + reflected_input);
  T log_t = static_cast<T>(log_lanczos_gamma_plus_one_half + std::log1pf(reflected_input / (kLanczosGamma + 0.5)));
  T result = static_cast<T>(log_t + num / denom - kLanczosGamma / gamma_plus_onehalf_plus_z);

  /* When we compute cot(pi * input), it should be careful that
   * pi * input can lose precision for near-integral values of `input`.
   * We shift values smaller than -0.5 into the range [-.5, .5] to
   * increase precision of pi * input and the resulting cotangent.
   */

  T reduced_input = static_cast<T>(input + std::abs(std::floor(input + 0.5)));
  T reflection = static_cast<T>(result - PI * std::cos(PI * reduced_input) / std::sin(PI * reduced_input));
  T real_result = static_cast<T>(need_to_reflect ? reflection : result);
  bool is_equal = false;
  if constexpr (std::is_same_v<T, float>) {
    is_equal = common::IsFloatEqual(input, std::floor(input));
  } else {
    is_equal = common::IsDoubleEqual(input, std::floor(input));
  }
  // Digamma has poles at negative integers and zero; return nan for those.
  return (input < static_cast<T>(0) && is_equal) ? std::numeric_limits<T>::quiet_NaN() : real_result;
}

template <typename T>
T MvlgammaGradCpuKernelMod::MvlgammaGradSingle(const T &y_grad, const T &x, const int64_t &p) const {
  T output = 0;
  const T HALF = static_cast<T>(0.5);
  for (int64_t i = 0; i < p; i++) {
    output += Digamma(x - HALF * static_cast<T>(i));
  }
  output *= y_grad;
  return output;
}

template <typename T>
bool MvlgammaGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  auto input_y_grad = static_cast<T *>(inputs[0]->addr);
  auto input_x = static_cast<T *>(inputs[1]->addr);
  auto output_x_grad = static_cast<T *>(outputs[0]->addr);

  for (int64_t i = 0; i < input_tensor_size_; i++) {
    *(output_x_grad + i) = MvlgammaGradSingle<T>(*(input_y_grad + i), *(input_x + i), attr_p_);
  }
  return true;
}

const std::vector<std::pair<KernelAttr, MvlgammaGradCpuKernelMod::KernelRunFunc>>
  &MvlgammaGradCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, MvlgammaGradCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &MvlgammaGradCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &MvlgammaGradCpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MvlgammaGrad, MvlgammaGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
