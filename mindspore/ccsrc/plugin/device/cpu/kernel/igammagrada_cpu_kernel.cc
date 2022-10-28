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
#include "plugin/device/cpu/kernel/igammagrada_cpu_kernel.h"
#include <limits>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
namespace mindspore {
namespace kernel {
namespace {
/**
 * Coefficients for the Lanczos approximation of the gamma function. The
 * coefficients are uniquely determined by the choice of g and n (kLanczosGamma
 * and kLanczosCoefficients.size() + 1). The coefficients below correspond to
 * [7, 9]. [5, 7], [7, 9], [9, 10], and [607/128.0, 15] were evaluated and [7,
 * 9] seemed to be the least sensitive to the quality of the log function. In
 * particular, [5, 7] is the only choice where -1.5e-5 <= lgamma(2) <= 1.5e-5
 * for a particularly inaccurate log function.
 * */
static constexpr double kLanczosGamma = 7;  // aka g
constexpr size_t kInputIndex0 = 0;
constexpr size_t kInputIndex1 = 1;
constexpr size_t kInputIndex2 = 2;
constexpr size_t kInputIndex3 = 3;
constexpr size_t kInputIndex4 = 4;
constexpr size_t kInputIndex5 = 5;
constexpr size_t kInputIndex6 = 6;
constexpr size_t kInputIndex7 = 7;
constexpr size_t kInputIndex8 = 8;
constexpr size_t kInputIndex9 = 9;
constexpr size_t kInputIndex10 = 10;
constexpr size_t kInputIndex11 = 11;
constexpr size_t kInputIndex12 = 12;
constexpr size_t kInputIndex13 = 13;
constexpr size_t kInputIndex14 = 14;
static constexpr double kBaseLanczosCoeff = 0.99999999999980993227684700473478;
static constexpr double M_pi = 3.141592653589793238462643383279;
static constexpr std::array<double, 8> kLanczosCoefficients = {
  676.520368121885098567009190444019, -1259.13921672240287047156078755283,
  771.3234287776530788486528258894,   -176.61502916214059906584551354,
  12.507343278686904814458936853,     -0.13857109526572011689554707,
  9.984369578019570859563e-6,         1.50563273514931155834e-7};
const double log_lanczos_gamma_plus_one_half = std::log(kLanczosGamma + 0.5);
constexpr int64_t kParallelDataNums = 256;
constexpr int64_t kSameShape = 0;
constexpr int64_t kXOneElement = 1;
constexpr int64_t kYOneElement = 2;
constexpr size_t kInputNum = 2;
constexpr size_t kOutputNum = 1;
constexpr int64_t VALUE = 1;
constexpr int64_t DERIVATIVE = 2;
constexpr int64_t kInputsNum = 2;
constexpr int64_t kOutputsNum = 1;
size_t get_element_num(const std::vector<int64_t> &shape) {
  size_t size = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    size *= static_cast<size_t>(shape[i]);
  }
  return size;
}
}  // namespace
/** Compute the Lgamma function using Lanczos' approximation from "A Precision
 * Approximation of the Gamma Function". SIAM Journal on Numerical Analysis
 * series B. Vol. 1:
 * lgamma(z + 1) = (log(2) + log(pi)) / 2 + (z + 1/2) * log(t(z)) - t(z) + A(z)
 * t(z) = z + kLanczosGamma + 1/2
 * A(z) = kBaseLanczosCoeff + sigma(k = 1, n, kLanczosCoefficients[i] / (z + k))
 */
template <typename T>
T Lgamma(const T &input) {
  T log_pi = std::log(M_pi);
  T log_sqrt_two_pi = (std::log(2) + std::log(M_pi)) / 2;

  /** If the input is less than 0.5 use Euler's reflection formula:
   * gamma(x) = pi / (sin(pi * x) * gamma(1 - x))
   */
  bool need_to_reflect = (input < 0.5);
  T input_after_reflect = need_to_reflect ? -input : input - 1;
  T sum = kBaseLanczosCoeff;
  for (size_t i = 0, end = kLanczosCoefficients.size(); i < end; ++i) {
    T lanczos_coefficient = kLanczosCoefficients[i];

    sum += lanczos_coefficient / (input_after_reflect + i + 1);
  }
  T gamma_plus_onehalf_plus_z = kLanczosGamma + 0.5 + input_after_reflect;
  T log_t = log_lanczos_gamma_plus_one_half + std::log1pf(input_after_reflect / (kLanczosGamma + 0.5));
  T log_y = log_sqrt_two_pi + (input_after_reflect + 0.5 - gamma_plus_onehalf_plus_z / log_t) * log_t + std::log(sum);
  T abs_input = std::abs(input);
  T abs_frac_input = abs_input - std::floor(abs_input);

  T reduced_frac_input = (abs_frac_input > 0.5) ? 1 - abs_frac_input : abs_frac_input;
  T reflection_denom = std::log(std::sin(M_pi * reduced_frac_input));

  T reflection = std::isfinite(reflection_denom) ? log_pi - reflection_denom - log_y : -reflection_denom;
  T result = need_to_reflect ? reflection : log_y;

  return std::isinf(input) ? std::numeric_limits<T>::infinity() : result;
}

template <typename T>
T Digamma(const T &input) {
  bool need_to_reflect = (input < 0.5);
  T reflected_input = need_to_reflect ? -input : input - 1;

  T num = 0;
  T denom = kBaseLanczosCoeff;

  for (size_t i = 0, end = kLanczosCoefficients.size(); i < end; ++i) {
    T lanczos_coefficient = kLanczosCoefficients[i];
    num -= lanczos_coefficient / ((reflected_input + i + 1) * (reflected_input + i + 1));
    denom += lanczos_coefficient / (reflected_input + i + 1);
  }

  T gamma_plus_onehalf_plus_z = kLanczosGamma + 0.5 + reflected_input;
  T log_t = log_lanczos_gamma_plus_one_half + std::log1pf(reflected_input / (kLanczosGamma + 0.5));

  T result = log_t + num / denom - kLanczosGamma / gamma_plus_onehalf_plus_z;

  T reduced_input = input + std::abs(std::floor(input + 0.5));
  T reflection = result - M_pi * std::cos(M_pi * reduced_input) / std::sin(M_pi * reduced_input);
  T real_result = need_to_reflect ? reflection : result;

  // Digamma has poles at negative integers and zero; return nan for those.
  return (input < 0 && input == std::floor(input)) ? std::numeric_limits<T>::quiet_NaN() : real_result;
}

template <typename T>
T use_igammact(const T &ax, const T &a, const T &x, T enabled, int mode) {
  T y = 1 - a;
  T z = x + y + 1;
  T c = 0;
  T pkm2 = 1;
  T qkm2 = x;
  T pkm1 = x + 1;
  T qkm1 = z * x;
  T ans = pkm1 / qkm1;
  T t = 1;
  T dpkm2_da = 0;
  T dqkm2_da = 0;
  T dpkm1_da = 0;
  T dqkm1_da = -x;
  T dans_da = (dpkm1_da - ans * dqkm1_da) / qkm1;
  std::vector<T> vals = {enabled, ans,  t,        y,        z,        c,        pkm1,   qkm1,
                         pkm2,    qkm2, dpkm2_da, dqkm2_da, dpkm1_da, dqkm1_da, dans_da};
  constexpr int k2000 = 2000;
  while (vals[kInputIndex0] && vals[kInputIndex5] < k2000) {
    enabled = vals[kInputIndex0];
    ans = vals[kInputIndex1];
    T tmp_var_t = vals[kInputIndex2];
    T tmp_var_y = vals[kInputIndex3];
    T tmp_var_z = vals[kInputIndex4];
    T tmp_var_c = vals[kInputIndex5];
    pkm1 = vals[kInputIndex6];
    qkm1 = vals[kInputIndex7];
    pkm2 = vals[kInputIndex8];
    qkm2 = vals[kInputIndex9];
    dpkm2_da = vals[kInputIndex10];
    dqkm2_da = vals[kInputIndex11];
    dpkm1_da = vals[kInputIndex12];
    dqkm1_da = vals[kInputIndex13];
    dans_da = vals[kInputIndex14];
    tmp_var_c += 1;
    tmp_var_y += 1;
    constexpr int TWO = 2;
    tmp_var_z += TWO;
    T yc = tmp_var_y * tmp_var_c;
    T pk = pkm1 * tmp_var_z - pkm2 * yc;
    T qk = qkm1 * tmp_var_z - qkm2 * yc;
    bool qk_is_nonzero = (qk != 0);
    T r = pk / qk;
    ans = qk_is_nonzero ? r : ans;
    T dpk_da = dpkm1_da * tmp_var_z - pkm1 - dpkm2_da * yc + pkm2 * tmp_var_c;
    T dqk_da = dqkm1_da * tmp_var_z - qkm1 - dqkm2_da * yc + qkm2 * tmp_var_c;
    T dans_da_new = qk_is_nonzero ? (dpk_da - ans * dqk_da) / qk : dans_da;
    T grad_conditional = qk_is_nonzero ? std::abs(dans_da_new - dans_da) : 1;
    pkm2 = pkm1;
    pkm1 = pk;
    qkm2 = qkm1;
    qkm1 = qk;
    dpkm2_da = dpkm1_da;
    dqkm2_da = dqkm1_da;
    dpkm1_da = dpk_da;
    dqkm1_da = dqk_da;
    bool rescale = std::abs(pk) > (1 / std::numeric_limits<T>::epsilon());
    pkm2 = rescale ? pkm2 * std::numeric_limits<T>::epsilon() : pkm2;
    pkm1 = rescale ? pkm1 * std::numeric_limits<T>::epsilon() : pkm1;
    qkm2 = rescale ? qkm2 * std::numeric_limits<T>::epsilon() : qkm2;
    qkm1 = rescale ? qkm1 * std::numeric_limits<T>::epsilon() : qkm1;
    dpkm2_da = rescale ? dpkm2_da * std::numeric_limits<T>::epsilon() : dpkm2_da;
    dqkm2_da = rescale ? dqkm2_da * std::numeric_limits<T>::epsilon() : dqkm2_da;
    dpkm1_da = rescale ? dpkm1_da * std::numeric_limits<T>::epsilon() : dpkm1_da;
    dqkm1_da = rescale ? dqkm1_da * std::numeric_limits<T>::epsilon() : dqkm1_da;
    T conditional = enabled && (grad_conditional > std::numeric_limits<T>::epsilon());
    vals[kInputIndex0] = conditional;
    vals[kInputIndex5] = tmp_var_c;
    if (enabled) {
      vals = {conditional, ans,  tmp_var_t, tmp_var_y, tmp_var_z, tmp_var_c, pkm1,       qkm1,
              pkm2,        qkm2, dpkm2_da,  dqkm2_da,  dpkm1_da,  dqkm1_da,  dans_da_new};
    }
  }
  ans = vals[kInputIndex1];
  if (mode == VALUE) {
    return ans * ax;
  }
  dans_da = vals[kInputIndex14];
  T dlogax_da = std::log(x) - Digamma<T>(a);
  switch (mode) {
    case DERIVATIVE:
      return ax * (ans * dlogax_da + dans_da);
    default:
      return -(dans_da + ans * dlogax_da) * x;
  }
}

template <typename T>
T use_igammacf(const T &ax, const T &a, T x, T enabled) {
  std::vector<T> vals = {enabled, a, 1, 1, x, 0, 0};
  while (vals[kInputIndex0] != 0) {
    enabled = vals[kInputIndex0];
    T r = vals[kInputIndex1];
    T c = vals[kInputIndex2];
    T ans = vals[kInputIndex3];
    x = vals[kInputIndex4];
    T dc_da = vals[kInputIndex5];
    T dans_da = vals[kInputIndex6];
    r += 1;
    dc_da = dc_da * (x / r) + (-1 * c * x) / (r * r);
    dans_da = dans_da + dc_da;
    c = c * (x / r);
    ans = ans + c;
    T conditional = enabled && (std::abs(dc_da / dans_da) > std::numeric_limits<T>::epsilon());
    vals[kInputIndex0] = conditional;
    if (enabled) {
      vals = {conditional, r, c, ans, x, dc_da, dans_da};
    }
  }
  T ans = vals[kInputIndex3];
  T dans_da = vals[kInputIndex6];
  if (a == 0) {
    return NAN;
  }
  T dlogax_da = std::log(x) - Digamma<T>(a + 1);
  return ax * (ans * dlogax_da + dans_da) / a;
}

template <typename T>
T IgammaGradASingle(const T &a, const T &x) {
  bool is_nan = std::isnan(a) || std::isnan(x);
  bool x_is_zero = (x == 0);
  bool domain_error = (x < 0) || (a <= 0);
  bool use_igammac = (x > 1) && (x > a);
  T ax = a * std::log(x) - x - Lgamma<T>(a);
  bool underflow = (ax < -std::log(std::numeric_limits<T>::max()));
  ax = std::exp(ax);
  T enabled = static_cast<T>(!(x_is_zero || domain_error || underflow || is_nan));
  T output;
  if (use_igammac != 0) {
    enabled = static_cast<T>(enabled && use_igammac);
    output = -use_igammact(ax, a, x, enabled, DERIVATIVE);
  } else {
    enabled = static_cast<T>(enabled && !(use_igammac));
    output = use_igammacf(ax, a, x, enabled);
  }
  output = (domain_error || is_nan || std::isnan(output)) ? std::numeric_limits<double>::quiet_NaN() : output;
  output = x_is_zero || (std::isinf(x) && !is_nan && !domain_error && !std::isinf(a)) ? 0 : output;
  return output;
}

template <typename T>
void IgammaGradACpuKernelMod::BcastCompute(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  auto a_data_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto x_data_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto z_data_addr = reinterpret_cast<T *>(outputs[0]->addr);
  size_t data_num = get_element_num(z_shape_);
  auto output_shape = CPUKernelUtils::GetBroadcastShape(a_shape_, x_shape_);
  BroadcastIterator iter(a_shape_, x_shape_, output_shape);
  if (data_num < kParallelDataNums) {
    iter.SetPos(0);
    for (size_t i = 0; i < data_num; i++) {
      T *a_index = a_data_addr + iter.GetInputPosA();  // i-th value of input0
      T *x_index = x_data_addr + iter.GetInputPosB();  // i-th value of input1
      *(z_data_addr + i) = IgammaGradASingle<T>(*a_index, *x_index);
      iter.GenNextPos();
    }
  } else {
    auto shard_igammaGradA = [z_data_addr, a_data_addr, x_data_addr, &iter](size_t start, size_t end) {
      iter.SetPos(start);
      for (size_t i = start; i < end; i++) {
        T *a_index = a_data_addr + iter.GetInputPosA();  // i-th value of input0
        T *x_index = x_data_addr + iter.GetInputPosB();  // i-th value of input1
        *(z_data_addr + i) = IgammaGradASingle<T>(*a_index, *x_index);
        iter.GenNextPos();
      }
    };
    ParallelLaunchAutoSearch(shard_igammaGradA, data_num, this, &parallel_search_info_);
  }
}

/* special compute is used in the following situations.
 * 1. the shapes of input1 and input2 are the same
 * 2. input1 is a 1D tensor with only one element or input1 is scalar
 * 3. input2 is a 1D tensor with only one element or input2 is scalar
 * 4. the shapes of input1 and input2 are different
 **/
template <typename T>
void IgammaGradACpuKernelMod::SpecialCompute(int64_t type, int64_t start, int64_t end, const T *input1, const T *input2,
                                             T *output) {
  switch (type) {
    case kSameShape: {
      auto cur_input1 = input1 + start;
      auto cur_input2 = input2 + start;
      for (int64_t i = start; i < end; ++i) {
        *output = IgammaGradASingle<T>(*cur_input1, *cur_input2);
        output = output + 1;
        cur_input1 = cur_input1 + 1;
        cur_input2 = cur_input2 + 1;
      }
      break;
    }
    case kXOneElement: {
      auto cur_input2 = input2 + start;
      for (int64_t i = start; i < end; ++i) {
        *output = IgammaGradASingle<T>(*input1, *cur_input2);
        output = output + 1;
        cur_input2 = cur_input2 + 1;
      }
      break;
    }
    case kYOneElement: {
      auto cur_input1 = input1 + start;
      for (int64_t i = start; i < end; ++i) {
        *output = IgammaGradASingle<T>(*cur_input1, *input2);
        output = output + 1;
        cur_input1 = cur_input1 + 1;
      }
      break;
    }
    default:
      break;
  }
}

template <typename T>
void IgammaGradACpuKernelMod::NoBcastCompute(const std::vector<kernel::AddressPtr> &inputs,
                                             const std::vector<kernel::AddressPtr> &outputs) {
  auto in0 = reinterpret_cast<T *>(inputs[0]->addr);
  auto in1 = reinterpret_cast<T *>(inputs[1]->addr);
  auto out0 = reinterpret_cast<T *>(outputs[0]->addr);
  auto in0_elements_nums = get_element_num(a_shape_);
  auto in1_elements_nums = get_element_num(x_shape_);
  auto data_num = get_element_num(z_shape_);
  int64_t type =
    in0_elements_nums == in1_elements_nums ? kSameShape : (in0_elements_nums == 1 ? kXOneElement : kYOneElement);
  if (data_num < kParallelDataNums) {
    SpecialCompute<T>(type, 0, data_num, in0, in1, out0);
  } else {
    auto shard_igammaGradA = [type, in0, in1, out0, this](int64_t start, int64_t end) {
      SpecialCompute<T>(type, start, end, in0, in1, out0 + start);
    };
    ParallelLaunchAutoSearch(shard_igammaGradA, data_num, this, &parallel_search_info_);
  }
}

bool IgammaGradACpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();
  constexpr size_t input_num = kInputsNum;
  constexpr size_t output_num = kOutputsNum;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  dtype_ = inputs[0]->GetDtype();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  return true;
}

int IgammaGradACpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  a_shape_ = inputs[0]->GetDeviceShapeAdaptively();
  x_shape_ = inputs[1]->GetDeviceShapeAdaptively();
  z_shape_ = outputs[0]->GetDeviceShapeAdaptively();
  return ret;
}

bool IgammaGradACpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'var' should be float32 or float64, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  return true;
}

template <typename T>
void IgammaGradACpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  size_t in0_elements_nums = get_element_num(a_shape_);
  size_t in1_elements_nums = get_element_num(x_shape_);
  bool isNeedBcast = (a_shape_ == x_shape_) || (in0_elements_nums == 1) || (in1_elements_nums == 1);
  if (isNeedBcast) {
    NoBcastCompute<T>(inputs, outputs);
  } else {
    BcastCompute<T>(inputs, outputs);
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IgammaGradA, IgammaGradACpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
