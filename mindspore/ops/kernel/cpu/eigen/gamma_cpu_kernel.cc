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

#include "kernel/cpu/eigen/gamma_cpu_kernel.h"
#include <cmath>
#include <random>
#include <functional>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
static constexpr size_t INPUT_NUM = 2;
static constexpr size_t OUTPUT_NUM = 1;
static constexpr int kReservedSamplesPerOutput = 256;
inline bool IsEqual(double x, double y) {
  const double epsilon = 1e-14;
  return std::abs(x - y) <= epsilon;
}
}  // namespace
bool GammaCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  int64_t seed = GetValue<int64_t>(primitive_->GetAttr(ops::kSeed));
  int64_t seed2 = GetValue<int64_t>(primitive_->GetAttr(ops::kSeed2));

  rng_.Init(seed, seed2);

  return true;
}

template <typename T>
void GammaCpuKernelMod::InferShape(const std::vector<KernelTensor *> &inputs) {
  const auto *shape_value = GetDeviceAddress<T>(inputs, 0);
  MS_EXCEPTION_IF_NULL(shape_value);
  for (int64_t i = 0; i < shape_shape_[0]; i++) {
    output_shape_.emplace_back(static_cast<int64_t>(shape_value[i]));
  }
  for (size_t i = 0; i < alpha_shape_.size(); i++) {
    output_shape_.emplace_back(alpha_shape_[i]);
  }
}

int GammaCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), INPUT_NUM, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), OUTPUT_NUM, kernel_name_);
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(inputs[1]);
  MS_EXCEPTION_IF_NULL(outputs[0]);
  alpha_shape_ = inputs[1]->GetShapeVector();
  alpha_dtype_ = inputs[1]->dtype_id();
  shape_dtype_ = inputs[0]->dtype_id();
  shape_shape_ = inputs[0]->GetShapeVector();
  output_shape_ = outputs[0]->GetShapeVector();

  return KRET_OK;
}

// T: float16 float32 float64 dtype of alpha, beta and output
template <typename T>
void GammaCpuKernelMod::Generate(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  const auto *alpha_flat = GetDeviceAddress<T>(inputs, 1);
  auto *samples_flat = GetDeviceAddress<T>(outputs, 0);

  int64_t num_samples =
    std::accumulate(output_shape_.begin(), output_shape_.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  if (num_samples == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' the sizes of output is zero.";
  }

  int64_t num_alphas =
    std::accumulate(alpha_shape_.begin(), alpha_shape_.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  if (num_alphas == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' the sizes of alpha is zero.";
  }
  int64_t samples_per_alpha = num_samples / num_alphas;

  random::PhiloxRandom rng = rng_.ReserveRandomOutputs(num_samples, kReservedSamplesPerOutput);

  CTask GammaTask = [this, samples_per_alpha, num_alphas, &rng, samples_flat, alpha_flat](int64_t start_output,
                                                                                          int64_t limit_output) {
    this->GenerateSamplesForRange<T>(start_output, limit_output, samples_per_alpha, num_alphas, rng, samples_flat,
                                     alpha_flat);
  };

  ParallelLaunchAutoSearch(GammaTask, static_cast<size_t>(num_alphas * samples_per_alpha), this,
                           &parallel_search_info_);
}

template <typename T>
void GammaCpuKernelMod::GenerateSamplesForRange(int64_t start_output, int64_t limit_output, int64_t samples_per_alpha,
                                                int64_t num_alphas, const random::PhiloxRandom &rng, T *samples_flat,
                                                const T *alpha_flat) {
  Normal normal;
  Uniform uniform;
  typename Normal::ResType norm_res;
  typename Uniform::ResType uniform_res;

  for (int64_t output_idx = start_output; output_idx < limit_output;) {
    int64_t alpha_idx = output_idx / samples_per_alpha;
    T *const samples_alpha_offset = samples_flat + alpha_idx;
    const double alpha_value = static_cast<double>(alpha_flat[alpha_idx]);

    if (IsEqual(alpha_value, 1.0)) {
      GenerateExponentialSamples<T>(&output_idx, limit_output, samples_per_alpha, num_alphas, rng, samples_alpha_offset,
                                    &uniform, &uniform_res);
    } else {
      GenerateGammaSamples<T>(&output_idx, limit_output, samples_per_alpha, num_alphas, rng, samples_alpha_offset,
                              alpha_value, &normal, &uniform, &norm_res, &uniform_res);
    }
  }
}

template <typename T>
void GammaCpuKernelMod::GenerateExponentialSamples(int64_t *output_idx, int64_t limit_output, int64_t samples_per_alpha,
                                                   int64_t num_alphas, const random::PhiloxRandom &rng,
                                                   T *samples_alpha_offset, Uniform *uniform,
                                                   typename Uniform::ResType *uniform_res) {
  using Eigen::numext::log;

  for (int64_t sample_idx = *output_idx % samples_per_alpha;
       sample_idx < samples_per_alpha && *output_idx < limit_output; sample_idx++, (*output_idx)++) {
    random::PhiloxRandom gen = rng;
    gen.Skip(static_cast<uint64_t>(kReservedSamplesPerOutput * (*output_idx)));
    int64_t uniform_remaining = 0;

    double u = GetNextUniformRandom(uniform, &gen, uniform_res, &uniform_remaining);
    const double res = -log(1.0 - u);
    samples_alpha_offset[sample_idx * num_alphas] = static_cast<T>(res);
  }
}

template <typename T>
void GammaCpuKernelMod::GenerateGammaSamples(int64_t *output_idx, int64_t limit_output, int64_t samples_per_alpha,
                                             int64_t num_alphas, const random::PhiloxRandom &rng,
                                             T *samples_alpha_offset, double alpha_value, Normal *normal,
                                             Uniform *uniform, typename Normal::ResType *norm_res,
                                             typename Uniform::ResType *uniform_res) {
  const bool alpha_less_than_one = alpha_value < 1;
  const double su = alpha_value + (alpha_less_than_one ? 2.0 / 3 : -1.0 / 3);
  const double cut = 1.0 / 3 / sqrt(su);

  for (int64_t sample_idx = *output_idx % samples_per_alpha;
       sample_idx < samples_per_alpha && *output_idx < limit_output; sample_idx++, (*output_idx)++) {
    random::PhiloxRandom gen = rng;
    gen.Skip(static_cast<uint64_t>(kReservedSamplesPerOutput * (*output_idx)));

    double res = GenerateSingleGammaSample(&gen, alpha_value, alpha_less_than_one, su, cut, normal, uniform, norm_res,
                                           uniform_res);
    samples_alpha_offset[sample_idx * num_alphas] = static_cast<T>(res);
  }
}

double GammaCpuKernelMod::GenerateSingleGammaSample(random::PhiloxRandom *gen, double alpha_value,
                                                    bool alpha_less_than_one, double su, double cut, Normal *normal,
                                                    Uniform *uniform, typename Normal::ResType *norm_res,
                                                    typename Uniform::ResType *uniform_res) {
  using Eigen::numext::log;
  using Eigen::numext::pow;

  int64_t norm_remaining = 0;
  int64_t uniform_remaining = 0;
  double u;
  double b;

  while (true) {
    if (norm_remaining == 0) {
      norm_remaining = Normal::kResultElementCount;
      *norm_res = (*normal)(gen);
    }
    norm_remaining--;
    const double x = (*norm_res)[norm_remaining];

    double v = 1 + cut * x;
    if (v <= 0) {
      continue;
    }
    v = v * v * v;

    u = GetNextUniformRandom(uniform, gen, uniform_res, &uniform_remaining);

    double u_max = 1 - 0.0331 * (x * x) * (x * x);
    double u_lmax = 0.5 * x * x + su * (1 - v + log(v));
    if ((u < u_max) || (log(u) < u_lmax)) {
      double res = su * v;
      if (alpha_less_than_one) {
        b = GetNextUniformRandom(uniform, gen, uniform_res, &uniform_remaining);
        res *= pow(b, 1 / alpha_value);
      }
      return res;
    }
  }
}

double GammaCpuKernelMod::GetNextUniformRandom(Uniform *uniform, random::PhiloxRandom *gen,
                                               typename Uniform::ResType *uniform_res, int64_t *uniform_remaining) {
  if (*uniform_remaining == 0) {
    *uniform_remaining = Uniform::kResultElementCount;
    *uniform_res = (*uniform)(gen);
  }
  --(*uniform_remaining);
  return (*uniform_res)[*uniform_remaining];
}

bool GammaCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                               const std::vector<KernelTensor *> &outputs) {
  output_shape_.clear();
  if (output_shape_.empty()) {
    if (shape_dtype_ == kNumberTypeInt32) {
      InferShape<int32_t>(inputs);
    } else if (shape_dtype_ == kNumberTypeInt64) {
      InferShape<int64_t>(inputs);
    }
    outputs[0]->SetShapeVector(output_shape_);
    auto ele_size = LongToSize(
      std::accumulate(output_shape_.begin(), output_shape_.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()));
    outputs[0]->set_size(ele_size * UnitSizeInBytes(outputs[0]->dtype_id()));
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' output size and input size mismatch.";
  }

  if (alpha_dtype_ == kNumberTypeFloat16) {
    Generate<float16>(inputs, outputs);
  } else if (alpha_dtype_ == kNumberTypeFloat32) {
    Generate<float>(inputs, outputs);
  } else if (alpha_dtype_ == kNumberTypeFloat64) {
    Generate<double>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "RandomGamma kernel data type [%s] not support." << TypeIdToType(alpha_dtype_)->ToString();
  }
  return true;
}

std::vector<KernelAttr> GammaCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RandomGamma, GammaCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
