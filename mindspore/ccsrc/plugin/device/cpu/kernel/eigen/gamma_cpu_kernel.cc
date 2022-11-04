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

#include "plugin/device/cpu/kernel/eigen/gamma_cpu_kernel.h"
#include <cmath>
#include <random>
#include <functional>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
static constexpr size_t INPUT_NUM = 2;
static constexpr size_t OUTPUT_NUM = 1;
bool GammaCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::RandomGamma>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "cast RandomGamma ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  seed_ = kernel_ptr->get_seed();
  seed2_ = kernel_ptr->get_seed2();
  generator_.Init(seed_, seed2_);

  outputs_ = outputs;
  output_shape_ = outputs[0]->GetShapeVector();
  alpha_shape_ = inputs[1]->GetShapeVector();
  alpha_dtype_ = inputs[1]->GetDtype();
  shape_dtype_ = inputs[0]->GetDtype();
  shape_shape_ = inputs[0]->GetShapeVector();

  is_need_retrieve_output_shape_ = true;

  return true;
}

template <typename T>
void GammaCpuKernelMod::InferShape(const std::vector<AddressPtr> &inputs) {
  const auto *shape_value = reinterpret_cast<T *>(inputs[0]->addr);
  for (int64_t i = 0; i < shape_shape_[0]; i++) {
    output_shape_.emplace_back(static_cast<int64_t>(shape_value[i]));
  }
  for (size_t i = 0; i < alpha_shape_.size(); i++) {
    output_shape_.emplace_back(alpha_shape_[i]);
  }
}

int GammaCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), INPUT_NUM, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), OUTPUT_NUM, kernel_name_);
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    ret = KRET_UNKNOWN_OUT_SHAPE;
    return ret;
  }
  alpha_shape_ = inputs[1]->GetShapeVector();
  alpha_dtype_ = inputs[1]->GetDtype();
  shape_dtype_ = inputs[0]->GetDtype();
  shape_shape_ = inputs[0]->GetShapeVector();
  return KRET_OK;
}

// T: float16 float32 float64 dtype of alpha, beta and output
template <typename T>
void GammaCpuKernelMod::Generate(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  const auto *alpha_flat = reinterpret_cast<T *>(inputs[1]->addr);
  auto *samples_flat = reinterpret_cast<T *>(outputs[0]->addr);

  int64_t num_samples = std::accumulate(output_shape_.begin(), output_shape_.end(), 1, std::multiplies<int64_t>());
  if (num_samples == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' the sizes of output is zero.";
  }

  using random::MSNormalDistribution;
  using random::MSPhiloxRandom;
  using random::MSUniformDistribution;
  typedef MSNormalDistribution<MSPhiloxRandom, double> Normal;
  typedef MSUniformDistribution<MSPhiloxRandom, double> Uniform;
#define UNIFORM(X)                                    \
  if (uniform_remaining == 0) {                       \
    uniform_remaining = Uniform::kResultElementCount; \
    uniform_res = uniform(&gen);                      \
  }                                                   \
  uniform_remaining--;                                \
  double X = uniform_res[uniform_remaining]

  static constexpr int kReservedSamplesPerOutput = 256;

  int64_t num_alphas = std::accumulate(alpha_shape_.begin(), alpha_shape_.end(), 1, std::multiplies<int64_t>());
  int64_t sample_shape_per_al = 0;
  if (num_alphas == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' the sizes of alpha is zero.";
  } else {
    sample_shape_per_al = num_samples / num_alphas;
  }

  MSPhiloxRandom rng = generator_.ReserveRandomOutputs(num_samples, kReservedSamplesPerOutput);

  auto DoWork = [sample_shape_per_al, num_alphas, &rng, samples_flat, alpha_flat](int64_t start_output,
                                                                                  int64_t limit_output) {
    using Eigen::numext::exp;
    using Eigen::numext::log;
    using Eigen::numext::pow;

    Normal normal;
    Uniform uniform;
    typename Normal::ResType norm_res;
    typename Uniform::ResType uniform_res;

    for (int64_t output_idx = start_output; output_idx < limit_output;) {
      int64_t alpha_idx = output_idx / sample_shape_per_al;
      T *const samples_alpha_offset = samples_flat + alpha_idx;
      const double alpha_value = static_cast<double>(alpha_flat[alpha_idx]);

      //      DISABLE_FLOAT_EQUALITY_WARNING
      if (alpha_value == static_cast<double>(1.0)) {
        //        ENABLE_FLOAT_EQUALITY_WARNING
        // Sample from an exponential distribution.
        for (int64_t sample_idx = output_idx % sample_shape_per_al;
             sample_idx < sample_shape_per_al && output_idx < limit_output; sample_idx++, output_idx++) {
          MSPhiloxRandom gen = rng;
          gen.Skip(static_cast<uint64_t>(kReservedSamplesPerOutput * output_idx));
          int64_t uniform_remaining = 0;
          UNIFORM(u);
          const double res = -log(1.0 - u);
          samples_alpha_offset[sample_idx * num_alphas] = static_cast<T>(res);
        }
      } else {
        // Transformation-rejection from pairs of uniform and normal random
        // variables. http://dl.acm.org/citation.cfm?id=358414
        const bool alpha_less_than_one = alpha_value < 1;
        const double su = alpha_value + (alpha_less_than_one ? 2.0 / 3 : -1.0 / 3);
        const double cut = 1.0 / 3 / sqrt(su);

        // Compute the rest of the samples for the current alpha value.
        for (int64_t sample_idx = output_idx % sample_shape_per_al;
             sample_idx < sample_shape_per_al && output_idx < limit_output; sample_idx++, output_idx++) {
          MSPhiloxRandom gen = rng;
          gen.Skip(static_cast<uint64_t>(kReservedSamplesPerOutput * output_idx));
          size_t norm_remaining = 0;
          int16_t uniform_remaining = 0;

          while (true) {
            if (norm_remaining == 0) {
              norm_remaining = Normal::kResultElementCount;
              norm_res = normal(&gen);
            }
            norm_remaining--;
            const double x = norm_res[norm_remaining];
            double v = 1 + cut * x;
            if (v <= 0) {
              continue;
            }
            v = v * v * v;
            UNIFORM(u);

            double u_max = 1 - 0.0331 * (x * x) * (x * x);
            double u_lmax = 0.5 * x * x + su * (1 - v + log(v));
            if ((u < u_max) || (log(u) < u_lmax)) {
              double res = su * v;
              if (alpha_less_than_one) {
                UNIFORM(b);
                res *= pow(b, 1 / alpha_value);
              }
              samples_alpha_offset[sample_idx * num_alphas] = static_cast<T>(res);
              break;
            }
          }
        }
      }
    }
  };
#undef UNIFORM
  ParallelLaunchAutoSearch(DoWork, static_cast<size_t>(num_alphas * sample_shape_per_al), this, &parallel_search_info_);
}

bool GammaCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                               const std::vector<AddressPtr> &outputs) {
  output_shape_.clear();
  if (output_shape_.empty()) {
    if (shape_dtype_ == kNumberTypeInt32) {
      InferShape<int32_t>(inputs);
    } else if (shape_dtype_ == kNumberTypeInt64) {
      InferShape<int64_t>(inputs);
    }
    outputs_[0]->SetShapeVector(output_shape_);
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
