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
#include <ctime>
#include <random>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
bool GammaCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Gamma>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "cast RandomGamma ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  seed_ = kernel_ptr->get_seed();
  seed2_ = kernel_ptr->get_seed2();
  generator_.Init(seed_, seed2_);

  output_shape_ = outputs[0]->GetShapeVector();
  alpha_shape_ = inputs[1]->GetShapeVector();
  alpha_dtype_ = inputs[1]->GetDtype();

  return true;
}

int GammaCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  auto input_shape_shape = inputs[0]->GetShapeVector();
  int shape_len = input_shape_shape.size();
  for (int i = 0; i < shape_len; i++) {
    if (input_shape_shape[i] != output_shape_[i]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' output size and input size mismatch.";
      return KRET_RESIZE_FAILED;
    }
  }
  return ret;
}

// T: float16 float32 float64 dtype of alpha, beta and output
template <typename T>
void GammaCpuKernelMod::Generate(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  const auto *alpha_flat = reinterpret_cast<T *>(inputs[1]->addr);
  auto *samples_flat = reinterpret_cast<T *>(outputs[0]->addr);

  int64_t num_samples = std::accumulate(output_shape_.begin(), output_shape_.end(), 0);
  if (num_samples == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' the sizes of output is zero.";
  }

  using random::PhiloxRandom;
  typedef random::NormalDistribution<random::PhiloxRandom, double> Normal;
  typedef random::UniformDistribution<random::PhiloxRandom, double> Uniform;
#define UNIFORM(X)                                    \
  if (uniform_remaining == 0) {                       \
    uniform_remaining = Uniform::kResultElementCount; \
    uniform_result = uniform(&gen);                   \
  }                                                   \
  uniform_remaining--;                                \
  double X = uniform_result[uniform_remaining]

  // Each attempt is 95+% successful, and requires 1-2 normal + 1 uniform
  static constexpr int kReservedSamplesPerOutput = 256;

  int64 num_alphas = std::accumulate(alpha_shape_.begin(), alpha_shape_.end(), 0);

  PhiloxRandom rng = generator_.ReserveRandomOutputs(num_samples * num_alphas, kReservedSamplesPerOutput);

  // We partition work first across alphas then across samples-per-alpha to
  // avoid a couple flops which can be done on a per-alpha basis.
  auto DoWork = [num_samples, num_alphas, &rng, samples_flat, alpha_flat](int64 start_output, int64 limit_output) {
    using Eigen::numext::exp;
    using Eigen::numext::log;
    using Eigen::numext::pow;

    // Capturing "rng" by-value would only make a copy for the _shared_
    // lambda.  Since we want to let each worker have its own copy, we pass
    // "rng" by reference and explicitly do a copy assignment.

    Normal normal;
    Uniform uniform;
    typename Normal::ResultType norm_result;
    typename Uniform::ResultType uniform_result;
    for (int64 output_idx = start_output; output_idx < limit_output;
         /* output_idx incremented within inner loop below */) {
      int64 alpha_idx = output_idx / num_samples;

      // Instead of +alpha_idx for each sample, we offset the pointer once.
      T *const samples_alpha_offset = samples_flat + alpha_idx;

      // Several calculations can be done on a per-alpha basis.
      const double alpha = static_cast<double>(alpha_flat[alpha_idx]);

      //      DISABLE_FLOAT_EQUALITY_WARNING
      if (alpha == static_cast<double>(1.0)) {
        //        ENABLE_FLOAT_EQUALITY_WARNING
        // Sample from an exponential distribution.
        for (int64 sample_idx = output_idx % num_samples; sample_idx < num_samples && output_idx < limit_output;
             sample_idx++, output_idx++) {
          // As we want data stable regardless of sharding
          // (including eventually on GPU), we skip on a per-sample basis.
          PhiloxRandom gen = rng;
          gen.Skip(kReservedSamplesPerOutput * output_idx);
          int16 uniform_remaining = 0;
          UNIFORM(u);
          const double res = -log(1.0 - u);
          samples_alpha_offset[sample_idx * num_alphas] = static_cast<T>(res);
        }       // for (sample_idx)
      } else {  // if alpha != 1.0
        // Transformation-rejection from pairs of uniform and normal random
        // variables. http://dl.acm.org/citation.cfm?id=358414
        //
        // The algorithm has an acceptance rate of ~95% for small alpha (~1),
        // and higher accept rates for higher alpha, so runtime is
        // O(NumAlphas * NumSamples * k) with k ~ 1 / 0.95.
        //
        // For alpha<1, we add one to d=alpha-1/3, and multiply the final
        // result by uniform()^(1/alpha)
        const bool alpha_less_than_one = alpha < 1;
        const double d = alpha + (alpha_less_than_one ? 2.0 / 3 : -1.0 / 3);
        const double c = 1.0 / 3 / sqrt(d);

        // Compute the rest of the samples for the current alpha value.
        for (int64 sample_idx = output_idx % num_samples; sample_idx < num_samples && output_idx < limit_output;
             sample_idx++, output_idx++) {
          // Since each sample may use a variable number of normal/uniform
          // samples, and we want data stable regardless of sharding
          // (including eventually on GPU), we skip on a per-sample basis.
          PhiloxRandom gen = rng;
          gen.Skip(kReservedSamplesPerOutput * output_idx);
          int16 norm_remaining = 0;
          int16 uniform_remaining = 0;

          // Keep trying until we don't reject a sample. In practice, we will
          // only reject ~5% at worst, for low alpha near 1.
          while (true) {
            if (norm_remaining == 0) {
              norm_remaining = Normal::kResultElementCount;
              norm_result = normal(&gen);
            }
            norm_remaining--;
            const double x = norm_result[norm_remaining];
            double v = 1 + c * x;
            if (v <= 0) {
              continue;
            }
            v = v * v * v;
            UNIFORM(u);
            // The first option in the if is a "squeeze" short-circuit to
            // dodge the two logs. Magic constant sourced from the paper
            // linked above. Upward of .91 of the area covered by the log
            // inequality is covered by the squeeze as well (larger coverage
            // for smaller values of alpha).
            if ((u < 1 - 0.0331 * (x * x) * (x * x)) || (log(u) < 0.5 * x * x + d * (1 - v + log(v)))) {
              double res = d * v;
              if (alpha_less_than_one) {
                UNIFORM(b);
                res *= pow(b, 1 / alpha);
              }
              samples_alpha_offset[sample_idx * num_alphas] = static_cast<T>(res);
              break;
            }
          }  // while: true
        }    // for: sample_idx
      }      // if (alpha == 1.0)
    }        // for: output_idx
  };         // DoWork
#undef UNIFORM
  ParallelLaunchAutoSearch(DoWork, num_alphas * num_samples, this, &parallel_search_info_);
}

bool GammaCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                               const std::vector<AddressPtr> &outputs) {
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
  std::vector<KernelAttr> support_list = {KernelAttr()
                                            .AddInputAttr(kNumberTypeInt32)
                                            .AddInputAttr(kNumberTypeFloat16)
                                            .AddInputAttr(kNumberTypeFloat16)
                                            .AddOutputAttr(kNumberTypeFloat16),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeInt32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddOutputAttr(kNumberTypeFloat32),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeInt32)
                                            .AddInputAttr(kNumberTypeFloat64)
                                            .AddInputAttr(kNumberTypeFloat64)
                                            .AddOutputAttr(kNumberTypeFloat64),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddInputAttr(kNumberTypeFloat16)
                                            .AddInputAttr(kNumberTypeFloat16)
                                            .AddOutputAttr(kNumberTypeFloat16),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddOutputAttr(kNumberTypeFloat32),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddInputAttr(kNumberTypeFloat64)
                                            .AddInputAttr(kNumberTypeFloat64)
                                            .AddOutputAttr(kNumberTypeFloat64)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Gamma, GammaCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
