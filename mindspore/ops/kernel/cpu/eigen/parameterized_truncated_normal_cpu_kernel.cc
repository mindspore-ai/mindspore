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

#include "kernel/cpu/eigen/parameterized_truncated_normal_cpu_kernel.h"
#include <cmath>
#include <ctime>
#include <random>
#include <vector>
#include <map>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "kernel/cpu/eigen/eigen_common_utils.h"

#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/philox_random.h"
#include "mindspore/ops/infer/parameterized_truncated_normal.h"

namespace mindspore {
namespace kernel {
namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 5;
const size_t kInput0 = 0;
const size_t kInput1 = 1;
const size_t kInput2 = 2;
const size_t kInput3 = 3;
const size_t kInput4 = 4;
const size_t kOutputData = 0;
static constexpr int kMaxIterations = 1000;
const std::map<TypeId, size_t> means_type_size_map = {
  {kNumberTypeFloat16, sizeof(float16)}, {kNumberTypeFloat32, sizeof(float)}, {kNumberTypeFloat64, sizeof(double)}};
}  // namespace

bool ParameterizedTruncatedNormalCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  uint64_t seed = static_cast<uint64_t>(GetValue<int64_t>(primitive_->GetAttr("seed")));
  uint64_t seed2 = static_cast<uint64_t>(GetValue<int64_t>(primitive_->GetAttr("seed2")));
  uint64_t init_seed = random::GetSeed(seed, seed2);
  rng_.seed(init_seed);
  input_type_ = inputs[kInput0]->dtype_id();
  input_means_type_ = inputs[kInput1]->dtype_id();
  input_stdevs_type_ = inputs[kInput2]->dtype_id();
  input_min_type_ = inputs[kInput3]->dtype_id();
  input_max_type_ = inputs[kInput4]->dtype_id();
  output_type_ = outputs[kOutputData]->dtype_id();
  return MatchKernelFunc(kernel_name_, inputs, outputs);
}

int ParameterizedTruncatedNormalCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                                     const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[kInput0]->GetDeviceShapeVector();
  input_means_shape_ = inputs[kInput1]->GetDeviceShapeVector();
  input_stdevs_shape_ = inputs[kInput2]->GetDeviceShapeVector();
  input_min_shape_ = inputs[kInput3]->GetDeviceShapeVector();
  input_max_shape_ = inputs[kInput4]->GetDeviceShapeVector();
  return KRET_OK;
}

template <typename T>
T ParameterizedTruncatedNormalCpuKernelMod::GetBatchSizeCheckDims(const std::vector<KernelTensor *> &inputs) {
  auto output_shape = GetDeviceAddress<T>(inputs, 0);
  return output_shape[0];
}

template <typename T_shape, typename T>
bool ParameterizedTruncatedNormalCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                            const std::vector<KernelTensor *> &,
                                                            const std::vector<KernelTensor *> &outputs) {
  auto output_shape = GetDeviceAddress<T_shape>(inputs, 0);
  size_t input_shape_num = inputs[0]->size() / sizeof(T_shape);
  // check shape
  MS_EXCEPTION_IF_NULL(output_shape);
  auto batch_size = output_shape[0];
  int sample_size = 1;
  for (size_t i = 1; i < input_shape_num; i++) {
    sample_size *= output_shape[i];
  }

  auto output_data = GetDeviceAddress<T>(outputs, 0);
  auto means = GetDeviceAddress<T>(inputs, kInput1);
  auto stdevs = GetDeviceAddress<T>(inputs, kInput2);
  auto minvals = GetDeviceAddress<T>(inputs, kInput3);
  auto maxvals = GetDeviceAddress<T>(inputs, kInput4);

  std::vector<T *> params = {means, stdevs, minvals, maxvals};

  auto input_means_num = input_means_shape_.size();
  auto input_stdevs_num = input_stdevs_shape_.size();
  auto input_mix_num = input_min_shape_.size();
  auto input_max_num = input_max_shape_.size();
  std::vector<size_t> params_idx;
  if (input_means_num > 1) {
    params_idx.push_back(kInput0);
  }
  if (input_stdevs_num > 1) {
    params_idx.push_back(kInput1);
  }
  if (input_mix_num > 1) {
    params_idx.push_back(kInput2);
  }
  if (input_max_num > 1) {
    params_idx.push_back(kInput3);
  }

  for (int batch = 0; batch < batch_size; batch++) {
    auto maxval = *params[3];
    auto minval = *params[2];
    auto stddev = *params[1];
    if (stddev <= static_cast<T>(0)) {
      MS_LOG(EXCEPTION) << "For ParameterizedTruncatedNormal, stdevs value must be greater than 0 in each batch.";
    }
    if (maxval < minval) {
      MS_LOG(EXCEPTION) << "For ParameterizedTruncatedNormal, max value must be greater than min value in each batch.";
    }
    Generate<T>(int64_t(sample_size), *params[0], *params[1], minval, maxval, &output_data);
    for (auto i : params_idx) {
      params[i] = params[i] + 1;
    }
  }
  return true;
}

template <typename T>
void ParameterizedTruncatedNormalCpuKernelMod::Generate(const int64_t size, const T mean, T stddev, T minval, T maxval,
                                                        T **output_ptr) {
  const T stddev_inside_bound = T(1.3);
  if ((isinf(minval) && minval < T(0)) || maxval < mean) {
    T tmp = minval;
    minval = maxval;
    maxval = tmp;
    stddev = -stddev;
  }
  auto tmp_num = (stddev == static_cast<T>(0)) ? static_cast<T>(1) : stddev;
  // Calculate normalized samples, then convert them.
  const T norm_min = (minval - mean) / tmp_num;
  const T norm_max = (maxval - mean) / tmp_num;
  const T sqrt_factor = sqrt((norm_min * norm_min) + T(4));
  const T cutoff = T(2) * exp(T(0.5) + (norm_min * (norm_min - sqrt_factor)) / T(4)) / (norm_min + sqrt_factor);
  const T diff = norm_max - norm_min;
  if (((norm_min < -stddev_inside_bound) && (norm_max >= T(0))) ||
      ((norm_max > stddev_inside_bound) && (norm_min <= T(0)))) {
    GenerateCase1(size, norm_min, norm_max, stddev, mean, output_ptr);
  } else if (diff < cutoff) {
    // Sample from a uniform distribution on [norm_min, norm_max].
    GenerateCase2(size, norm_min, norm_max, stddev, mean, output_ptr);
  } else {
    GenerateCase3(size, norm_min, norm_max, stddev, mean, output_ptr);
  }
  return;
}

template <typename T>
void ParameterizedTruncatedNormalCpuKernelMod::GenerateCase1(const int64_t size, const T norm_min, const T norm_max,
                                                             const T stddev, const T mean, T **output_ptr) {
  auto output = *output_ptr;
  std::normal_distribution<double> normal_dist(0, 1);
  int sample_num = 0;
  while (sample_num < size) {
    for (int iter = 0; iter <= kMaxIterations;) {
      T normal_sample = static_cast<T>(normal_dist(rng_));
      if ((normal_sample >= norm_min) && (normal_sample <= norm_max)) {
        *output = normal_sample * stddev + mean;
        if (stddev <= static_cast<T>(0)) {
          *output = static_cast<T>(INFINITY);
        } else {
          output = output + 1;
        }
        sample_num++;
        break;
      } else {
        iter++;
        if (iter > kMaxIterations) {
          *output_ptr = output;
          MS_LOG(EXCEPTION) << "For ParameterizedTruncatedNormal, randn rejection sampler exceeded maximum iterations.";
        }
      }
    }
  }
  *output_ptr = output;
  return;
}

template <typename T>
void ParameterizedTruncatedNormalCpuKernelMod::GenerateCase2(const int64_t size, const T norm_min, const T norm_max,
                                                             const T stddev, const T mean, T **output_ptr) {
  auto output = *output_ptr;
  std::uniform_real_distribution<double> unifrom_dist(0, 1);
  int sample_num = 0;
  const T diff = norm_max - norm_min;
  const T plus_Factor = (norm_min < T(0)) ? T(0) : norm_min * norm_min;
  while (sample_num < size) {
    for (int iter = 0; iter <= kMaxIterations;) {
      T uniform_sample = T(unifrom_dist(rng_));
      T z = uniform_sample * diff + norm_min;
      T g = (plus_Factor - z * z) / T(2.0);
      bool accept = static_cast<T>(unifrom_dist(rng_)) <= exp(g);
      if (accept || iter + 1 >= kMaxIterations) {
        if (!accept) {
          *output_ptr = output;
          MS_LOG(EXCEPTION) << "For ParameterizedTruncatedNormal, "
                            << "uniform rejection sampler, exceeded max iterations. Sample may contain outliers.";
        }
        *output = z * stddev + mean;
        if (stddev <= static_cast<T>(0)) {
          *output = static_cast<T>(INFINITY);
        } else {
          output = output + 1;
        }
        sample_num++;
        break;
      } else {
        iter++;
      }
    }
  }
  *output_ptr = output;
  return;
}

template <typename T>
void ParameterizedTruncatedNormalCpuKernelMod::GenerateCase3(const int64_t size, const T norm_min, const T norm_max,
                                                             const T stddev, const T mean, T **output_ptr) {
  auto output = *output_ptr;
  std::uniform_real_distribution<double> unifrom_dist(0, 1);
  int sample_num = 0;
  const T alpha = (norm_min + sqrt((norm_min * norm_min) + T(4))) / T(2);
  while (sample_num < size) {
    for (int iter = 0; iter <= kMaxIterations;) {
      T uniform_sample = T(unifrom_dist(rng_));
      T z = -log(uniform_sample) / alpha + norm_min;
      const T x = norm_min < alpha ? alpha - z : norm_min - alpha;
      const T g = exp(-x * x / T(2.0));
      const T u = T(unifrom_dist(rng_));
      bool accept = (u <= g && z < norm_max);
      if (accept || iter + 1 >= kMaxIterations) {
        if (!accept) {
          *output_ptr = output;
          MS_LOG(EXCEPTION) << "For ParameterizedTruncatedNormal, "
                            << "exponential distribution rejection sampler exceeds max iterations. "
                            << "Sample may contain outliers.";
        }
        *output = z * stddev + mean;
        output = output + 1;
        sample_num++;
        break;
      } else {
        iter++;
      }
    }
  }
  *output_ptr = output;
  return;
}

const std::vector<std::pair<KernelAttr, ParameterizedTruncatedNormalCpuKernelMod::KernelRunFunc>> &
ParameterizedTruncatedNormalCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, ParameterizedTruncatedNormalCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &ParameterizedTruncatedNormalCpuKernelMod::LaunchKernel<int32_t, float16>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ParameterizedTruncatedNormalCpuKernelMod::LaunchKernel<int32_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &ParameterizedTruncatedNormalCpuKernelMod::LaunchKernel<int32_t, double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &ParameterizedTruncatedNormalCpuKernelMod::LaunchKernel<int64_t, float16>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ParameterizedTruncatedNormalCpuKernelMod::LaunchKernel<int64_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &ParameterizedTruncatedNormalCpuKernelMod::LaunchKernel<int64_t, double>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ParameterizedTruncatedNormal, ParameterizedTruncatedNormalCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
