/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/log_uniform_candidate_sampler_cpu_kernel.h"
#include <cmath>
#include <map>
#include <utility>
#include <algorithm>
#include <unordered_set>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/log_uniform_candidate_sampler.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace kernel {
bool LogUniformCandidateSamplerCpuKernel::Init(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs) {
  auto op = std::dynamic_pointer_cast<ops::LogUniformCandidateSampler>(base_operator);
  if (op == nullptr) {
    MS_LOG(ERROR) << "cast op LogUniformCandidateSampler failed!";
    return false;
  }
  this->num_true_ = op->get_num_true();
  this->num_sampled_ = op->get_num_sampled();
  this->unique_ = op->get_unique();
  this->seed_ = op->get_seed();
  this->range_max_ = op->get_range_max();
  this->log_range_ = log1p(range_max_);
  if (unique_ && range_max_ < num_sampled_) {
    MS_LOG(ERROR) << "When unique is True, range_max must be greater than or equal to num_sampled";
    return false;
  }
  int64_t seed = 87654321;
  int64_t seed2 = seed_;
  generator_.Init(seed, seed2);
  reserveSamplesNr_ = 2048 * num_sampled_;
  return true;
}

static float CalcExpectedCount(float p, int num_sampled, int num_tries) {
  if (num_tries == num_sampled) {
    return p * num_sampled;
  }
  return -std::expm1(num_tries * std::log1p(-p));
}

float LogUniformCandidateSamplerCpuKernel::Probability(int64_t value) const {
  return (log((value + 2.0) / (value + 1.0))) / log_range_;
}

int64_t LogUniformCandidateSamplerCpuKernel::Sample(random::SinglePhiloxRandom *single) {
  double d = single->GenDouble();
  int64_t val = static_cast<int64_t>(exp(d * log_range_)) - 1;
  return val % range_max_;
}

int LogUniformCandidateSamplerCpuKernel::Resize(const BaseOperatorPtr &base_operator,
                                                const std::vector<KernelTensorPtr> &inputs,
                                                const std::vector<KernelTensorPtr> &outputs,
                                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KRET_OK;
  if ((ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    return ret;
  }
  auto true_classes_shape = input_shapes_.at(0);
  if (true_classes_shape.size() != 2) {
    MS_LOG(ERROR) << "input true_classes dims should be 2.";
    return KRET_RESIZE_FAILED;
  }

  if (true_classes_shape[1] != num_true_) {
    MS_LOG(ERROR) << "input true_classes dim[1] should equal to num_true, true_classes.dim[1] = "
                  << true_classes_shape[1] << ", num_true = " << num_true_;
    return KRET_RESIZE_FAILED;
  }

  auto sampled_candidates_shape = output_shapes_.at(0);
  if (sampled_candidates_shape.size() != 1 || sampled_candidates_shape[0] != static_cast<int64_t>(num_sampled_)) {
    MS_LOG(ERROR) << "output sampled_candidates shape should equal to (num_sampled, ), sampled_candidates shape = "
                  << VectorToString(sampled_candidates_shape) << ", num_sampled_ = " << num_sampled_;
    return KRET_RESIZE_FAILED;
  }

  auto true_expected_count_shape = output_shapes_.at(1);
  if (true_expected_count_shape != true_classes_shape) {
    MS_LOG(ERROR)
      << "output true_expected_count shape should be same with true_classes shape, true_expected_count shape = "
      << VectorToString(true_expected_count_shape) << ", true_classes shape = " << VectorToString(true_classes_shape);
    return KRET_RESIZE_FAILED;
  }

  auto sampled_expected_count_shape = output_shapes_.at(2);
  if (sampled_expected_count_shape.size() != 1 ||
      sampled_expected_count_shape[0] != static_cast<int64_t>(num_sampled_)) {
    MS_LOG(ERROR)
      << "output sampled_expected_count shape shape should equal to (num_sampled, ), sampled_expected_count shape = "
      << VectorToString(sampled_candidates_shape) << ", num_sampled_ = " << num_sampled_;
    return KRET_RESIZE_FAILED;
  }
  return ret;
}
bool LogUniformCandidateSamplerCpuKernel::Launch(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &workspace,
                                                 const std::vector<AddressPtr> &outputs) {
  int64_t *true_classes = static_cast<int64_t *>(inputs.at(0)->addr);
  auto true_classes_size = input_size_list_.at(0);
  int64_t *sampled_candidates = static_cast<int64_t *>(outputs.at(0)->addr);
  float *true_expected_count = static_cast<float *>(outputs.at(1)->addr);
  float *sampled_expected_count = static_cast<float *>(outputs.at(2)->addr);

  auto gen = generator_.ReserveSamples32(reserveSamplesNr_);

  random::SinglePhiloxRandom single(&gen);

  int num_tries = 0;
  if (unique_) {
    std::unordered_set<int64_t> used(num_sampled_);
    int32_t idx = 0;
    while (idx < num_sampled_) {
      num_tries++;
      int64_t value = Sample(&single);
      if (used.find(value) == used.end()) {
        sampled_candidates[idx++] = value;
        used.emplace(value);
      }
    }
  } else {
    for (int32_t idx = 0; idx < num_sampled_; idx++) {
      sampled_candidates[idx] = Sample(&single);
    }
    num_tries = num_sampled_;
  }

  for (int32_t i = 0; i < num_sampled_; i++) {
    sampled_expected_count[i] = CalcExpectedCount(Probability(sampled_candidates[i]), num_sampled_, num_tries);
  }

  for (size_t i = 0; i < true_classes_size; i++) {
    true_expected_count[i] = CalcExpectedCount(Probability(true_classes[i]), num_sampled_, num_tries);
  }
  return true;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LogUniformCandidateSampler, LogUniformCandidateSamplerCpuKernel);
}  // namespace kernel
}  // namespace mindspore
