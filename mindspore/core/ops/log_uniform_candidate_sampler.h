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

#ifndef MINDSPORE_CORE_OPS_RANDOM_LOG_UNIFORM_CANDIDATE_SAMPLER_H_
#define MINDSPORE_CORE_OPS_RANDOM_LOG_UNIFORM_CANDIDATE_SAMPLER_H_
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/base_operator.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLogUniformCandidateSampler = "LogUniformCandidateSampler";
/// \brief Generates random labels with a log-uniform distribution for sampled_candidates.
/// Refer to Python API @ref mindspore.ops.log_uniform_candidate_sampler for more details.
class MIND_API LogUniformCandidateSampler : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(LogUniformCandidateSampler);
  /// \brief Constructor.
  LogUniformCandidateSampler() : BaseOperator(kNameLogUniformCandidateSampler) {
    InitIOName({"true_classes"}, {"sampled_candidates", "true_expected_count", "sampled_expected_count"});
  }

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] num_true The number of target classes per training example.
  /// \param[in] num_sampled The number of classes to randomly sample.
  /// \param[bool] unique Determines whether sample with rejection. If `unique` is True, all sampled classes in a batch
  /// are unique.
  /// \param[in] range_max The number of possible classes. When `unique` is True, `range_max` must be greater than or
  /// equal to `num_sampled`.
  /// \param[in] seed Random seed, must be non-negative.
  void Init(int64_t num_true = 1, int64_t num_sampled = 5, bool unique = true, int64_t range_max = 5, int64_t seed = 0);

  inline void set_num_true(int64_t num_true) { (void)this->AddAttr(kNumTrue, api::MakeValue(num_true)); }

  inline int64_t get_num_true() { return GetValue<int64_t>(GetAttr(kNumTrue)); }

  inline void set_num_sampled(int64_t num_sampled) { (void)this->AddAttr(kNumSampled, api::MakeValue(num_sampled)); }

  inline int64_t get_num_sampled() { return GetValue<int64_t>(GetAttr(kNumSampled)); }

  inline void set_unique(bool unique) { (void)this->AddAttr(kUnique, api::MakeValue(unique)); }

  inline bool get_unique() { return GetValue<bool>(GetAttr(kUnique)); }

  inline void set_range_max(int64_t range_max) { (void)this->AddAttr(kRangeMax, api::MakeValue(range_max)); }

  inline int64_t get_range_max() { return GetValue<int64_t>(GetAttr(kRangeMax)); }

  inline void set_seed(int64_t seed) { (void)this->AddAttr(kSeed, api::MakeValue(seed)); }

  inline int64_t get_seed() { return GetValue<int64_t>(GetAttr(kSeed)); }
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RANDOM_LOG_UNIFORM_CANDIDATE_SAMPLER_H_
