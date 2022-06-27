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
#ifndef MINDSPORE_CORE_OPS_UNIFORM_CANDIDATE_SAMPLER_H_
#define MINDSPORE_CORE_OPS_UNIFORM_CANDIDATE_SAMPLER_H_

#include <vector>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameUniformCandidateSampler = "UniformCandidateSampler";
/// \brief samples a set of classes(sampled_candidates) from a given range based on uniform distribution. candidates
/// are drawn with replacemen or not dependding on the input parameter `unique`.
/// Refer to Python API @ref mindspore.ops.UniformCandidateSampler for more details.
class MIND_API UniformCandidateSampler : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(UniformCandidateSampler);
  /// \brief Constructor.
  UniformCandidateSampler() : BaseOperator(kNameUniformCandidateSampler) {
    InitIOName({"true_classes "}, {"sampled_candidates", "true_expected_count", "sampled_expected_count"});
  }

  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.UniformCandidateSampler for the inputs.
  void Init(int64_t num_true, int64_t num_sampled, bool unique, int64_t range_max, int64_t seed = 0,
            bool remove_accidental_hits = false);
  /// \brief Set number of target classes
  void set_num_true(int64_t num_true);
  void set_num_sampled(int64_t num_sampled);
  void set_unique(bool unique);
  void set_range_max(int64_t range_max);
  void set_seed(int64_t seed);
  void set_remove_accidental_hits(bool remove_accidental_hits);

  /// \brief Get number of target classes
  int64_t get_num_true() const;
  int64_t get_num_sampled() const;
  bool get_unique() const;
  int64_t get_range_max() const;
  int64_t get_seed() const;
  bool get_remove_accidental_hits() const;
};

abstract::AbstractBasePtr UniformCandidateSamplerInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &,
                                                       const std::vector<abstract::AbstractBasePtr> &);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_UNIFORM_CANDIDATE_SAMPLER_H_
