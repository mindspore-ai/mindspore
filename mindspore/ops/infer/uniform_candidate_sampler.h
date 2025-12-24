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
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameUniformCandidateSampler = "UniformCandidateSampler";
/// \brief samples a set of classes(sampled_candidates) from a given range based on uniform distribution. candidates
/// are drawn with replacemen or not dependding on the input parameter `unique`.
/// Refer to Python API @ref mindspore.ops.UniformCandidateSampler for more details.
class OPS_API UniformCandidateSampler : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(UniformCandidateSampler);
  /// \brief Constructor.
  UniformCandidateSampler() : BaseOperator(kNameUniformCandidateSampler) {
    InitIOName({"true_classes "}, {"sampled_candidates", "true_expected_count", "sampled_expected_count"});
  }

  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.UniformCandidateSampler for the inputs.
  ///
  /// \param[in] num_true The number of target classes in each training example.
  /// \param[in] num_sampled The number of classes to randomly sample.
  /// \param[in] unique Whether all sampled classes in a batch are unique.
  /// \param[in] range_max The number of possible classes, must be non-negative.
  /// \param[in] seed Used for random number generation, must be non-negative. If seed has a value of 0, the seed will
  ///            be replaced with a randomly generated value.
  /// \param[in] remove_accidental_hits Whether accidental hit is removed.
  void Init(int64_t num_true, int64_t num_sampled, bool unique, int64_t range_max, int64_t seed = 0,
            bool remove_accidental_hits = false);

  /// \brief Method to set number of target classes
  ///
  /// \param[in] num_true The number of target classes in each training example.
  void set_num_true(int64_t num_true);

  /// \brief Method to set number of classes to randomly sample.
  ///
  /// \param[in] num_sampled The number of classes to randomly sample.
  void set_num_sampled(int64_t num_sampled);

  /// \brief Method to set flag, whether all sampled classes in a batch are unique.
  ///
  /// \param[in] unique The flag of whether all sampled classes in a batch are unique.
  void set_unique(bool unique);

  /// \brief Method to set number of possible classes.
  ///
  /// \param[in] range_max The number of possible classes.
  void set_range_max(int64_t range_max);

  /// \brief Method to set seed, used for random number generation
  ///
  /// \param[in] seed The seed used for random number generation.
  void set_seed(int64_t seed);

  /// \brief Method to set flag, whether accidental hit is removed.
  ///
  /// \param[in] remove_accidental_hits The flag of whether accidental hit is removed.
  void set_remove_accidental_hits(bool remove_accidental_hits);

  /// \brief Method to get number of target classes
  int64_t get_num_true() const;

  /// \brief Method to get number of classes to randomly sample.
  int64_t get_num_sampled() const;

  /// \brief Method to get flag, whether all sampled classes in a batch are unique.
  bool get_unique() const;

  /// \brief Method to get number of possible classes.
  int64_t get_range_max() const;

  /// \brief Method to get seed used for random number generation.
  int64_t get_seed() const;

  /// \brief Method to get flag, whether accidental hit is removed.
  bool get_remove_accidental_hits() const;
};

OPS_API abstract::AbstractBasePtr UniformCandidateSamplerInfer(const abstract::AnalysisEnginePtr &,
                                                               const PrimitivePtr &,
                                                               const std::vector<abstract::AbstractBasePtr> &);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_UNIFORM_CANDIDATE_SAMPLER_H_
