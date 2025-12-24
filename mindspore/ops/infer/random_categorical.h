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

#ifndef MINDSPORE_CORE_OPS_RANDOM_CATEGORICAL_H_
#define MINDSPORE_CORE_OPS_RANDOM_CATEGORICAL_H_
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameRandomCategorical = "RandomCategorical";
/// \brief RandomCategorical defined RandomCategorical operator prototype of lite.
class OPS_API RandomCategorical : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(RandomCategorical);
  /// \brief Constructor.
  RandomCategorical() : BaseOperator(kNameRandomCategorical) {
    InitIOName({"logits", "num_sample", "seed"}, {"output"});
  }

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] num_sample Define number of sample to be drawn. Only constant values is allowed.
  /// \param[in] seed Define random seed.
  void Init(int64_t num_sample, int64_t seed);

  /// \brief Method to set num_sample attributes.
  ///
  /// \param[in] num_sample Define number of sample to be drawn. Only constant values is allowed.
  void set_num_sample(int64_t num_sample);

  /// \brief Method to get num_sample attributes.
  ///
  /// \return num_sample attributes.
  int64_t get_num_sample() const;

  /// \brief Method to set seed attributes.
  ///
  /// \param[in] seed Define random seed.
  void set_seed(int64_t seed);

  /// \brief Method to get seed attributes.
  ///
  /// \return seed attributes.
  int64_t get_seed() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RANDOM_CATEGORICAL_H_
