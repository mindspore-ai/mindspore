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

#ifndef MINDSPORE_CORE_OPS_RANDOM_NORMAL_H_
#define MINDSPORE_CORE_OPS_RANDOM_NORMAL_H_
#include <string>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameRandomNormal = "RandomNormal";
/// \brief RandomNormal defined RandomNormal operator prototype of lite.
class MIND_API RandomNormal : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(RandomNormal);
  /// \brief Constructor.
  RandomNormal() : BaseOperator(kNameRandomNormal) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] seed Define random seed.
  /// \param[in] mean Define random mean.
  /// \param[in] scale Define random standard deviation.
  void Init(float seed, float mean, float scale);

  /// \brief Method to set seed attributes.
  ///
  /// \param[in] seed Define random seed.
  void set_seed(float seed);

  /// \brief Method to set mean attributes.
  ///
  /// \param[in] mean Define random mean.
  void set_mean(float mean);

  /// \brief Method to set scale attributes.
  ///
  /// \param[in] scale Define random standard deviation.
  void set_scale(float scale);

  /// \brief Method to get seed attributes.
  ///
  /// \return seed attributes.
  float get_seed() const;

  /// \brief Method to get mean attributes.
  ///
  /// \return mean attributes.
  float get_mean() const;

  /// \brief Method to get scale attributes.
  ///
  /// \return scale attributes.
  float get_scale() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RANDOM_NORMAL_H_
