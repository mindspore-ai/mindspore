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
#ifndef MINDSPORE_CORE_OPS_RANDOM_CHOICE_WITH_MASK_H_
#define MINDSPORE_CORE_OPS_RANDOM_CHOICE_WITH_MASK_H_
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kRandomChoiceWithMask = "RandomChoiceWithMask";
class OPS_API RandomChoiceWithMask : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(RandomChoiceWithMask);
  RandomChoiceWithMask() : BaseOperator(kRandomChoiceWithMask) { InitIOName({"input_x"}, {"y", "mask"}); }

  void Init() const {}

  /// \brief Set seed.
  void set_seed(const int64_t seed);

  /// \brief Set seed2.
  void set_seed2(const int64_t seed2);

  /// \brief Set count.
  void set_count(const int64_t count);

  /// \brief Method to get seed attributes.
  ///
  /// \return seed attributes.
  int64_t get_seed() const;

  /// \brief Method to get seed2 attributes.
  ///
  /// \return seed2 attributes.
  int64_t get_seed2() const;

  /// \brief Method to get count attributes.
  ///
  /// \return count attributes.
  int64_t get_count() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RANDOM_CHOICE_WITH_MASK_H_
