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

#ifndef MINDSPORE_CORE_OPS_RANDOM_CHOICE_WITH_MASK_H
#define MINDSPORE_CORE_OPS_RANDOM_CHOICE_WITH_MASK_H

#include <vector>
#include <memory>
#include "ops/base_operator.h"
#include "mindapi/base/macros.h"

namespace mindspore {
namespace ops {
constexpr auto kNameRandomChoiceWithMask = "RandomChoiceWithMask";
/// \brief RandomChoiceWithMask operator prototype.
class MIND_API RandomChoiceWithMask : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(RandomChoiceWithMask);
  /// \brief Constructor
  RandomChoiceWithMask() : BaseOperator(kNameRandomChoiceWithMask) { InitIOName({"input_x"}, {"index", "mask"}); }
  /// \brief Method to init the op.
  void Init(const int64_t count = 256, const int64_t seed = 0, const int64_t seed2 = 0);
  /// \brief Method to set count.
  void set_count(const int64_t count);
  /// \brief Method to get count.
  int64_t get_count();
  /// \brief Method to set seed.
  void set_seed(const int64_t seed);
  /// \brief Method to get seed.
  int64_t get_seed();
  /// \brief Method to set seed2.
  void set_seed2(const int64_t seed2);
  /// \brief Method to get seed2.
  int64_t get_seed2();
};
abstract::AbstractBasePtr RandomChoiceWithMaskInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                    const std::vector<abstract::AbstractBasePtr> &input_args);
using RandomChoiceWithMaskPtr = std::shared_ptr<RandomChoiceWithMask>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RANDOM_CHOICE_WITH_MASK_H
