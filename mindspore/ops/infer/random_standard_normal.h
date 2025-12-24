/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_RANDOM_STANDARD_NORMAL_H_
#define MINDSPORE_CORE_OPS_RANDOM_STANDARD_NORMAL_H_
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameRandomStandardNormal = "RandomStandardNormal";
/// \brief RandomStandardNormal defined RandomStandardNormal operator prototype of lite.
class OPS_API RandomStandardNormal : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(RandomStandardNormal);
  /// \brief Constructor.
  RandomStandardNormal() : BaseOperator(kNameRandomStandardNormal) { InitIOName({"shape"}, {"output"}); }

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] seed Define random seed.
  /// \param[in] seed2 Define random seed2.
  void Init(int64_t seed, int64_t seed2);

  /// \brief Method to set seed attributes.
  ///
  /// \param[in] seed Define random seed.
  void set_seed(int64_t seed);

  /// \brief Method to set seed2 attributes.
  ///
  /// \param[in] seed2 Define random seed2.
  void set_seed2(int64_t seed2);

  /// \brief Method to get seed attributes.
  ///
  /// \return seed attributes.
  int64_t get_seed() const;

  /// \brief Method to get seed2 attributes.
  ///
  /// \return seed2 attributes.
  int64_t get_seed2() const;
};

OPS_API abstract::AbstractBasePtr RandomStandardNormalInfer(const abstract::AnalysisEnginePtr &,
                                                            const PrimitivePtr &primitive,
                                                            const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RANDOM_STANDARD_NORMAL_H_
