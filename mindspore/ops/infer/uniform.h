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

#ifndef MINDSPORE_CORE_OPS_UNIFORM_H_
#define MINDSPORE_CORE_OPS_UNIFORM_H_
#include <memory>
#include <set>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameUniform = "Uniform";
constexpr auto kFrom = "from";
constexpr auto kTo = "to";
/// \brief Produces random floating-point values i, uniformly distributed to the interval [0, 1).
/// Refer to Python API @ref mindspore.ops.Uniform for more details.

class OPS_API Uniform : public BaseOperator {
 public:
  Uniform() : BaseOperator(kNameUniform) { InitIOName({"x"}, {"y"}); }
  /// \brief Method to init the ops attributes.
  void Init(const float from, const float to, const int64_t seed, const int64_t offset);
  /// \brief Set from.
  void set_from(const float from);
  /// \brief Set to.
  void set_to(const float to);
  /// \brief Set seed.
  void set_seed(const int64_t seed);
  /// \brief Set offset.
  void set_offset(const int64_t offset);
  /// \brief Get from.
  ///
  /// \return from.
  float get_from() const;
  /// \brief Get to.
  ///
  /// \return to.
  float get_to() const;
  /// \brief Get seed.
  ///
  /// \return seed.
  int64_t get_seed() const;
  /// \brief Get offset.
  ///
  /// \return offset.
  int64_t get_offset() const;

  MIND_API_BASE_MEMBER(Uniform);
};

OPS_API abstract::AbstractBasePtr UniformInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_UNIFORM_H_
