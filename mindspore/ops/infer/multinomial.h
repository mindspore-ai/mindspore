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

#ifndef MINDSPORE_CORE_OPS_MULTINOMIAL_H_
#define MINDSPORE_CORE_OPS_MULTINOMIAL_H_
#include <memory>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMultinomial = "Multinomial";
/// \brief Returns a tensor with the same rows as `x`, each row has num_samples sampled indices.
/// Refer to Python API @ref mindspore.ops.Multinomial for more details.
class OPS_API Multinomial : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Multinomial);
  /// \brief Constructor.
  Multinomial() : BaseOperator(kNameMultinomial) { InitIOName({"x", "num_samples"}, {"output"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Multinomial for the inputs.
  void Init(const int64_t seed = 0, const int64_t seed2 = 0);

  /// \brief Set seed. Defaults to 0.
  void set_seed(const int64_t seed);
  /// \brief Get seed.
  ///
  /// \return seed.
  int64_t get_seed() const;

  /// \brief Set seed2. Defaults to 0.
  void set_seed2(const int64_t seed2);
  /// \brief Get seed2.
  ///
  /// \return seed2.
  int64_t get_seed2() const;
};

OPS_API abstract::AbstractBasePtr MultinomialInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args);
using PrimMultinomial = std::shared_ptr<Multinomial>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_MULTINOMIAL_H_
