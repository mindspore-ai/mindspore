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

#ifndef MINDSPORE_CORE_OPS_MULTINOMIAL_WITH_REPLACEMENT_H_
#define MINDSPORE_CORE_OPS_MULTINOMIAL_WITH_REPLACEMENT_H_
#include <memory>
#include <vector>
#include "abstract/abstract_value.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMultinomialWithReplacement = "MultinomialWithReplacement";
class OPS_API MultinomialWithReplacement : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(MultinomialWithReplacement);
  MultinomialWithReplacement() : BaseOperator(kNameMultinomialWithReplacement) {
    InitIOName({"x", "seed", "offset"}, {"y"});
  }
  void Init(int64_t numsamples, bool replacement = false);
  void set_numsamples(int64_t numsamples);
  int64_t get_numsamples() const;
  void set_replacement(bool replacement);
  bool get_replacement() const;
};
AbstractBasePtr MultinomialWithReplacementInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args);
using PrimMultinomialWithReplacementPtr = std::shared_ptr<MultinomialWithReplacement>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MULTINOMIAL_WITH_REPLACEMENT_H_
