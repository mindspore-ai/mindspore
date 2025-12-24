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

#ifndef MINDSPORE_CORE_OPS_NTH_ELEMENT_H_
#define MINDSPORE_CORE_OPS_NTH_ELEMENT_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameNthElement = "NthElement";
/// \brief Finds values of the n-th order statistic for the last dimension.
/// Refer to Python API @ref mindspore.ops.NthElement for more details.
class OPS_API NthElement : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(NthElement);
  /// \brief Constructor.
  NthElement() : BaseOperator(kNameNthElement) { InitIOName({"input", "n"}, {"output"}); }
  void Init(const bool use_locking = false);

  /// \brief Set reverse.
  void set_reverse(const bool reverse);

  /// \brief Get reverse.
  ///
  /// \return reverse.
  bool get_reverse() const;
};
OPS_API abstract::AbstractBasePtr NthElementInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_FLATTEN_H_
