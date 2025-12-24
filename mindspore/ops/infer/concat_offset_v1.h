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

#ifndef MINDSPORE_CORE_OPS_CONCAT_OFFSET_V1_H_
#define MINDSPORE_CORE_OPS_CONCAT_OFFSET_V1_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameConcatOffsetV1 = "ConcatOffsetV1";
/// \brief Computes offsets of concat inputs within its output.
/// Refer to Python API @ref mindspore.ops.ConcatOffsetV1 for more details.
class OPS_API ConcatOffsetV1 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ConcatOffsetV1);
  /// \brief Constructor.
  ConcatOffsetV1() : BaseOperator(kNameConcatOffsetV1) { InitIOName({"axis", "x"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.ConcatOffsetV1 for the inputs.
  void Init() const {}
};
OPS_API abstract::AbstractBasePtr ConcatOffsetV1Infer(const abstract::AnalysisEnginePtr &,
                                                      const PrimitivePtr &primitive,
                                                      const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_CONCAT_OFFSET_V1_H_
