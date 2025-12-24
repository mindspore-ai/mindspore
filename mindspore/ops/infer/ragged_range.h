/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_RAGGEDRANGE_H_
#define MINDSPORE_CORE_OPS_RAGGEDRANGE_H_
#include <algorithm>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameRaggedRange = "RaggedRange";
/// \brief RaggedRange operator prototype.
class OPS_API RaggedRange : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(RaggedRange);
  /// \brief Constructor
  RaggedRange() : BaseOperator(kNameRaggedRange) {
    InitIOName({"starts", "limits", "deltas"}, {"rt_nested_splits", "rt_dense_values"});
  }
  /// \brief Method to init the op.
  void Init() const {}
};
OPS_API abstract::AbstractBasePtr RaggedRangeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimRaggedRangePtr = std::shared_ptr<RaggedRange>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RAGGED_RANGE_H_
