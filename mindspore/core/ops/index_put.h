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

#ifndef MINDSPORE_CORE_OPS_INDEX_PUT_H_
#define MINDSPORE_CORE_OPS_INDEX_PUT_H_
#include <memory>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameIndexPut = "IndexPut";
/// \brief Adds values x2 to specified indices of tensor x1.
/// Refer to Python API @ref mindspore.ops.IndexPut for more details.
class MIND_API IndexPut : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(IndexPut);
  /// \brief Constructor.
  IndexPut() : BaseOperator(kNameIndexPut) { InitIOName({"x1", "x2", "indices"}, {"y"}); }
};

abstract::AbstractBasePtr IndexPutInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_INDEX_PUT_H_
