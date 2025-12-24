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

#ifndef MINDSPORE_CORE_OPS_COL2IM_H_
#define MINDSPORE_CORE_OPS_COL2IM_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameCol2Im = "Col2Im";
/// \brief Col2Im operation. Refer to Python API @ref mindspore.ops.Col2Im for more details.
class OPS_API Col2Im : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Col2Im);

  /// \brief Constructor.
  Col2Im() : BaseOperator(kNameCol2Im) { InitIOName({"x", "output_size"}, {"y"}); }
};

OPS_API abstract::AbstractBasePtr Col2ImInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args);
using PrimCol2ImPtr = std::shared_ptr<Col2Im>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_COL2IM_H_
