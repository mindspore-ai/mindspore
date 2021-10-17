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

#ifndef MINDSPORE_CORE_OPS_DIAG_PART_H_
#define MINDSPORE_CORE_OPS_DIAG_PART_H_
#include <vector>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
/// \brief Extracts the diagonal part from given tensor.
/// Refer to Python API @ref mindspore.ops.DiagPart for more details.
class MS_CORE_API DiagPart : public PrimitiveC {
 public:
  /// \brief Constructor.
  DiagPart() : PrimitiveC(prim::kPrimDiagPart->name()) { InitIOName({"input_x"}, {"output"}); }
  /// \brief Destructor.
  ~DiagPart() = default;
  MS_DECLARE_PARENT(DiagPart, PrimitiveC);
};
AbstractBasePtr DiagPartInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args);
using PrimDiagPartPtr = std::shared_ptr<DiagPart>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_DIAG_PART_H_
