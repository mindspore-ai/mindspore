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

#ifndef MINDSPORE_CORE_OPS_DIAG_H_
#define MINDSPORE_CORE_OPS_DIAG_H_
#include <vector>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
/// \brief Constructs a diagonal tensor with a given diagonal values.
/// Refer to Python API @ref mindspore.ops.Diag for more details.
class MS_CORE_API Diag : public PrimitiveC {
 public:
  /// \brief Constructor.
  Diag() : PrimitiveC(prim::kPrimDiag->name()) { InitIOName({"input_x"}, {"output"}); }
  /// \brief Destructor.
  ~Diag() = default;
  MS_DECLARE_PARENT(Diag, PrimitiveC);
};
AbstractBasePtr DiagInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args);
using PrimDiagPtr = std::shared_ptr<Diag>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_DIAG_H_
