/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_REAL_DIV_H_
#define MINDSPORE_CORE_OPS_REAL_DIV_H_
#include <string>
#include <vector>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameRealDiv = prim::kRealDiv;
/// \brief Divides the first input tensor by the second input tensor in floating-point type element-wise.
/// Refer to Python API @ref mindspore.ops.RealDiv for more details.
class MS_CORE_API RealDiv : public PrimitiveC {
 public:
  /// \brief Constructor.
  RealDiv() : PrimitiveC(kNameRealDiv) { InitIOName({"x", "y"}, {"output"}); }
  /// \brief Destructor.
  ~RealDiv() = default;
  MS_DECLARE_PARENT(RealDiv, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.RealDiv for the inputs.
  void Init() {}
};

AbstractBasePtr RealDivInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args);
using PrimRealDivPtr = std::shared_ptr<RealDiv>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_REAL_DIV_H_
