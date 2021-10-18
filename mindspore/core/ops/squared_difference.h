/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_SQUARED_DIFFERENCE_H_
#define MINDSPORE_CORE_OPS_SQUARED_DIFFERENCE_H_
#include <vector>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSquaredDifference = "SquaredDifference";
/// \brief Subtracts the second input tensor from the first input tensor element-wise and returns square of it.
/// Refer to Python API @ref mindspore.ops.SquaredDifference for more details.
class MS_CORE_API SquaredDifference : public PrimitiveC {
 public:
  /// \brief Constructor.
  SquaredDifference() : PrimitiveC(kNameSquaredDifference) { InitIOName({"x", "y"}, {"output"}); }
  /// \brief Destructor.
  ~SquaredDifference() = default;
  MS_DECLARE_PARENT(SquaredDifference, PrimitiveC);
  /// \brief Init.
  void Init() {}
};
AbstractBasePtr SquaredDifferenceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args);
using PrimSquaredDifferencePtr = std::shared_ptr<SquaredDifference>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SQUARED_DIFFERENCE_H_
