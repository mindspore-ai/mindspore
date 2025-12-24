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

#ifndef MINDSPORE_CORE_OPS_EXPAND_H_
#define MINDSPORE_CORE_OPS_EXPAND_H_
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameExpand = "Expand";
/// \brief Expand the tensor ‘x‘ to the size of ‘shape‘
///// Refer to Python API @ref mindspore.ops.Expand for more details.
class OPS_API Expand : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Expand);
  /// \brief Constructor.
  Expand() : BaseOperator(kNameExpand) { InitIOName({"x", "shape"}, {"y"}); }
  /// \brief Destructor.
  void Init() const {}
};

OPS_API abstract::AbstractBasePtr ExpandInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const std::vector<abstract::AbstractBasePtr> &input_args);

using PrimExpand = std::shared_ptr<Expand>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_EXPAND_H_
