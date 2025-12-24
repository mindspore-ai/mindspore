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

#ifndef MINDSPORE_CORE_OPS_DROPOUT_GEN_MASK_H_
#define MINDSPORE_CORE_OPS_DROPOUT_GEN_MASK_H_

#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameDropoutGenMask = "DropoutGenMask";
constexpr auto kNameStatelessDropOutGenMask = "StatelessDropOutGenMask";

/// \brief Generates the mask value for the input shape.
/// Refer to Python API @ref mindspore.ops.DropoutGenMask for more details.
class OPS_API DropoutGenMask : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(DropoutGenMask);
  /// \brief Constructor.
  DropoutGenMask() : BaseOperator(kNameDropoutGenMask) {}
  /// \brief Constructor with op name.
  explicit DropoutGenMask(const std::string &name) : BaseOperator(name) {}
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.DropoutGenMask for the inputs.
  void Init() const {}
};

/// \brief Generates the mask value for the input shape.
class OPS_API StatelessDropOutGenMask : public DropoutGenMask {
 public:
  MIND_API_BASE_MEMBER(StatelessDropOutGenMask);
  /// \brief Constructor.
  StatelessDropOutGenMask() : DropoutGenMask(kNameStatelessDropOutGenMask) {}
};

OPS_API abstract::AbstractBasePtr DropoutGenMaskInfer(const abstract::AnalysisEnginePtr &,
                                                      const PrimitivePtr &primitive,
                                                      const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_DROPOUT_GEN_MASK_H_
