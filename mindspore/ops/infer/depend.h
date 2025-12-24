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

#ifndef MINDSPORE_CORE_OPS_DEPEND_H_
#define MINDSPORE_CORE_OPS_DEPEND_H_
#include <memory>
#include <vector>

#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameDepend = "Depend";
/// \brief Depend defined Depend operator prototype.
class OPS_API Depend : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Depend);
  /// \brief Constructor.
  Depend() : BaseOperator(kNameDepend) {}

  /// \brief Method to init the op's attributes.
  void Init() const {}
};
OPS_API abstract::AbstractBasePtr DependInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimDepend = std::shared_ptr<Depend>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_DEPEND_H_
