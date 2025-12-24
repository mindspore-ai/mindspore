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

#ifndef MINDSPORE_CORE_OPS_ADD_V2_H_
#define MINDSPORE_CORE_OPS_ADD_V2_H_

#include <memory>
#include <string>
#include <vector>
#include "ops/base_operator.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAddV2 = "AddV2";
/// \brief Adds two input tensors element-wise.
class OPS_API AddV2 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(AddV2);
  /// \brief Constructor.
  AddV2() : BaseOperator(kNameAddV2) { InitIOName({"x", "y"}, {"output"}); }
  explicit AddV2(const std::string k_name) : BaseOperator(k_name) { InitIOName({"x", "y"}, {"output"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.AddV2 for the inputs.
  void Init() const {}
};

OPS_API abstract::AbstractBasePtr AddV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ADD_V2_H_
