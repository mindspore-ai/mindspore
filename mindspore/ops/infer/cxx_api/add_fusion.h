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

#ifndef MINDSPORE_CORE_OPS_ADD_FUSION_H_
#define MINDSPORE_CORE_OPS_ADD_FUSION_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAddFusion = "AddFusion";
/// \brief AddFusion defined Add operator prototype of lite.
class OPS_API AddFusion : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(AddFusion);
  /// \brief Constructor.
  AddFusion() : BaseOperator(kNameAddFusion) { InitIOName({"x", "y"}, {"output"}); }

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] activation_type Define the activation type.
  void Init(const ActivationType activation_type);

  /// \brief Method to set activation type.
  ///
  /// \param[in] activation_type Define the activation type.
  void set_activation_type(const ActivationType activation_type);

  /// \brief Method to get activation type.
  ///
  /// \return activation type.
  ActivationType get_activation_type() const;
};

abstract::AbstractBasePtr AddFusionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ADD_FUSION_H_
