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

#ifndef MINDSPORE_CORE_OPS_FUSED_ADA_FACTOR_H_
#define MINDSPORE_CORE_OPS_FUSED_ADA_FACTOR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameFusedAdaFactor = "FusedAdaFactor";
constexpr auto kNameFusedAdaFactorWithGlobalNorm = "FusedAdaFactorWithGlobalNorm";
/// \brief FusedAdaFactor operation. Refer to Python API @ref mindspore.ops.FusedAdaFactor for more details.
class OPS_API FusedAdaFactor : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(FusedAdaFactor);
  /// \brief Constructor.
  FusedAdaFactor() : BaseOperator(kNameFusedAdaFactor) {}

  /// \brief Constructor with op name.
  explicit FusedAdaFactor(const std::string &name) : BaseOperator(name) {}

  /// \brief Set enable_scale_parameter.
  void set_enable_scale_parameter(bool flag);
  /// \brief Get enable_scale_parameter.
  ///
  /// \return enable_scale_parameter.
  bool get_enable_scale_parameter() const;

  /// \brief Set enable_first_moment.
  void set_enable_first_moment(bool flag);
  /// \brief Get enable_first_moment.
  ///
  /// \return enable_first_moment.
  bool get_enable_first_moment() const;

  /// \brief Set enable_weight_decay.
  void set_enable_weight_decay(bool flag);
  /// \brief Get enable_weight_decay.
  ///
  /// \return enable_weight_decay.
  bool get_enable_weight_decay() const;
};

/// \brief FusedAdaFactorWithGlobalNorm operation. Refer to Python API @ref mindspore.ops.FusedAdaFactorWithGlobalNorm
/// for more details.
class OPS_API FusedAdaFactorWithGlobalNorm : public FusedAdaFactor {
 public:
  MIND_API_BASE_MEMBER(FusedAdaFactorWithGlobalNorm);
  /// \brief Constructor.
  FusedAdaFactorWithGlobalNorm() : FusedAdaFactor(kNameFusedAdaFactorWithGlobalNorm) {}
};

OPS_API abstract::AbstractBasePtr FusedAdaFactorInfer(const abstract::AnalysisEnginePtr &,
                                                      const PrimitivePtr &primitive,
                                                      const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_FUSED_ADA_FACTOR_H_
