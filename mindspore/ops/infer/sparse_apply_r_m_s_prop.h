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

#ifndef MINDSPORE_CORE_OPS_SPARSE_APPLY_R_M_S_PROP_H_
#define MINDSPORE_CORE_OPS_SPARSE_APPLY_R_M_S_PROP_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseApplyRMSProp = "SparseApplyRMSProp";
/// \brief Update relevant entries according to the rmsprop algorithm.
class OPS_API SparseApplyRMSProp : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SparseApplyRMSProp);
  /// \brief Constructor.
  SparseApplyRMSProp() : BaseOperator(kNameSparseApplyRMSProp) {
    InitIOName({"var", "ms", "mom", "lr", "grad", "indices"}, {"var", "ms", "mom"});
  }

  /// \brief Set rho, the decay rate.
  void set_rho(const float epsilon);
  /// \brief Get rho.
  ///
  /// \return rho.
  float get_rho() const;
  /// \brief Set momentum.
  void set_momentum(const float momentum);
  /// \brief Get momentum.
  ///
  /// \return momentum.
  float get_momentum() const;
  /// \brief Set epsilon, A small value (float) added for numerical stability.
  void set_epsilon(const float epsilon);
  /// \brief Get epsilon.
  ///
  /// \return epsilon.
  float get_epsilon() const;

  /// \brief Set use_locking, A bool where if True, updating var, ms and mom is protected by a lock. Default: False.
  void set_use_locking(const bool use_locking);
  /// \brief Get use_locking.
  ///
  /// \return use_locking.
  bool get_use_locking() const;
};

OPS_API abstract::AbstractBasePtr SparseApplyRMSPropInfer(const abstract::AnalysisEnginePtr &,
                                                          const PrimitivePtr &primitive,
                                                          const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPARSE_APPLY_R_M_S_PROP_H_
