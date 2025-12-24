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
#ifndef MINDSPORE_CORE_OPS_SPARSE_APPLY_ADAGRAD_V2_H_
#define MINDSPORE_CORE_OPS_SPARSE_APPLY_ADAGRAD_V2_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "ops_utils/op_constants.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseApplyAdagradV2 = "SparseApplyAdagradV2";
class OPS_API SparseApplyAdagradV2 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SparseApplyAdagradV2);
  SparseApplyAdagradV2() : BaseOperator(kNameSparseApplyAdagradV2) {
    InitIOName({"var", "accum", "grad", "indices"}, {"var", "accum"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.SparseApplyAdagradV2 for the inputs.
  void Init(float lr, float epsilon, bool update_slots = true, bool use_locking = false);

  /// \brief Set lr.
  void set_lr(float lr);
  /// \brief Get lr.
  ///
  /// \return lr.
  float get_lr() const;

  /// \brief Set epsilon.
  void set_epsilon(float epsilon);
  /// \brief Get epsilon.
  ///
  /// \return epsilon.
  float get_epsilon() const;

  /// \brief Set update_slots.
  void set_update_slots(bool update_slots);
  /// \brief Get update_slots.
  ///
  /// \return update_slots.
  bool get_update_slots() const;

  /// \brief Set use_locking.
  void set_use_locking(bool use_locking);
  /// \brief Get use_locking.
  ///
  /// \return use_locking.
  bool get_use_locking() const;
};

OPS_API abstract::AbstractBasePtr SparseApplyAdagradV2Infer(const abstract::AnalysisEnginePtr &,
                                                            const PrimitivePtr &primitive,
                                                            const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SPARSE_APPLY_ADAGRAD_V2_H_
