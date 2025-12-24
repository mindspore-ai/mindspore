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

#ifndef MINDSPORE_CORE_OPS_SPARSE_APPLY_FTRL_H_
#define MINDSPORE_CORE_OPS_SPARSE_APPLY_FTRL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseApplyFtrl = "SparseApplyFtrl";
/// \brief SparseApplyFtrl operation. Refer to Python API @ref mindspore.ops.SparseApplyFtrl for more details.
class OPS_API SparseApplyFtrl : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SparseApplyFtrl);
  /// \brief Constructor.
  SparseApplyFtrl() : BaseOperator(kNameSparseApplyFtrl) {
    InitIOName({"var", "accum", "linear", "grad", "indices"}, {"var", "accum", "linear"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.SparseApplyFtrl for the inputs.
  void Init(float lr, float l1, float l2, float lr_power, bool use_locking = false);
  /// \brief Set lr.
  void set_lr(float lr);
  /// \brief Get lr.
  ///
  /// \return lr.
  float get_lr() const;

  /// \brief Set l1.
  void set_l1(float l1);
  /// \brief Get l1.
  ///
  /// \return l1.
  float get_l1() const;

  /// \brief Set l2.
  void set_l2(float l2);
  /// \brief Get l2.
  ///
  /// \return l2.
  float get_l2() const;

  /// \brief Set lr_power.
  void set_lr_power(float lr_power);
  /// \brief Get lr_power.
  ///
  /// \return lr_power.
  float get_lr_power() const;

  /// \brief Set use_locking.
  void set_use_locking(bool use_locking);
  /// \brief Get use_locking.
  ///
  /// \return use_locking.
  bool get_use_locking() const;
};

OPS_API abstract::AbstractBasePtr SparseApplyFtrlInfer(const abstract::AnalysisEnginePtr &,
                                                       const PrimitivePtr &primitive,
                                                       const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPARSE_APPLY_FTRL_H_
