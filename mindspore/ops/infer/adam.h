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

#ifndef MINDSPORE_CORE_OPS_ADAM_H_
#define MINDSPORE_CORE_OPS_ADAM_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAdam = "Adam";
/// \brief Updates gradients by the Adaptive Moment Estimation (Adam) algorithm.
/// Refer to Python API @ref mindspore.ops.Adam for more details.
class OPS_API Adam : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Adam);
  /// \brief Constructor.
  Adam() : BaseOperator(kNameAdam) {}
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Adam for the inputs.
  void Init(const bool use_locking = false, const bool use_nesterov = false);
  /// \brief Set use_locking.
  void set_use_locking(const bool use_locking);
  /// \brief Set use_nesterov.
  void set_use_nesterov(const bool use_nesterov);
  /// \brief Get use_locking.
  ///
  /// \return use_locking.
  bool get_use_locking() const;
  /// \brief Get use_nesterov.
  ///
  /// \return use_nesterov.
  bool get_use_nesterov() const;
};
using kPrimAdamPtr = std::shared_ptr<Adam>;
OPS_API abstract::AbstractBasePtr ApplyAdamInferFunc(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                     const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ADAM_H_
