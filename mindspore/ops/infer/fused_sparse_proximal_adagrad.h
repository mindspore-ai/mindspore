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

#ifndef MINDSPORE_CORE_OPS_FUSED_SPARSE_PROXIMAL_ADAGRAD_H_
#define MINDSPORE_CORE_OPS_FUSED_SPARSE_PROXIMAL_ADAGRAD_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameFusedSparseProximalAdagrad = "FusedSparseProximalAdagrad";
/// \brief Softmax operation. Refer to Python API @ref mindspore.ops.Softmax for more details.
class OPS_API FusedSparseProximalAdagrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(FusedSparseProximalAdagrad);
  /// \brief Constructor.
  FusedSparseProximalAdagrad() : BaseOperator(kNameFusedSparseProximalAdagrad) {
    InitIOName({"var", "accum", "lr", "l1", "l2", "grad", "indices"}, {"var", "accum"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.FusedSparseProximalAdagrad for the inputs.
  void Init(bool use_locking = false);
  /// \brief Set use_locking.
  void set_use_locking(bool use_locking);
  /// \brief Get use_locking.
  ///
  /// \return use_locking.
  bool get_use_locking() const;
};

OPS_API abstract::AbstractBasePtr FusedSparseProximalAdagradInfer(
  const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
  const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_FUSED_SPARSE_PROXIMAL_ADAGRAD_H_
