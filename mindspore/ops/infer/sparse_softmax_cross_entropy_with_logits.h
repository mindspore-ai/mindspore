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

#ifndef MINDSPORE_CORE_OPS_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H_
#define MINDSPORE_CORE_OPS_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseSoftmaxCrossEntropyWithLogits = "SparseSoftmaxCrossEntropyWithLogits";
/// \brief Computes the softmax cross-entropy value between logits and sparse encoding labels.
/// Refer to Python API @ref mindspore.ops.SparseSoftmaxCrossEntropyWithLogits for more details.
class OPS_API SparseSoftmaxCrossEntropyWithLogits : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SparseSoftmaxCrossEntropyWithLogits);
  /// \brief Constructor.
  SparseSoftmaxCrossEntropyWithLogits() : BaseOperator(kNameSparseSoftmaxCrossEntropyWithLogits) {}
  /// \brief Init.
  /// Refer to the parameters of python API @ref mindspore.ops.SparseSoftmaxCrossEntropyWithLogits for the inputs.
  void Init(const bool is_grad = false);
  /// \brief Set is_grad.
  void set_is_grad(const bool is_grad);
  /// \brief Get is_grad.
  ///
  /// \return is_grad.
  bool get_is_grad() const;
};
OPS_API abstract::AbstractBasePtr SparseSoftmaxCrossEntropyWithLogitsInfer(
  const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
  const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H_
