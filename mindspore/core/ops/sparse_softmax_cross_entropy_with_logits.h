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
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseSoftmaxCrossEntropyWithLogits = "SparseSoftmaxCrossEntropyWithLogits";
class SparseSoftmaxCrossEntropyWithLogits : public PrimitiveC {
 public:
  SparseSoftmaxCrossEntropyWithLogits() : PrimitiveC(kNameSparseSoftmaxCrossEntropyWithLogits) {}
  ~SparseSoftmaxCrossEntropyWithLogits() = default;
  MS_DECLARE_PARENT(SparseSoftmaxCrossEntropyWithLogits, PrimitiveC);
  void Init(const bool is_grad = false);
  void set_is_grad(const bool is_grad);
  bool get_is_grad() const;
};
AbstractBasePtr SparseSoftmaxCrossEntropyWithLogitsInfer(const abstract::AnalysisEnginePtr &,
                                                         const PrimitivePtr &primitive,
                                                         const std::vector<AbstractBasePtr> &input_args);
using PrimSparseSoftmaxCrossEntropyWithLogitsPtr = std::shared_ptr<SparseSoftmaxCrossEntropyWithLogits>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H_
