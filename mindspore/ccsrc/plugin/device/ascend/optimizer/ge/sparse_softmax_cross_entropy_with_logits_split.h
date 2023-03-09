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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_SPARSE_SOFTMAX_CROSSENTROPY_WITH_LOGITS_SPLIT_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_SPARSE_SOFTMAX_CROSSENTROPY_WITH_LOGITS_SPLIT_H_

#include <string>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class SparseSoftmaxCrossEntropyWithLogitsSplit : public PatternProcessPass {
 public:
  explicit SparseSoftmaxCrossEntropyWithLogitsSplit(const std::string &name = "", bool multigraph = true)
      : PatternProcessPass(name, multigraph) {}
  ~SparseSoftmaxCrossEntropyWithLogitsSplit() override = default;

  const BaseRef DefinePattern() const override = 0;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};

class SparseSoftmaxCrossEntropyWithLogitsSplitCond1 : public SparseSoftmaxCrossEntropyWithLogitsSplit {
 public:
  explicit SparseSoftmaxCrossEntropyWithLogitsSplitCond1(bool multigraph = true)
      : SparseSoftmaxCrossEntropyWithLogitsSplit("sparse_softmax_cross_entropy_with_logits_split_cond1", multigraph) {}
  ~SparseSoftmaxCrossEntropyWithLogitsSplitCond1() override = default;

  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};

class SparseSoftmaxCrossEntropyWithLogitsSplitCond2 : public SparseSoftmaxCrossEntropyWithLogitsSplit {
 public:
  explicit SparseSoftmaxCrossEntropyWithLogitsSplitCond2(bool multigraph = true)
      : SparseSoftmaxCrossEntropyWithLogitsSplit("sparse_softmax_cross_entropy_with_logits_split_cond2", multigraph) {}
  ~SparseSoftmaxCrossEntropyWithLogitsSplitCond2() override = default;

  const BaseRef DefinePattern() const override;
};

class SparseSoftmaxCrossEntropyWithLogitsSplitInfer : public SparseSoftmaxCrossEntropyWithLogitsSplit {
 public:
  explicit SparseSoftmaxCrossEntropyWithLogitsSplitInfer(bool multigraph = true)
      : SparseSoftmaxCrossEntropyWithLogitsSplit("sparse_softmax_cross_entropy_with_logits_split_infer", multigraph) {}
  ~SparseSoftmaxCrossEntropyWithLogitsSplitInfer() override = default;

  const BaseRef DefinePattern() const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_SPARSE_SOFTMAX_CROSSENTROPY_WITH_LOGITS_SPLIT_H_
