/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_UNIFY_MINDIR_H
#define MINDSPORE_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_UNIFY_MINDIR_H

#include <memory>
#include <string>
#include <vector>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR : public PatternProcessPass {
 public:
  explicit SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR(
    const std::string &name = "sparse_softmax_cross_entropy_with_logits_unify_mindir", bool multigraph = true)
      : PatternProcessPass(name, multigraph) {}
  ~SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
};

class GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR : public PatternProcessPass {
 public:
  explicit GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR(bool multigraph = true)
      : PatternProcessPass("grad_sparse_softmax_cross_entropy_with_logits_unify_mindir", multigraph) {}
  ~GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;
};

class GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV2 : public PatternProcessPass {
 public:
  explicit GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV2(bool multigraph = true)
      : PatternProcessPass("grad_sparse_softmax_cross_entropy_with_logits_unify_mindir_v2", multigraph) {}
  ~GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV2() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;
};

class PynativeSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR : public SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR {
 public:
  explicit PynativeSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR(bool multigraph = true)
      : SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR("pynative_sparse_softmax_cross_entropy_with_logits_unify_mindir",
                                                       multigraph) {}
  ~PynativeSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;
};

class PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR : public PatternProcessPass {
 public:
  explicit PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR(bool multigraph = true)
      : PatternProcessPass("pynative_grad_sparse_softmax_cross_entropy_with_logits_unify_mindir", multigraph) {}
  ~PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
};

class PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV2 : public PatternProcessPass {
 public:
  explicit PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV2(bool multigraph = true)
      : PatternProcessPass("pynative_grad_sparse_softmax_cross_entropy_with_logits_unify_mindir_v2", multigraph) {}
  ~PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV2() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
};

class PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV3 : public PatternProcessPass {
 public:
  explicit PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV3(bool multigraph = true)
      : PatternProcessPass("pynative_grad_sparse_softmax_cross_entropy_with_logits_unify_mindir_v3", multigraph) {}
  ~PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV3() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_UNIFY_MINDIR_H
