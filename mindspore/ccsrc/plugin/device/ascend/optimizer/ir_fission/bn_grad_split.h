/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_BN_GRAD_SPLIT_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_BN_GRAD_SPLIT_H_

#include <string>
#include <vector>
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace opt {
class BnGradSplit : public PatternProcessPass {
 public:
  explicit BnGradSplit(const string name = "bn_grad_split", bool multigraph = true)
      : PatternProcessPass(name, multigraph) {}
  ~BnGradSplit() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 protected:
  void CreateOutputsOfUpdateGrad(const FuncGraphPtr &graph, const CNodePtr &bn_grad_node,
                                 std::vector<AnfNodePtr> *bn_update_grad_outputs, bool is_dynamic) const;
  void CreateOutputsOfReduceGrad(const FuncGraphPtr &graph, const CNodePtr &bn_grad_node,
                                 const std::vector<AnfNodePtr> &bn_update_grad_outputs,
                                 std::vector<AnfNodePtr> *bn_reduce_grad_outputs, bool is_dynamic) const;

 private:
  CNodePtr BNGradSplitForTBE(const FuncGraphPtr &func_graph, const CNodePtr &cnode) const;
};

class SyncBnGradSplit : public BnGradSplit {
 public:
  explicit SyncBnGradSplit(bool multigraph = true) : BnGradSplit("sync_bn_grad_split", multigraph) {}
  ~SyncBnGradSplit() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  CNodePtr SyncBNGradSplitForTBE(const FuncGraphPtr &func_graph, const CNodePtr &cnode) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_BN_GRAD_SPLIT_H_
