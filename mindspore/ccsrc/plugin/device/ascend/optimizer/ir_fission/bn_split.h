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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_BN_SPLIT_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_BN_SPLIT_H_

#include <string>
#include <vector>
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace opt {
class BnSplit : public PatternProcessPass {
 public:
  explicit BnSplit(const string &name = "bn_split", bool multigraph = true) : PatternProcessPass(name, multigraph) {}
  ~BnSplit() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 protected:
  bool CreateOutputsOfBNTrainingReduce(const FuncGraphPtr &graph, const CNodePtr &bn_cnode,
                                       std::vector<AnfNodePtr> *bn_training_reduce_outputs, bool is_dynamic) const;
  AnfNodePtr CreateOutputsOfBNTrainingUpdate(const FuncGraphPtr &graph, const CNodePtr &bn_cnode,
                                             const std::vector<AnfNodePtr> &bn_training_reduce_outputs,
                                             bool is_dynamic) const;

 private:
  AnfNodePtr SplitBatchNormForTBE(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const;
};

class SyncBnSplit : public BnSplit {
 public:
  explicit SyncBnSplit(bool multigraph = true) : BnSplit("sync_bn_split", multigraph) {}
  ~SyncBnSplit() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  AnfNodePtr SyncBNSplitForTBE(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const;
};

AnfNodePtr CreateValueNodeOfDeviceNumReciprocal(const FuncGraphPtr &graph, const CNodePtr &sync_bn_cnode);

AnfNodePtr CreateAllReduceAndMul(const FuncGraphPtr &graph, const AnfNodePtr &allreduce_input,
                                 const CNodePtr &sync_bn_cnode, const PatternProcessPass &pass, bool is_dynamic);
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_BN_SPLIT_H_
