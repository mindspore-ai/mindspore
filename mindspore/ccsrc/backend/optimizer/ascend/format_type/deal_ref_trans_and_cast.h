/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_DEAL_REF_TRANS_AND_CAST_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_DEAL_REF_TRANS_AND_CAST_H_
#include <memory>
#include "ir/anf.h"
#include "backend/optimizer/common/optimizer.h"
#include "backend/optimizer/ascend/ir_fission/transdata_split.h"
#include "backend/optimizer/common/pattern_engine.h"
#include "backend/optimizer/ascend/ascend_helper.h"

namespace mindspore {
namespace opt {
class DealRefTransAndCast : public TransDataSplit {
 public:
  explicit DealRefTransAndCast(bool multigraph = true) : TransDataSplit(multigraph, "deal_ref_trans_and_cast") {}
  ~DealRefTransAndCast() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  CNodePtr MakeDependency(const CNodePtr &getitem, const CNodePtr &final_node, const CNodePtr &cnode,
                          const FuncGraphPtr &func_graph) const;
  CNodePtr SplitTransdataIfNotSupported(const FuncGraphPtr &func_graph, const CNodePtr &cnode) const;
  void DealBroadCastAsRef(const FuncGraphPtr &func_graph, const CNodePtr &cnode) const;
  CNodePtr DealRefSigleOutput(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                              const std::shared_ptr<kernel::OpInfo> &op_info) const;
  CNodePtr DealRefForMultipleOutput(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                    const std::shared_ptr<kernel::OpInfo> &op_info) const;
  CNodePtr AddAdditionalToRefOutput(const FuncGraphPtr &func_graph, const CNodePtr &cnode, size_t output_index,
                                    size_t input_index, const CNodePtr &get_item) const;
  void AddRefPairToKernelGraph(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const AnfNodePtr &get_item,
                               const AnfNodePtr &final_node, size_t final_index,
                               const session::KernelWithIndex &origin_pair) const;
  void AddRefNodePairToKernelGraph(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const size_t output_index,
                                   const size_t input_index) const;
  session::KernelWithIndex FindRefOriginNode(const AnfNodePtr &node) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_DEAL_REF_TRANS_AND_CAST_H_
