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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_DEAL_REF_OUTPUT_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_DEAL_REF_OUTPUT_H_
#include <memory>
#include "ir/anf.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/pattern_engine.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"

namespace mindspore {
namespace opt {
class DealRefOutput : public PatternProcessPass {
 public:
  explicit DealRefOutput(bool multigraph = true) : PatternProcessPass("deal_ref_output", multigraph) {}
  ~DealRefOutput() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  CNodePtr MakeDependency(const AnfNodePtr &get_item, const AnfNodePtr &final_node, const CNodePtr &cnode,
                          const FuncGraphPtr &func_graph) const;
  void DealBroadCastAsRef(const FuncGraphPtr &func_graph, const CNodePtr &cnode) const;
  AnfNodePtr DealRefSingleOutput(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                 const std::shared_ptr<kernel::OpInfo> &op_info) const;
  AnfNodePtr DealRefForMultipleOutput(const FuncGraphPtr &func_graph, const CNodePtr &orig_cnode,
                                      const std::shared_ptr<kernel::OpInfo> &op_info) const;
  AnfNodePtr AddAdditionalToRefOutput(const FuncGraphPtr &func_graph, const CNodePtr &cnode, size_t output_index,
                                      size_t input_index, const AnfNodePtr &get_item) const;
  void AddRefPairToKernelGraph(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const AnfNodePtr &get_item,
                               const AnfNodePtr &final_node, size_t final_index,
                               const session::KernelWithIndex &origin_pair) const;
  void AddRefNodePairToKernelGraph(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const size_t output_index,
                                   const size_t input_index) const;
  session::KernelWithIndex FindRefOriginNode(const AnfNodePtr &node) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_DEAL_REF_OUTPUT_H_
