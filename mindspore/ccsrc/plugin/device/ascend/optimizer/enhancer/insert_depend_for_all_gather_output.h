/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_INSERT_DEPEND_FOR_ALL_GATHER_OUTPUT_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_INSERT_DEPEND_FOR_ALL_GATHER_OUTPUT_H_
#include <string>
#include <memory>
#include <map>
#include <vector>

#include "include/backend/optimizer/pass.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/optimizer/optimizer.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"

namespace mindspore {
namespace opt {
class InsertDependForAllGatherOutput : public Pass {
 public:
  InsertDependForAllGatherOutput()
      : Pass("insert_depend_for_all_gather_output"), kernel_select_(std::make_shared<KernelSelect>()) {}
  ~InsertDependForAllGatherOutput() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  std::map<int64_t, std::vector<AnfNodePtr>> forward_each_seg_first_recv_;
  std::vector<AnfNodePtr> redistribution_all_gather_node_;
  std::vector<AnfNodePtr> get_next_tuplegetitem_node_;
  std::map<int64_t, AnfNodePtr> all_gather_node_;
  std::map<int64_t, AnfNodePtr> forward_last_seg_each_micro_recv_;
  AnfNodePtr pipeline_param_send_ = nullptr;
  KernelSelectPtr kernel_select_;
  void InsertDepend(const AnfNodePtr &prior_node, const AnfNodePtr &post_node, const FuncGraphPtr &root) const;
  int64_t DealSegment(const std::vector<AnfNodePtr> &node_list);
  void ReorderGetnext(const FuncGraphPtr &graph, bool *changed);
  bool IsLastSegWithRecv(int64_t seg_max, std::shared_ptr<CNode> cnode);
  bool IsGatherNode(std::shared_ptr<CNode> cnode, bool is_recompute);
  bool IsRedistriuteAllGatherNode(int64_t seg_max, std::shared_ptr<CNode> cnode);
  void GetEachSegSend(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &node_list, int64_t seg_max);
  void IsChanged(const FuncGraphPtr &graph, AnfNodePtr node, int64_t segment_info, bool *changed);
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_INSERT_DEPEND_FOR_ALL_GATHER_OUTPUT_H_
