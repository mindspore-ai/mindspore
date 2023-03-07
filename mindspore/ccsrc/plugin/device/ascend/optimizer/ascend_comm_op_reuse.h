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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_ASCEND_COMM_OP_REUSE_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_ASCEND_COMM_OP_REUSE_H_

#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "include/backend/kernel_graph.h"

namespace mindspore {
namespace opt {
class AscendCommOpReuse {
 public:
  AscendCommOpReuse(const KernelGraphPtr &root_graph, const size_t &max_comm_op_reuse_num)
      : root_graph_(root_graph), max_comm_op_reuse_num_(max_comm_op_reuse_num) {}
  void Run();

 private:
  void FindAllCommOp();
  void InsertMonadForReusedCommOp();
  void InsertMonadForReusedCommOpRecur(const KernelGraphPtr &kg, std::set<KernelGraphPtr> *memo);
  std::vector<std::pair<CNodePtr, KernelGraphPtr>> FindCommOpRecur(const KernelGraphPtr &kg,
                                                                   std::set<KernelGraphPtr> *memo);
  void AnalyseCommOpReuse();
  KernelGraphPtr CreateCommSubGraph(const CNodePtr &comm_op);
  void ReplaceCommOpToCallNode();

  KernelGraphPtr root_graph_ = {};
  std::vector<std::pair<CNodePtr, KernelGraphPtr>> all_comm_ops_ = {};  // use vector to keep order
  std::map<CNodePtr, KernelGraphPtr> reused_comm_sub_graphs_ = {};      // origin comm op to reused comm sub graph
  const size_t max_comm_op_reuse_num_;
  size_t total_comm_op_reuse_num_ = 0;
  uint32_t comm_subgraph_sum_ = 50000;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_ASCEND_COMM_OP_REUSE_H_
