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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_INSERT_DEPEND_FOR_ALL_REDUCE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_INSERT_DEPEND_FOR_ALL_REDUCE_H_
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
class InsertDependForAllReduce : public Pass {
 public:
  InsertDependForAllReduce() : Pass("insert_depend_for_all_reduce"), kernel_select_(std::make_shared<KernelSelect>()) {}
  ~InsertDependForAllReduce() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  void InsertDepend(const AnfNodePtr &prior_node, const AnfNodePtr &post_node, const FuncGraphPtr &graph) const;
  void InsertAllReduceOpAfterSendOp(const FuncGraphPtr &graph);
  void HandleAllReduceUsersNode(const FuncGraphPtr &graph);
  void FindEachSegLastSend();
  KernelSelectPtr kernel_select_;
  std::vector<AnfNodePtr> all_reduce_node_;
  int64_t min_fusion_ = INT64_MAX;
  int64_t micro_max_ = 0;
  std::vector<AnfNodePtr> node_list_;
  AnfNodePtr last_allreduce_ = nullptr;
  std::map<std::int64_t, AnfNodePtr> backward_each_seg_last_send_;
  std::vector<std::vector<AnfNodePtr>> allreduce_users_list_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_INSERT_DEPEND_FOR_ALL_REDUCE_H_
