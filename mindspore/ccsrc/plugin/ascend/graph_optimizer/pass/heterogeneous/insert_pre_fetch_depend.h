/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_HETEROGENEOUS_INSERT_PRE_FETCH_DEPEND_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_HETEROGENEOUS_INSERT_PRE_FETCH_DEPEND_H
#include <vector>

#include "include/backend/common/pass_manager/optimizer.h"

namespace mindspore {
namespace opt {
// Insert Depend for MoveTo(In) in backward, so that data can be prefetched from offload.
class InsertPreFetchDepend : public Pass {
 public:
  InsertPreFetchDepend() : Pass("insert_pre_fetch_depend") {}
  ~InsertPreFetchDepend() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  void Init(const FuncGraphPtr &graph);
  void MakeExecOrderCache();
  void InsertDepend(const CNodePtr &move_to_node, int64_t prefetch);
  size_t CalPreFetchCeiling(const CNodePtr &move_to_node);
  size_t GetFirstUserExecOrder(const CNodePtr &move_to_node);
  bool CalExecutionOrder(const CNodePtr &move_to_node, int64_t prefetch, size_t *pre_exec_order,
                         size_t *post_exec_order);
  HashMap<CNodePtr, size_t> exec_order_cache_;
  HashMap<CNodePtr, size_t> exec_order_cache_without_moveto_;
  std::vector<CNodePtr> exec_order_without_moveto_;
  FuncGraphPtr func_graph_{nullptr};
  KernelGraphPtr kernel_graph_{nullptr};
  FuncGraphManagerPtr manager_{nullptr};
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_HETEROGENEOUS_INSERT_PRE_FETCH_DEPEND_H
