/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_OPTIMIZER_UTILS_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_OPTIMIZER_UTILS_H_

#include <vector>
#include "ir/func_graph.h"

namespace mindspore {
namespace opt {
class OptimizerUtils {
 public:
  static void MoveContrlDepend(const FuncGraphPtr &func_graph, const AnfNodePtr &from_node, const AnfNodePtr &to_node);
  static std::vector<CNodePtr> MoveDataDepend(const FuncGraphPtr &func_graph, const AnfNodePtr &from_node,
                                              const CNodePtr &to_node);
  static void ReplaceDataDepend(const FuncGraphPtr &func_graph, const std::vector<CNodePtr> &old_nodes,
                                const AnfNodePtr &new_node);
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_OPTIMIZER_UTILS_H_
