/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_REORDER_OPS_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_REORDER_OPS_H_

#include <memory>
#include <vector>
#include <string>
#include "include/backend/optimizer/pass.h"

namespace mindspore::graphkernel {
struct NodeIOInfo {
  std::vector<std::string> inputs_format;
  std::vector<std::string> outputs_format;
  std::vector<TypeId> inputs_type;
  std::vector<TypeId> outputs_type;
};

class ReorderOps : public opt::Pass {
 public:
  ReorderOps() : Pass("reorder_ops") {}
  ~ReorderOps() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  void SetTypeInsensitiveNodeInputs(const CNodePtr &node, const std::vector<size_t> &indexes,
                                    const std::vector<AnfNodePtr> &new_input_at_indexes,
                                    std::vector<AnfNodePtr> *new_inputs) const;
  void SetTypeInsensitiveNodeInputsInfo(const CNodePtr &node, const std::vector<size_t> &indexes,
                                        const std::vector<AnfNodePtr> &input_at_indexes, NodeIOInfo *new_inputs_info,
                                        bool from_input) const;
  bool ReorderTypeInsensitiveCastDown(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &mng,
                                      const CNodePtr &node) const;
  bool ReorderCastUpTypeInsensitive(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &mng,
                                    const CNodePtr &node) const;
  bool ReorderCastTypeInsensitive(const FuncGraphPtr &func_graph) const;
};
using ReorderOpsPtr = std::shared_ptr<ReorderOps>;
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_REORDER_OPS_H_
