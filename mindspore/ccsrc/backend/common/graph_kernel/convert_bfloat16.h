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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_CONVERT_BFLOAT_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_CONVERT_BFLOAT_H_

#include <string>
#include <utility>
#include <vector>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore::graphkernel {
/**
 * @brief Add Cast for op's inputs if the input data type is bfloat16
 * @example
 *   sub_graph(p0: bfloat16, p1: bfloat16) {
 *     %0 = Op(p0, p1)
 *     return %0
 *   }
 *   ---------->
 *   sub_graph(p0: bfloat16, p1: bfloat16) {
 *     %0 = Cast(p0, float32)
 *     %1 = Cast(p1, float32)
 *     %2 = Op(%0, %1)
 *     %3 = Cast(%2, bfloat16)
 *     return %3
 *   }
 */
class ConvertBFloat16 : public opt::Pass {
 public:
  ConvertBFloat16() : Pass("convert_bfloat16") {}
  ~ConvertBFloat16() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  AnfNodePtr GetCastedInput(const AnfNodePtr &input_node, TypeId dst_type, const FuncGraphPtr &func_graph);
  AnfNodePtr CastTensor(const ValueNodePtr &value_node);
  void CastInput(const CNodePtr &cnode, size_t input_idx, const FuncGraphPtr &func_graph);
  void GetKeepBF16Nodes(const FuncGraphPtr &func_graph);
  bool Process(const FuncGraphPtr &func_graph);
  HashMap<AnfNodePtr, AnfNodePtr> cast_nodes_;
  // (keep_bf16_node, {node_user, input_idx}), node_user's input[input_idx] is keep_bf16_node
  HashMap<AnfNodePtr, std::vector<std::pair<CNodePtr, size_t>>> keep_bf16_nodes_;
  CNodePtr last_node_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_CONVERT_BFLOAT_H_
