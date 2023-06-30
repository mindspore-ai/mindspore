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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_REDUCE_CONCAT_ONLINE_FUSION_PASS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_REDUCE_CONCAT_ONLINE_FUSION_PASS_H_

#include <stack>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include "include/model.h"
#include "src/executor/kernel_exec.h"
#include "src/litert/lite_model.h"
#include "src/litert/inner_context.h"
#include "src/litert/sub_graph_split.h"
#include "src/litert/pass/online_fusion/online_fusion_pass.h"
#include "src/common/prim_util.h"
#include "nnacl/split_parameter.h"

namespace mindspore::lite {
class ReduceConcatOnlineFusionPass : public OnlineFusionPass {
 public:
  explicit ReduceConcatOnlineFusionPass(SearchSubGraph *search_subgrap) : OnlineFusionPass(search_subgrap) {}
  ~ReduceConcatOnlineFusionPass() = default;

 private:
  void DoOnlineFusion() override;
  void DoReduceConcatFusionPass();
  bool DoReduceConcatFusion(uint32_t node_id);

  bool SatifyReduceConcatParse(uint32_t in_node, int *lastAxisSize);

  void DeleteReduceConcatOriginNode(SearchSubGraph::Subgraph *subgraph, const std::vector<uint32_t> &positions);
  int CreateReduceConcatCustomNode(LiteGraph::Node *node, SearchSubGraph::Subgraph *subgraph,
                                   std::vector<uint32_t> *new_input_indices, std::vector<uint32_t> *positions);
};  // namespace mindspore::lite
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_REDUCE_CONCAT_ONLINE_FUSION_PASS_H_
