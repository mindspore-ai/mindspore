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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_SPLIT_REDUCE_CONCAT_ONLINE_FUSION_PASS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_SPLIT_REDUCE_CONCAT_ONLINE_FUSION_PASS_H_

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
class SplitReduceConcatOnlineFusionPass : public OnlineFusionPass {
 public:
  explicit SplitReduceConcatOnlineFusionPass(SearchSubGraph *search_subgrap) : OnlineFusionPass(search_subgrap) {}
  ~SplitReduceConcatOnlineFusionPass() = default;

 private:
  void DoOnlineFusion() override;
  void DoSplitReduceConcatFusion(uint32_t node_id);
  bool SatifyReduceConcatParse(SearchSubGraph::Subgraph *subgraph, uint32_t node_id, int split_concat_axis,
                               std::vector<uint32_t> *positions);
  void DeleteOriginNode(SearchSubGraph::Subgraph *subgraph, const std::vector<uint32_t> &positions);
  void DoSplitReduceConcatFusionPass();
  int CreateCustomNode(LiteGraph::Node *node, SearchSubGraph::Subgraph *subgraph, SplitParameter *split_param,
                       const std::vector<uint32_t> &positions);
};  // namespace mindspore::lite
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_SPLIT_REDUCE_CONCAT_ONLINE_FUSION_PASS_H_
