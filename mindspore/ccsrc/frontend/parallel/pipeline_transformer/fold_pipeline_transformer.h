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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_FOLD_PIPELINE_TRANSFORMER_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_FOLD_PIPELINE_TRANSFORMER_H_

#include <set>
#include <utility>
#include <string>
#include <memory>
#include <vector>
#include "ir/value.h"
#include "ir/graph_utils.h"
#include "base/base.h"
#include "utils/hash_map.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/pipeline_transformer/pipeline_transformer.h"

namespace mindspore {
namespace parallel {

class FoldPipelineTransformer : public PipelineTransformer {
 public:
  FoldPipelineTransformer(const FuncGraphManagerPtr &manager, int stage, const FuncGraphPtr &root, int64_t global_rank,
                          int64_t per_stage_rank_num)
      : PipelineTransformer(manager, stage, root, global_rank, per_stage_rank_num) {}
  ~FoldPipelineTransformer() = default;
  void Coloring() override;
  void BroadCastColoring() override;
  void CutGraph() override;

  SendAttr InsertSend(const AnfNodePtr &parameter, int64_t user_node_stage, int64_t node_stage, const ValuePtr &value,
                      int64_t segment);
  AnfNodePtr InsertReceive(const FuncGraphPtr &graph, const AnfNodePtr &node, const AnfNodePtr &use_node, int index,
                           int64_t user_node_stage, int64_t node_stage, const ValuePtr &value,
                           const AnfNodePtr &graph_param, int64_t segment);

  void CutBorderForNode(const FuncGraphPtr &graph, const AnfNodePtr &node, std::vector<AnfNodePtr> *send_ops,
                        std::vector<int64_t> *send_ops_segment, std::vector<AnfNodePtr> *receive_ops);
  AnfNodePtr Reuse(const AnfNodePtr &node, int64_t stage, int64_t node_segment,
                   const std::vector<AnfNodePtr> &out_input, const std::vector<int64_t> &out_input_segment,
                   const std::string &tag);
  std::pair<std::vector<AnfNodePtr>, std::vector<AnfNodePtr>> CutBorder(const FuncGraphPtr &graph) override;
  AnfNodePtr HandleParameterGraph(const AnfNodePtr &node, const AnfNodePtr &use_node, int64_t stage, int64_t user_stage,
                                  const ValuePtr &micro, size_t pos, const std::vector<AnfNodePtr> &ops) override;
  std::pair<std::vector<AnfNodePtr>, std::vector<AnfNodePtr>> HandleSharedParameter() override;

 private:
  void CreateForwardGroup2();
  int64_t ComputeRecvTag(int64_t node_stage, int64_t user_node_stage, int64_t stage_num, int64_t src_rank);
  void ColorForNodes();
  std::vector<std::string> group_ = {};
};

class NodeSegmentInfo {
 public:
  explicit NodeSegmentInfo(int64_t segment) : segment_(segment) {}
  ~NodeSegmentInfo() = default;

  int64_t segment() const { return segment_; }

  // Key for user data.
  constexpr static char key[] = "NodeSegmentInfo";

 private:
  int64_t segment_;
};

}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_FOLD_PIPELINE_TRANSFORMER_H_
