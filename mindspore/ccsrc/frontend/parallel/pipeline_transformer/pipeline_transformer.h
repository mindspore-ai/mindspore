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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_PIPELINE_TRANSFORMER_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_PIPELINE_TRANSFORMER_H_

#include <utility>
#include <string>
#include <memory>
#include <vector>
#include "ir/value.h"
#include "ir/graph_utils.h"
#include "base/base.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/graph_util/generate_graph.h"

namespace mindspore {
namespace parallel {
using TensorLayoutPtr = std::shared_ptr<TensorLayout>;
using TensorInfoPtr = std::shared_ptr<TensorInfo>;

typedef struct {
  ValueListPtr shape;
  TypePtr type;
  AnfNodePtr depend;
} SendAttr;

class PipelineTransformer {
 public:
  PipelineTransformer(const FuncGraphManagerPtr &manager, int stage, const FuncGraphPtr &root, int64_t global_rank,
                      int64_t per_stage_rank_num)
      : manager_(manager),
        stage_(stage),
        root_(root),
        main_graph_(nullptr),
        virtual_dataset_(nullptr),
        global_rank_(global_rank),
        per_stage_rank_num_(per_stage_rank_num) {}
  virtual ~PipelineTransformer() = default;
  void Coloring();
  void MainGraph();
  void LabelMicroBatch();
  void BroadCastColoring();
  void CutGraph();
  void ParameterColoring();
  void CoverSensShape();
  void ElimGraphStage();
  void ElimParameter();

 private:
  void CreateForwardGroup();
  AnfNodePtr ActualOp(const AnfNodePtr &node);
  bool IsParameterGraph(const AnfNodePtr &node);
  AnfNodeIndexSet GetActualOpUsers(const std::pair<AnfNodePtr, int> &node_pair, NodeUsersMap *node_users_map);
  AnfNodePtr HandleParameterGraph(const AnfNodePtr &node, const AnfNodePtr &use_node, int64_t stage, int64_t user_stage,
                                  const ValuePtr &micro, size_t pos, const std::vector<AnfNodePtr> ops);
  ValuePtr SetMicroBatch(const AnfNodePtr &node, int64_t micro_size);
  std::vector<AnfNodePtr> HandleSharedParameter();
  SendAttr InsertSend(const AnfNodePtr &parameter, int64_t user_node_stage, int64_t node_stage, const ValuePtr &value);
  AnfNodePtr InsertReceive(const FuncGraphPtr &graph, const AnfNodePtr &node, const AnfNodePtr &use_node, int index,
                           int64_t user_node_stage, int64_t node_stage, const ValuePtr &value,
                           const AnfNodePtr &graph_param);
  std::pair<std::vector<AnfNodePtr>, std::vector<AnfNodePtr>> CutBorder(const FuncGraphPtr &graph);
  AnfNodePtr Reuse(const AnfNodePtr &node, int64_t stage, const std::vector<AnfNodePtr> &out_input,
                   const std::string &tag);
  AnfNodePtr FindPipelineCareNode(const AnfNodePtr &node);
  std::pair<OperatorInfoPtr, int> GetOpInfo(const AnfNodePtr &node);
  std::pair<OperatorInfoPtr, int> GetParameterPair(const AnfNodePtr &node);
  OperatorInfoPtr CreateOpInfo(const CNodePtr &cnode, int tuple_index);
  bool LabelParameterStart(const FuncGraphPtr &graph, const CNodePtr &graph_cnode);
  bool NeedGrad(const CNodePtr &cnode, const CNodePtr &graph_cnode);
  CNodePtr GraphOutNode(const AnfNodePtr &node, int tuple_index);
  bool IsPipelineCareNode(const CNodePtr &cnode);
  std::pair<CNodePtr, FuncGraphPtr> FindSensNode();
  FuncGraphManagerPtr manager_;
  int64_t stage_;
  FuncGraphPtr root_;
  FuncGraphPtr main_graph_;
  AnfNodePtr virtual_dataset_;
  int64_t global_rank_;
  int64_t per_stage_rank_num_;
  TypePtr type_ptr_;
  ValueListPtr shape_;
  AnfNodePtr virtual_param_;
  int64_t micro_size_ = 0;
  std::vector<std::string> group_ = {};
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_PIPELINE_TRANSFORMER_H_
