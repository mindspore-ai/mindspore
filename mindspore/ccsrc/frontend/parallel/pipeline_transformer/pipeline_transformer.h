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
#include "ops/array_ops.h"

namespace mindspore {
namespace parallel {
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
  virtual void Coloring();
  virtual void BroadCastColoring();
  virtual void CutGraph();
  void LabelGenMaskFusion();
  bool MainGraph();
  void LabelMicroBatch();
  void ParameterColoring();
  void ElimGraphStage();
  void ModifyParameterList();

  void CreateForwardGroup();
  AnfNodePtr GetArgumentsByParameter(const AnfNodePtr &parameter);
  void RemoveMonadNode();
  AnfNodePtr CreateTupleZeroTensor(const AnfNodePtr &node, size_t index);
  std::vector<AnfNodePtr> GetLoadNodeByParam(const AnfNodePtr &param) const;
  AnfNodePtr ActualOp(const AnfNodePtr &node);
  bool IsParameterGraph(const AnfNodePtr &node) const;
  virtual AnfNodePtr HandleParameterGraph(const AnfNodePtr &node, const AnfNodePtr &use_node, int64_t stage,
                                          int64_t user_stage, const ValuePtr &micro, size_t pos,
                                          const std::vector<AnfNodePtr> &ops);
  AnfNodeIndexSet GetParameterLoadUsers(const AnfNodePtr &node, const NodeUsersMap &node_users_map) const;
  ValuePtr SetMicroBatch(const AnfNodePtr &node, int64_t micro_size, size_t batch_axis) const;
  virtual std::pair<std::vector<AnfNodePtr>, std::vector<AnfNodePtr>> HandleSharedParameter();
  SendAttr InsertSend(const AnfNodePtr &parameter, int64_t user_node_stage, int64_t node_stage, const ValuePtr &value);
  AnfNodePtr InsertReceive(const FuncGraphPtr &graph, const AnfNodePtr &node, const AnfNodePtr &use_node, int index,
                           int64_t user_node_stage, int64_t node_stage, const ValuePtr &value,
                           const AnfNodePtr &graph_param);
  virtual std::pair<std::vector<AnfNodePtr>, std::vector<AnfNodePtr>> CutBorder(const FuncGraphPtr &graph);
  void CutBorderForNode(const FuncGraphPtr &graph, const AnfNodePtr &node, std::vector<AnfNodePtr> *send_ops,
                        std::vector<AnfNodePtr> *receive_ops);
  AnfNodePtr Reuse(const AnfNodePtr &node, int64_t stage, const std::vector<AnfNodePtr> &out_input,
                   const std::string &tag) const;
  AnfNodePtr FindPipelineCareNode(const AnfNodePtr &node) const;
  std::pair<OperatorInfoPtr, int> GetOpInfo(const AnfNodePtr &node);
  TensorInfo GetTensorInfo(const std::pair<OperatorInfoPtr, int> &op_info_pair, bool is_param);
  std::pair<OperatorInfoPtr, int> GetParameterPair(const AnfNodePtr &node);
  OperatorInfoPtr CreateOpInfo(const CNodePtr &cnode, int tuple_index);
  bool LabelParameterStart(const FuncGraphPtr &graph);
  bool NeedGrad(const CNodePtr &cnode);
  CNodePtr GraphOutNode(const AnfNodePtr &node, int tuple_index);
  bool IsPipelineCareNode(const CNodePtr &cnode) const;
  void RedundancyNode(const AnfNodePtr &node, mindspore::HashMap<CNodePtr, std::vector<AnfNodePtr>> *make_tuple_map);
  bool IsRedundancyParameter(const AnfNodePtr &parameter, const std::vector<AnfNodePtr> &non_cloned_parameters);
  void ElimParameter();
  tensor::TensorPtr CreateZeroseOutput(const AnfNodePtr &node, size_t index);
  AnfNodePtr GetZeroOutputs(const FuncGraphPtr &graph);

  std::pair<OperatorInfoPtr, int> GetOpInfoPair(const AnfNodePtr &node, const AnfNodePtr &graph_param,
                                                AnfNodePtr *care_node, bool *is_param);
  void SetNodeAbstract(const std::vector<AnfNodePtr> &nodes);
  std::vector<AnfNodePtr> FetchSend(const AnfNodePtr &node, bool pipeline_param, bool single_pipeline_end,
                                    size_t end_index);
  AnfNodePtr GenNewSendFromOld(const AnfNodePtr &node, const AnfNodePtr &send_input, const ValuePtr &value);
  void HandleGraphOutputs(const std::vector<AnfNodePtr> &nodes);

  std::vector<AnfNodePtr> FetchRecv(const AnfNodePtr &node, bool pipeline_param);
  AnfNodePtr GenNewRecvFromOld(const AnfNodePtr &node, const AnfNodePtr &input, const ValuePtr &value);
  void ResetSharedCellParamAndArgu(const std::vector<std::vector<AnfNodePtr>> &pipeline_begins_fetched,
                                   const std::vector<AnfNodePtr> &newly_added_params,
                                   const std::vector<AnfNodePtr> &reserved_inputs);
  // set shared_cell_ parameters, and call_input
  void HandleGraphInputs(const std::vector<AnfNodePtr> &recv_ops);
  bool GetStageByArgument(const CNodePtr &node, size_t index, const std::vector<AnfNodePtr> &parameters,
                          const NodeUsersMap &node_users_map, std::set<int64_t> *const parameter_stage);
  size_t GetBatchAxisForInput(const AnfNodeIndexSet &input_node_users) const;
  FuncGraphManagerPtr manager_;
  int64_t stage_;
  FuncGraphPtr root_;
  FuncGraphPtr main_graph_;
  FuncGraphPtr shared_cell_;
  AnfNodePtr virtual_dataset_;
  int64_t global_rank_;
  int64_t per_stage_rank_num_;
  TypePtr type_ptr_;
  ValueListPtr shape_;
  AnfNodePtr virtual_param_;
  int64_t micro_size_ = 0;
  std::vector<std::string> group_ = {};
  mindspore::HashMap<AnfNodePtr, std::set<int64_t>> parameter_color_map_ = {};
  bool is_train_{true};
  std::vector<AnfNodePtr> shared_cell_users_;
  bool enable_share_cell_ = false;
};

bool IsInWhiteList(const CNodePtr &cnode);
std::pair<ValueListPtr, TypePtr> GetShapeType(const AnfNodePtr &node, const Shape &shape, size_t index);

class NodeStageInfo {
 public:
  explicit NodeStageInfo(int64_t stage) : stage_(stage) {}
  ~NodeStageInfo() = default;

  int64_t stage() const { return stage_; }

  // Key for user data.
  constexpr static char key[] = "NodeStageInfo";

 private:
  int64_t stage_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_PIPELINE_TRANSFORMER_H_
