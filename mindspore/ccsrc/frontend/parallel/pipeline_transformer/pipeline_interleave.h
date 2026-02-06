/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_PIPELINE_INTERLEAVE_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_PIPELINE_INTERLEAVE_H_

#include <set>
#include <utility>
#include <string>
#include <memory>
#include <vector>

#include "ir/value.h"
#include "base/base.h"
#include "utils/hash_map.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/pipeline_transformer/pipeline_transformer.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "primitive/array_ops.h"

namespace mindspore {
namespace parallel {
class PipelineInterleave {
 public:
  PipelineInterleave(const FuncGraphManagerPtr &manager, int stage, const FuncGraphPtr &root)
      : manager_(manager), stage_(stage), root_(root), main_graph_(nullptr), virtual_dataset_(nullptr) {}
  virtual ~PipelineInterleave() = default;
  void Init();
  void Coloring();
  void BroadCastColoring();
  void CutBorder();
  void LabelGenMaskFusion();
  bool MainGraph();
  void LabelMicroBatch();
  void ParameterColoring();
  void ElimParameter();
  bool IsNoUpdateParameterStage(const int64_t stage);

 private:
  void CreateSendReceiveGroup();
  std::shared_ptr<NodeStageInfo> GetStageInfoByGraph(const FuncGraphPtr &fg);
  void InsertSendReceiveForParameter(const AnfNodePtr &param, const AnfNodePtr &node, int64_t src_stage,
                                     int64_t dst_stage, int64_t chunk, int64_t index, int64_t order);
  void InsertSendReceiveForSharedParam(const AnfNodePtr &parameter, const AnfNodePtr &argument, int64_t *order);
  void HandleSharedParam(int64_t *order);
  void RedundancyNode(const AnfNodePtr &node, mindspore::HashMap<CNodePtr, std::vector<AnfNodePtr>> *make_tuple_map);
  bool IsRedundancyParameter(const AnfNodePtr &parameter, const std::vector<AnfNodePtr> &non_cloned_parameters);
  void InsertSendReceive(const AnfNodePtr &node, const AnfNodePtr &user_node, int64_t order, int64_t index,
                         bool is_v_shape = False);
  void SetScheduler();
  void RemoveMonadNode();
  void BroadCastGraphStage(const FuncGraphPtr &fg);
  std::vector<AnfNodePtr> GetLoadNodeByParam(const AnfNodePtr &param) const;
  ValuePtr SetMicroBatch(const AnfNodePtr &node, int64_t micro_size, size_t batch_axis) const;
  void FreezeGradient();
  void CutBorderForNode(const FuncGraphPtr &graph, const AnfNodePtr &node, int64_t *order);
  bool GetStageByArgument(const CNodePtr &node, size_t index, const std::vector<AnfNodePtr> &parameters,
                          const NodeUsersMap &node_users_map, std::set<int64_t> *const parameter_stage);
  void FindStridedSliceNodes(const AnfNodePtr &node, AnfNodeSet *strided_slice_nodes) const;
  size_t MicroSize(const AnfNodeIndexSet &input_node_users) const;
  size_t GetBatchAxisForInput(const AnfNodeIndexSet &input_node_users) const;
  mindspore::HashMap<int64_t, std::vector<int64_t>> BuildStageRanksMap() const;
  FuncGraphManagerPtr manager_;
  NodeUsersMap node_users_map_;
  int64_t stage_;
  FuncGraphPtr root_;
  FuncGraphPtr main_graph_;
  FuncGraphPtr shared_cell_;
  AnfNodePtr virtual_dataset_;
  int64_t micro_size_ = 0;
  mindspore::HashMap<AnfNodePtr, std::set<int64_t>> parameter_color_map_ = {};
  std::string world_group_;
  std::string group_;
  bool is_train_{true};
  bool is_v_shape_{false};
  int64_t global_rank_ = 0;
  int64_t per_stage_rank_num_ = 0;
  bool is_vpp_ = false;
  int64_t param_stage_ = -1;
};

class PipelinePostProcess {
 public:
  explicit PipelinePostProcess(const FuncGraphManagerPtr &manager, int64_t stage, int64_t stage_num, FuncGraphPtr root)
      : manager_(manager), stage_(stage), stage_num_(stage_num), root_(root) {}
  virtual ~PipelinePostProcess() = default;

  void Init(const std::vector<AnfNodePtr> &nodes);
  void ModifySendRecvAttr(const std::vector<AnfNodePtr> &all_nodes);
  void GraphPartition(const std::vector<AnfNodePtr> &all_nodes);
  void ElimGraphStage();
  void ModifyParameterList();
  void HandleSendParam();

 private:
  void LabelInterleaveIndex();
  void RemoveMonadNodeBetweenStage(const CNodePtr &cnode);
  void EliminateUpdateStateMakeTupleWithUselessLoadNode();
  void RemoveUselessOriginSharedCell();
  void RemoveMonadNode(const FuncGraphPtr &fg, int64_t chunk);
  std::vector<AnfNodePtr> PartitionChunkGraph(const FuncGraphPtr &fg, int64_t chunk);
  std::vector<AnfNodePtr> PartitionVShapeChunkGraph(const std::vector<AnfNodePtr> &sends);
  void GetSendsRecvs(const FuncGraphPtr &fg, int64_t chunk, std::vector<AnfNodePtr> *recvs,
                     std::vector<AnfNodePtr> *sends, std::vector<AnfNodePtr> *temp);
  void SetNodeAbstract(const std::vector<AnfNodePtr> &nodes);
  AnfNodePtr GetZeroOutputs(const FuncGraphPtr &graph);
  AnfNodePtr GenNewNodeFromOld(const AnfNodePtr &node, const AnfNodePtr &input, int64_t micro, int64_t index);
  AnfNodePtr GenNewParamRecv(const AnfNodePtr &node, const ParameterPtr &param);
  std::vector<AnfNodePtr> GenerateMainGraphSend(const std::vector<AnfNodePtr> &nodes, const AnfNodePtr &node,
                                                const ValuePtr &micro, const ValuePtr &index);
  AnfNodePtr GenerateMainGraphRecv(const AnfNodePtr &fg_node, const AnfNodePtr &recv);
  void MoveSharedParamMirrorOutCall(const std::vector<AnfNodePtr> &all_nodes);
  FuncGraphManagerPtr manager_;
  int64_t stage_;
  int64_t stage_num_;
  FuncGraphPtr root_;
  int64_t chunk_num_ = 1;
  bool is_v_shape_{false};
  FuncGraphPtr main_graph_;
  FuncGraphPtr shared_cell_;
  std::vector<AnfNodePtr> shared_cell_users_;
};

bool IsolatedNodeAttach(const FuncGraphPtr &root, const opt::OptimizerPtr &optimizer);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_PIPELINE_INTERLEAVE_H_
