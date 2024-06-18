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

#ifndef PARALLEL_AUTO_PARALLEL_REC_GENERATE_STRATEGY_H_
#define PARALLEL_AUTO_PARALLEL_REC_GENERATE_STRATEGY_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <list>
#include <map>

#include "frontend/parallel/auto_parallel/rec_core/rec_graph.h"
#include "frontend/parallel/ops_info/operator_info.h"

namespace mindspore {
namespace parallel {
static std::map<std::string, Dimensions> param_strategy_;
class RecStrategyPropagator {
 public:
  typedef std::list<size_t> prop_list_t;

 private:
  std::shared_ptr<Graph> graph_;
  const std::vector<std::shared_ptr<OperatorInfo>> &ops_;
  std::shared_ptr<std::vector<std::vector<size_t>>> eli_list_;
  const std::vector<std::vector<std::string>> &input_tensor_names_;
  std::shared_ptr<std::vector<size_t>> index_list_;
  bool is_training_;
  std::vector<std::vector<size_t>> shared_tensors_ops_;
  FuncGraphPtr root_;

  prop_list_t forward_;
  prop_list_t backward_;
  std::shared_ptr<std::vector<size_t>> no_stra_op_list_;
  std::vector<size_t> source_ops_;

  void FixInvalidStra();
  void CheckConnectedComponents();

  void AjustToNoTraining();

  void ApplyStrategy(size_t i_op, const Strategies &strategy);

  size_t GenerateEliminatedOperatorStrategyForward(size_t min_devices = 1);
  size_t GenerateEliminatedOperatorStrategyBackward(size_t min_devices = 1);
  size_t GenerateRemainingOperatorStrategy();
  size_t ModifyParamSharingOpsStrategy();
  size_t AssignStandaloneAndBatchParallelOpStrategy();

  std::map<std::string, std::vector<std::pair<size_t, size_t>>> GetParamUsers();
  void SetParamStrategy();
  size_t ApplyParamStrategy();

 public:
  RecStrategyPropagator(const std::shared_ptr<Graph> &graph, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                        const std::shared_ptr<std::vector<std::vector<size_t>>> &eli_list,
                        const std::vector<std::vector<std::string>> &input_tensor_names,
                        const std::shared_ptr<std::vector<size_t>> &index_list, bool is_training,
                        const std::vector<std::vector<size_t>> &shared_tensors_ops, const FuncGraphPtr &root);

  size_t GetMaxDimNum(size_t i_op);
  Dimensions GetDefaultStrategy(size_t i_op);

  size_t CopyMainOperatorsStrategy();
  size_t PropagateFromInputs();
  size_t PropagateFromOutputs();

  void GenerateNoStraList();
  void ExtraShardMatmulOnBatchDim();

  void GenerateStrategyV1();
  void GenerateStrategyV3();
};

Dimensions GetInputStrategy(const std::shared_ptr<Graph> &graph, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                            const std::shared_ptr<std::vector<size_t>> &index_list, size_t i_op,
                            size_t incoming_op_index);

void GenerateStrategy(const std::shared_ptr<Graph> &graph, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                      const std::shared_ptr<std::vector<std::vector<size_t>>> &eli_list,
                      const std::vector<std::vector<std::string>> &input_tensor_names,
                      const std::shared_ptr<std::vector<size_t>> &index_list, bool is_training,
                      const std::vector<std::vector<size_t>> &shared_tensors_ops, const FuncGraphPtr &root);
Dimensions PrepareMatMulStrategy(Graph::NodeType *node, bool transpose_a, bool transpose_b, size_t iter_op_inputs);
Strategies PrepareMatMul(Graph::NodeType *node, const std::shared_ptr<OperatorInfo> &op);
Dimensions PrepareBatchMatMulStrategy(Graph::NodeType *node, const bool transpose_a, const bool transpose_b,
                                      const size_t iter_op_inputs, const size_t dim_num);
Strategies PrepareBatchMatMul(Graph::NodeType *node, const std::shared_ptr<OperatorInfo> &op);
Strategies PreparePropagateBatchMatMul(const std::shared_ptr<OperatorInfo> &op, Dimensions basic_stra);
Strategies PrepareBiasAdd(const std::shared_ptr<Dimensions> &strategy);
Strategies PrepareStridedSlice(const std::shared_ptr<OperatorInfo> &op, Dimensions basic_stra, bool dyn_shape_tmp_fix);
Strategies PrepareSoftMax(const std::shared_ptr<OperatorInfo> &op, const Dimensions &basic_stra);
Strategies PrepareLayerNorm(const std::shared_ptr<OperatorInfo> &op, Dimensions basic_stra);
Strategies PrepareOneHot(const std::shared_ptr<OperatorInfo> &op, Dimensions strategy);
Strategies PrepareGather(const std::shared_ptr<OperatorInfo> &op, Dimensions strategy, bool dyn_shape_tmp_fix);
Dimensions PrepareGatherV2OutputStrategy(const std::shared_ptr<OperatorInfo> &op);
Strategies PrepareL2Normalize(const std::shared_ptr<OperatorInfo> &op, Dimensions strategy);
Strategies PrepareAxisRelatedStrategy(Graph::NodeType *node, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                      const size_t iter_ops);
Strategies MakeRecSearchStrategy(Graph::NodeType *node, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                 const size_t iter_ops);
Strategies MakeDataParallelStrategy(Graph::NodeType *node, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                    const size_t iter_ops);
Strategies MakeFullBatchStrategy(Graph::NodeType *node, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                 const size_t iter_ops);
void SetBackToRawStrategy(const std::shared_ptr<OperatorInfo> &op);
Strategies PrepareStrategy(Graph::NodeType *node, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                           const size_t iter_ops, const bool dyn_shape_tmp_fix);
bool HasStrategy(std::shared_ptr<OperatorInfo> op);
size_t FindIndexOfOperatorIncoming(const std::vector<std::vector<std::string>> &input_tensor_names, size_t iter_ops);
std::pair<size_t, size_t> FindIndexOfOperatorOutgoing(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                      const std::vector<std::vector<std::string>> &input_tensor_names,
                                                      size_t iter_ops);
Dimensions CopyIncomingOperatorOutputStrategy(Graph::NodeType *node,
                                              const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                              const size_t iter_ops, const size_t incoming_op_index);
Dimensions PrepareReshapeOutputStrategy(const std::shared_ptr<OperatorInfo> &op);
Dimensions PrepareTransposeOutputStrategy(const std::shared_ptr<OperatorInfo> &op);
Dimensions PrepareExpandDimsOutputStrategy(const std::shared_ptr<OperatorInfo> &op);
Dimensions PrepareIncomingArithmeticOpeartorInputStrategy(const std::shared_ptr<OperatorInfo> &op);
Dimensions PrepareIncomingOperatorInputStrategy(const std::shared_ptr<OperatorInfo> &op);
Dimensions GetAxisList(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const int64_t iter_ops);
Dimensions ModifyStrategyIfSqueezeIncoming(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                           const size_t incoming_op_index, Dimensions strategy);
Dimensions ModifyStrategyIfReduceIncoming(const std::shared_ptr<OperatorInfo> &op, Dimensions strategy);
Dimensions GetDimListFromAttrs(const std::shared_ptr<OperatorInfo> &op);
Dimensions ModifyStrategyIfArgIncoming(const std::shared_ptr<OperatorInfo> &op, Dimensions strategy);
Dimensions ModifyStrategyIfFlattenIncoming(const std::shared_ptr<OperatorInfo> &op, Dimensions strategy);
Dimensions CopyIncomingOperatorInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                             const size_t iter_ops, const size_t incoming_op_index);
Strategies GenerateStrategiesFromStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                                          Dimensions basic_stra, bool dyn_shape_tmp_fix);
Strategies CheckBroadcast(const std::shared_ptr<OperatorInfo> &op, const Dimensions &strategy);
Dimensions ApplyBroadcast(const std::shared_ptr<OperatorInfo> &op, const Dimensions &strategy,
                          bool broadcast_first_tensor);
Strategies CheckDivisible(const std::shared_ptr<OperatorInfo> &op, const Dimensions &strategy);
Dimensions ModifyStrategyIfSqueezeOutgoing(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                                           Dimensions strategy);
Dimensions PrepareTransposeInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops, size_t i_ops,
                                         size_t outgoing_op_index, size_t iter_op_inputs);
Dimensions CopyOutgoingOperatorInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops, size_t iter_ops,
                                             size_t outgoing_op_index, size_t iter_op_inputs, bool dyn_shape_tmp_fix);
}  // namespace parallel
}  // namespace mindspore
#endif  // PARALLEL_AUTO_PARALLEL_REC_GENERATE_STRATEGY_H_
