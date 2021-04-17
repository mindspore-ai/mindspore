/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_CONVERT_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_CONVERT_H_

#define DRAW_GE_GRAPH

#include <memory>
#include <map>
#include <set>
#include <vector>
#include <unordered_map>
#include <string>
#include <utility>
#include <stack>
#include <fstream>
#include <sstream>

#include "ir/anf.h"
#include "ir/func_graph.h"
#include "transform/graph_ir/util.h"
#include "ir/tensor.h"
#include "transform/graph_ir/df_graph_manager.h"
#include "utils/config_manager.h"
#include "transform/graph_ir/op_adapter.h"
#include "graph/operator_reg.h"
#ifdef OPEN_SOURCE
#include "ge/client/ge_api.h"
#else
#include "external/ge/ge_api.h"
#endif
#include "graph/tensor.h"
#include "ops/hcom_ops.h"

namespace mindspore {
namespace transform {
using TensorOrderMap = std::map<std::string, std::shared_ptr<tensor::Tensor>>;
using HcomBroadcast = ge::op::HcomBroadcast;
class DfGraphConvertor {
 public:
  explicit DfGraphConvertor(const AnfGraphPtr &anf_graph)
      : anf_graph_(anf_graph), df_graph_(std::make_shared<DfGraph>(anf_graph_->ToString())) {
#if (!defined ENABLE_GE) || (defined ENABLE_INFER)
    training_ = anf_graph->has_flag("training");
#else
    training_ = ENABLE_TRAIN;
#endif
    distribute_ = anf_graph->has_flag("broadcast_flag");
    if (anf_graph->has_flag("broadcast_flag")) {
      ConfigManager::GetInstance().set_parallel_strategy(ParallelStrategy::DISTRIBUTION);
    } else {
      ConfigManager::GetInstance().set_parallel_strategy(ParallelStrategy::ONE_DEVICE);
    }

    MS_LOG(INFO) << "Create DfGraphConvertor with training: " << training_ << ", distribute: " << distribute_;
  }

  ~DfGraphConvertor() {}

  static void RegisterAdapter(const std::string &name, OpAdapterPtr adpt);
  static void RegisterAdapter(const std::string &name, OpAdapterPtr train_adpt, OpAdapterPtr infer_adpt);

  void DrawComputeGraph(const std::string &name) {
    std::ofstream fout(name);
    if (!fout.is_open()) {
      MS_LOG(ERROR) << "Open file '" << name << "' failed!";
      return;
    }
    fout << compute_sout_.str();
    fout.close();
  }
  void DrawInitGraph(const std::string &name) {
    std::ofstream fout(name);
    if (!fout.is_open()) {
      MS_LOG(ERROR) << "Open file '" << name << "' failed!";
      return;
    }
    fout << init_sout_.str();
    fout.close();
  }
  void DrawSaveCheckpointGraph(const std::string &name) {
    std::ofstream fout(name);
    if (!fout.is_open()) {
      MS_LOG(ERROR) << "Open file '" << name << "' failed!";
      return;
    }
    fout << checkpoint_sout_.str();
    fout.close();
  }

  DfGraphConvertor &ConvertAllNode();
  DfGraphConvertor &BuildGraph();
  DfGraphConvertor &InitParam(const TensorOrderMap &tensors);
  DfGraphConvertor &GenerateCheckpointGraph();
  DfGraphConvertor &GenerateBroadcastGraph(const TensorOrderMap &tensors);
  void InitParamWithData(const TensorOrderMap &tensors);
  void SetOpInput(const OpAdapterPtr &adpt, const CNodePtr &node);
  void SetupBroadcast(const std::shared_ptr<HcomBroadcast> &broadcast, const std::vector<GeTensorDesc> &broadcast_desc,
                      const DfGraphPtr &broadcast_graph, std::vector<ge::Operator> broadcast_input);
  void MakeDatasetHandler(const std::string &name, const size_t &input_idx, const AnfNodePtr &it);
  void SetupParamInitSubGraph(const TensorOrderMap &tensors, std::vector<ge::Operator> *init_input);
  void DrawParamInitSubGraph(const std::string &name, const AnfNodePtr &it);

  DfGraphPtr GetComputeGraph();
  DfGraphPtr GetInitGraph();
  DfGraphPtr GetSaveCheckpointGraph();
  DfGraphPtr GetBroadcastGraph();
  static OpAdapterPtr FindAdapter(const std::string &op_name, bool train = false);
  static OpAdapterPtr FindAdapter(AnfNodePtr node, bool train = false);
  int ErrCode() const { return static_cast<int>(error_); }

  bool is_training() const { return training_; }
  void set_training(bool is_training) { training_ = is_training; }

 protected:
  void InitLoopVar(std::vector<ge::Operator> *init_input);

 private:
  std::ostringstream compute_sout_;
  std::ostringstream init_sout_;
  std::ostringstream checkpoint_sout_;
  std::ostringstream restore_checkpoint_sout_;
  std::unordered_map<AnfNode *, std::string> op_draw_name_;

  AnfNodePtr TraceTupleGetItem(const CNodePtr &node, uint64_t *index);
  AnfNodePtr TraceMakeTuple(const CNodePtr &node, uint64_t index);
  AnfNodePtr TraceDepend(const CNodePtr &node);
  OutHandler TraceRealOp(AnfNodePtr node);
  OutHandler GetHandler(const AnfNodePtr &node, const std::stack<uint64_t> &index_stack, AnfNode *const draw_index);
  OperatorPtr Convert(AnfNodePtr node);
  OperatorPtr ConvertCNode(CNodePtr node);
  std::vector<OperatorPtr> ConvertDependNode(AnfNodePtr node);
  AnfNodePtr GetRealOpNode(AnfNodePtr node);
  std::vector<AnfNodePtr> GetDependNodes(const AnfNodePtr &node);
  OperatorPtr ConvertParameter(AnfNodePtr node);
  Status TryConvertValueNodeToMultiConst(const ValueNodePtr node);
  OperatorPtr ConvertValueNode(ValueNodePtr node);
  void GetCaseNodeInput(const CNodePtr node, const CNodePtr input_node);
  void ConvertTupleGetItem(const CNodePtr node);
  void GetDependOnParameterUse(const CNodePtr &node, const AnfNodePtr &src_node, const AnfNodePtr &dest_node,
                               const std::shared_ptr<std::vector<OperatorPtr>> &src_ops_list,
                               const std::shared_ptr<std::vector<OperatorPtr>> &dst_ops_list);
  bool GetControlDependList(const CNodePtr &node, const std::shared_ptr<std::vector<OperatorPtr>> &src_ops_list,
                            const std::shared_ptr<std::vector<OperatorPtr>> &dst_ops_list);
  void DrawControlDepend(const AnfNodePtr &src_node, const AnfNodePtr &dest_node);
  void ConvertControlDependNode(const CNodePtr node);
  void ConvertMakeTuple(const CNodePtr node);
  void ConvertTopK(const CNodePtr node);
  bool CheckCNode(const std::string &name, const CNodePtr node);
  void TraceOutput(AnfNodePtr node);
  void TraceOutputFromParameter(const AnfNodePtr &anf_out);
  void TraceOutputFromTupleGetItem(const AnfNodePtr &anf_out);
  void SetNodeInput(AnfNodePtr node);
  void SetOpControlInput(const AnfNodePtr &node);
  void UpdateOpDesc(AnfNodePtr node);
  void SetSubgraph(AnfNodePtr node);
  void ProcessSubgraph(AnfNodePtr node, const std::vector<AnfNodePtr> &inputs);
  void BuildSaveCheckpointGraph();
  void DrawCNode(const CNodePtr node, const OpAdapterPtr adpt);
  void UpdateDataOpDesc(const AnfNodePtr &it, const OperatorPtr &op) const;
  void AddGraphConstInput(const OperatorPtr &op);
  OperatorPtr ToOperatorPtr(const AnfNodePtr &node);
  bool IsSourceEdgeNode(const AnfNodePtr &node);
  bool IsControlEdgeNode(const AnfNodePtr &node);
  void AddEdgeForLoad(const AnfNodePtr &node);
  void AddEdgeToCache(const AnfNodePtr &src, const AnfNodePtr &dest);
  void FindDestOps(const AnfNodePtr &node, const std::shared_ptr<std::vector<AnfNodePtr>> &node_list, bool top);
  AnfNodePtr ParseLoadInput(const CNodePtr &cnode);
  void AutoMonadSetControlInput(const AnfNodePtr &node);
  void AutoMonadCollectInput(const AnfNodePtr &node);
  void AutoMonadSetInput(const AnfNodePtr &node);
  void SetTupleOpInput(const OpAdapterPtr &adpt, const CNodePtr &node, const AnfNodePtr &pred, const OperatorPtr &src,
                       int index);
  void UpdateTupleOutCache(void);
  AnfNodePtr GetRealInputNode(const CNodePtr &node, const AnfNodePtr &input);

  std::shared_ptr<AnfGraph> anf_graph_{nullptr};
  std::shared_ptr<DfGraph> df_graph_{nullptr};
  std::shared_ptr<DfGraph> init_graph_{nullptr};
  std::shared_ptr<DfGraph> save_ckp_graph_{nullptr};
  std::shared_ptr<DfGraph> restore_ckp_graph_{nullptr};
  std::shared_ptr<DfGraph> broadcast_graph_{nullptr};
  std::unordered_map<AnfNode *, DfGraph> branches_map_;
  std::unordered_map<AnfNode *, OperatorPtr> op_cache_;
  std::unordered_map<AnfNode *, std::vector<ControlEdge>> control_depend_cache_;
  std::unordered_map<AnfNodePtr, std::set<AnfNodePtr>> monad_control_edge_cache_;
  /* record "tuple_getitem"<->"out_handler" mapping */
  std::unordered_map<AnfNode *, OutHandler> out_handle_cache_;
  /* record "make_tuple"<->"out_handler vector" mapping */
  std::unordered_map<AnfNode *, std::shared_ptr<std::vector<OutHandler>>> tuple_out_handle_cache_;
  std::unordered_map<AnfNode *, std::shared_ptr<std::vector<AnfNodePtr>>> case_input_handle_cache_;
  std::unordered_map<std::string, AnfNodePtr> params_;
  std::unordered_map<std::string, OperatorPtr> vars_;
  std::vector<std::pair<ge::Operator, std::string>> graph_outputs_;
  std::vector<OperatorPtr> graph_const_inputs_;
  std::vector<OperatorPtr> init_ops_;
  std::vector<OperatorPtr> broadcast_ops_;
  std::vector<AnfNodePtr> inputs_;
  OperatorPtr dataset_iter_getnext_;
  Status error_ = SUCCESS;
  bool training_ = false;
  bool distribute_ = false;
  bool use_inputs_ = false;
};
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_CONVERT_H_
