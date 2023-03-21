/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include <cstdlib>
#include <memory>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <utility>
#include <stack>
#include <fstream>
#include <sstream>

#include "include/common/utils/config_manager.h"
#include "utils/hash_map.h"
#include "utils/ms_context.h"
#include "ops/core_ops.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/tensor.h"
#include "transform/graph_ir/df_graph_manager.h"
#include "transform/graph_ir/op_adapter.h"
#include "graph/operator_reg.h"
#include "external/ge/ge_api.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "ops/hcom_ops.h"

namespace mindspore {
namespace transform {
class BaseOpAdapter;
using HcomBroadcast = ::ge::op::HcomBroadcast;

using ParamIndexMap = std::map<std::size_t, std::size_t>;
enum class GraphType { kNormal, kCond, kBody, kAfter, kBranch };
enum class DfsVisitFlag { kUnVisited, kVisiting, kVisited };
constexpr char kFuncGraphTypeName[] = "FuncGraph";
constexpr char kKernelGraphTypeName[] = "KernelGraph";

class GeOpConvertor {
 public:
  static std::map<std::string, ValuePtr> GetAttrAndValue(const AnfNodePtr &node, const bool training);

  static std::string GetOpType(const AnfNodePtr &node, const bool training);

  static std::shared_ptr<GeTensorDesc> GetTensorDesc(const ShapeVector &dev_shape, const TypeId &dev_type,
                                                     const std::string &dev_format, const ShapeVector &ori_shape,
                                                     const std::string &ori_format);

  static mindspore::HashMap<std::string, std::string> GetNeedAddInput(const AnfNodePtr &node, const bool training);

  static bool IsDynamicInput(const AnfNodePtr &node, const size_t idx);

  static std::map<int, std::string> GetAclInputNames(const AnfNodePtr &node);

  static std::map<int, std::string> GetAclOutputNames(const AnfNodePtr &node);

  static std::map<int, std::string> GetAclDynamicInputNames(const AnfNodePtr &node);

  static std::map<int, std::string> GetAclDynamicOutputNames(const AnfNodePtr &node);
};

class DfGraphConvertor {
 public:
  explicit DfGraphConvertor(const AnfGraphPtr &anf_graph) : anf_graph_(anf_graph) {
    MS_EXCEPTION_IF_NULL(anf_graph);
    auto env_ge = mindspore::common::GetEnv("MS_ENABLE_GE");
    auto env_training = mindspore::common::GetEnv("MS_GE_TRAIN");
    if (env_ge == "1" && env_training == "1") {
      training_ = true;
    } else {
      training_ = anf_graph->has_flag("training");
    }
    distribute_ = anf_graph->has_flag("broadcast_flag");
    if (anf_graph->has_flag("broadcast_flag")) {
      ConfigManager::GetInstance().set_parallel_strategy(ParallelStrategy::DISTRIBUTION);
    } else {
      ConfigManager::GetInstance().set_parallel_strategy(ParallelStrategy::ONE_DEVICE);
    }
    is_kernel_graph_ = anf_graph_->type_name() == kKernelGraphTypeName;
    df_graph_ = std::make_shared<DfGraph>(anf_graph_->ToString());

    AddGraphDynamicInputs(anf_graph_->parameters());

    std::string graph_type = is_kernel_graph_ ? "kernel_graph" : "func_graph";
    std::string graph_name = anf_graph_->ToString();
    MS_LOG(INFO) << "Create DfGraphConvertor with graph: " << graph_name << "(type: " << graph_type << ")"
                 << ", training: " << training_ << ", dynamic input: " << dynamic_shape_inputs_
                 << ", distribute: " << distribute_;
  }

  ~DfGraphConvertor() {}

  static void RegisterAdapter(const std::string &name, OpAdapterPtr adpt);
  static void RegisterAdapter(const std::string &name, OpAdapterPtr train_adpt, OpAdapterPtr infer_adpt);

  void DrawComputeGraph(const std::string &name) {
#ifndef ENABLE_SECURITY
    std::ofstream fout(name);
    if (!fout.is_open()) {
      MS_LOG(ERROR) << "Open file '" << name << "' failed!";
      return;
    }
    fout << compute_sout_.str();
    fout.close();
#endif
  }

  void DrawInitGraph(const std::string &name) {
#ifndef ENABLE_SECURITY
    std::ofstream fout(name);
    if (!fout.is_open()) {
      MS_LOG(ERROR) << "Open file '" << name << "' failed!";
      return;
    }
    fout << init_sout_.str();
    fout.close();
#endif
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
  DfGraphConvertor &BuildGraph(const std::string &name);
  DfGraphConvertor &InitParam(const TensorOrderMap &tensors);
  DfGraphConvertor &GenerateCheckpointGraph();
  DfGraphConvertor &GenerateBroadcastGraph(const TensorOrderMap &tensors);
  void InitParamWithConst(const TensorOrderMap &tensors);
  OutHandler GetNormalOpInput(const AnfNodePtr &node, const AnfNodePtr &pred);
  void DrawOpInput(const AnfNodePtr &node, const AnfNodePtr &pred, size_t i);
  void SetOpInput(const OpAdapterPtr &adpt, const CNodePtr &node);
  void SetupBroadcast(const std::shared_ptr<HcomBroadcast> &broadcast, const std::vector<GeTensorDesc> &broadcast_desc,
                      const DfGraphPtr &broadcast_graph, std::vector<::ge::Operator> broadcast_input);
  void MakeDatasetHandler(const std::string &name, const size_t &input_idx, const AnfNodePtr &it);
  void SetupParamInitSubGraph(const TensorOrderMap &tensors, const std::vector<::ge::Operator> *init_input);
  void DrawParamInitSubGraph(const std::string &name, const AnfNodePtr &it);

  DfGraphPtr GetComputeGraph();
  DfGraphPtr GetInitGraph();
  DfGraphPtr GetSaveCheckpointGraph();
  DfGraphPtr GetBroadcastGraph();
  int ErrCode() const { return static_cast<int>(error_); }

  bool is_training() const { return training_; }
  void set_training(bool is_training) { training_ = is_training; }
  bool dynamic_shape_inputs() const { return dynamic_shape_inputs_; }
  std::vector<ShapeVector> input_shapes() { return input_shapes_; }

 protected:
  void InitLoopVar(std::vector<::ge::Operator> *init_input);

 private:
  std::ostringstream compute_sout_;
  std::ostringstream init_sout_;
  std::ostringstream checkpoint_sout_;
  std::ostringstream restore_checkpoint_sout_;
  mindspore::HashMap<AnfNode *, std::string> op_draw_name_;
  std::map<std::string, std::string> param_format_;

  AnfNodePtr TraceTupleGetItem(const CNodePtr &node, uint64_t *index);
  OutHandler GetHandler(const AnfNodePtr &node);
  OperatorPtr Convert(AnfNodePtr node);
  OperatorPtr ConvertCNode(CNodePtr node);
  OperatorPtr ConvertParameter(AnfNodePtr node);
  void SetNodeAbstract(const CNodePtr &node) const;
  Status TryConvertValueNodeToMultiConst(const ValueNodePtr node);
  OperatorPtr ConvertValueNode(ValueNodePtr node);
  void SaveParamFormat(CNodePtr node);
  void GetBranchNodeInput(const CNodePtr node);
  void ConvertTopK(const CNodePtr &node);
  void ConvertSpaceBatchNd(const FuncGraphPtr anf_graph) const;
  AnfNodePtr CreateCast(const AnfNodePtr &input, const TypePtr &dst_type) const;
  void ConvertReshape(const CNodePtr &node);
  void ConvertAllReduce(const CNodePtr &node);
  void ConvertOCRRecPreHandle(const CNodePtr &node);
  void ConvertConv2D(const CNodePtr &node);
  void ConvertDynamicStitch(const CNodePtr &node);
  std::vector<int64_t> CastToInt(const ValuePtr &value) const;
  void TransDataType(const FuncGraphPtr &anf_graph) const;
  void TransInputDataType(const CNodePtr &node, std::string node_name) const;
  void TransAttrDataType(const CNodePtr &node, const std::string &node_name) const;
  bool CheckCNode(const std::string &name, const CNodePtr node);
  void SetNodeInput(AnfNodePtr node);
  void UpdateOpDesc(AnfNodePtr node);
  void SetSubgraph(const AnfNodePtr &node);
  void ProcessSubgraph(const AnfNodePtr &node, const AnfNodePtr &branch_node, ParamIndexMap &branch_to_parent_node_map);
  void BuildSaveCheckpointGraph();
  void DrawCNode(const CNodePtr node, const OpAdapterPtr adpt);
  void UpdateDataOpDesc(const AnfNodePtr &it, const OperatorPtr &op) const;
  void UpdateConstOpDesc(const AnfNodePtr &it, const OperatorPtr &op) const;
  void AddGraphConstInput(const OperatorPtr &op);
  void AddGraphDynamicInputs(const AnfNodePtrList &params);
  AnfNodePtr ParseLoadInput(const CNodePtr &cnode) const;
  void SetGraphInputs(std::vector<Operator> *inputs);
  void TransformConstOp(const CNodePtr &node, const AnfNodePtr &pred);
  AnfNodePtr GetRealInputNode(const CNodePtr &node, const AnfNodePtr &input);
  void SetupDatasetIterGetNextNode();

  void ConvertWhileNode(const CNodePtr &node);
  void CacheWhileGraph(const CNodePtr &cnode);
  void ConvertWhileBody(const AnfNodePtr &node);
  std::shared_ptr<std::vector<Operator>> GetWhileSubGraphInput();
  void BuildWhileSubGraph();
  void ConvertWhileCond(const AnfNodePtr &node);
  void ConvertWhileAfter(const AnfNodePtr &node);
  void BuildWhileAfterSubGraph();
  void GetCallNodeInputs(const CNodePtr &node);
  std::vector<Operator> GetWhileBodyOutputs();
  bool IsSubGraph() const { return graph_type_ == GraphType::kCond || graph_type_ == GraphType::kBody; }
  bool IsCondGraph() const { return graph_type_ == GraphType::kCond; }
  bool IsBodyGraph() const { return graph_type_ == GraphType::kBody; }
  bool IsBranchGraph() const { return graph_type_ == GraphType::kBranch; }
  bool IsAfterGraph() const { return graph_type_ == GraphType::kAfter; }
  bool IsNormalGraph() const { return graph_type_ == GraphType::kNormal; }
  void SetParamIndexMap(const std::vector<AnfNodePtr> &graphs);
  void SetWhileOutputHandle(const OperatorPtr &prev_while_op);
  void GetWhileUsedInputIndex(const std::vector<AnfNodePtr> &graphs);

  bool IsDataInput(const AnfNodePtr &node, const AnfNodePtr &input, size_t input_index) const;
  void SetMakeTupleInput(const OpAdapterPtr &adpt, const CNodePtr &make_tuple_node);
  void SetMergeInput(const OpAdapterPtr &adpt, const CNodePtr &merge_node);
  bool IsMergeOrSwitchLayerInput(const CNodePtr &node) const;
  void SetDynamicInputHandleByMultiInput(const OpAdapterPtr &adpt, const CNodePtr &node,
                                         const CNodePtr &from_node_input);
  void SetNodeControlInput(const AnfNodePtr &node, const AnfNodePtr &input);
  void SetGraphOutputs(bool is_main_graph = false);
  std::vector<OutHandler> GetInputHandles(const AnfNodePtr &node, const AnfNodePtr &input);

  // Identity Optimization
  void IdentityOptimization();
  std::string GetGNodeName(const ::ge::GNode &node) const;
  std::string GetGNodeType(const ::ge::GNode &node) const;
  bool IsIdentityRedundant(const ::ge::GNode &node) const;
  void RemoveIdentity(::ge::GNode identity_node);
  void NoOpOptimization();
  bool IsNoOpRedundant(const ::ge::GNode &node) const;
  void RemoveNoOp(::ge::GNode noop);
  std::shared_ptr<std::vector<DfGraph>> BuildBranchGraphs(const CNodePtr &cnode);

  std::shared_ptr<AnfGraph> anf_graph_{nullptr};
  std::shared_ptr<DfGraph> df_graph_{nullptr};
  std::shared_ptr<DfGraph> init_graph_{nullptr};
  std::shared_ptr<DfGraph> save_ckp_graph_{nullptr};
  std::shared_ptr<DfGraph> restore_ckp_graph_{nullptr};
  std::shared_ptr<DfGraph> broadcast_graph_{nullptr};
  mindspore::HashMap<AnfNode *, DfGraph> branches_map_;
  mindspore::HashMap<AnfNode *, OperatorPtr> op_cache_;
  /* record "getnext"<->"out_handler" mapping */
  mindspore::HashMap<AnfNode *, OutHandler> out_handle_cache_;
  /* record "value tuple"<->"out_handler vector" mapping */
  mindspore::HashMap<AnfNode *, std::shared_ptr<std::vector<OutHandler>>> tuple_out_handle_cache_;
  mindspore::HashMap<AnfNode *, std::shared_ptr<std::vector<AnfNodePtr>>> branch_input_handle_cache_;
  mindspore::HashMap<std::string, AnfNodePtr> params_;
  mindspore::HashMap<std::string, OperatorPtr> vars_;
  std::vector<std::pair<::ge::Operator, std::string>> graph_outputs_;
  std::vector<AnfNodePtr> graph_anf_outputs_;
  std::vector<OperatorPtr> graph_const_inputs_;
  std::vector<OperatorPtr> init_ops_;
  std::vector<OperatorPtr> broadcast_ops_;
  std::vector<AnfNodePtr> inputs_;
  ShapeArray input_shapes_;
  OperatorPtr dataset_iter_getnext_;
  OperatorPtr queue_data_;
  OperatorPtr get_next_from_queue_;
  Status error_ = SUCCESS;
  bool training_ = false;
  bool distribute_ = false;
  bool use_inputs_ = false;
  bool dynamic_shape_inputs_ = false;

  AnfNodePtr while_cond_node_ = nullptr;
  mindspore::HashMap<AnfNodePtr, std::shared_ptr<std::vector<DfGraph>>> while_dfgraph_cache_;

  CNodePtr cur_while_node_ = nullptr;
  size_t cur_while_node_out_size_ = 0;
  mindspore::HashMap<size_t, OutHandler> while_const_input_index_;
  mindspore::HashMap<size_t, OutHandler> prev_while_const_input_index_;
  mindspore::HashMap<size_t, size_t> prev_cond_to_while_out_index_;
  mindspore::HashMap<OperatorPtr, std::shared_ptr<tensor::Tensor>> const_op_to_value_;
  AnfNodePtr prev_while_node_ = nullptr;
  size_t prev_while_node_out_size_ = 0;

  mindspore::HashMap<AnfNodePtr, std::vector<AnfNodePtr>> while_graph_cache_;
  mindspore::HashMap<AnfNodePtr, std::shared_ptr<std::vector<OutHandler>>> call_input_handle_cache_;
  mindspore::HashMap<AnfNodePtr, std::shared_ptr<std::vector<OutHandler>>> while_output_handle_cache_;
  AnfNodePtr call_node_in_while_body_ = nullptr;
  GraphType graph_type_ = GraphType::kNormal;

  ParamIndexMap body_cond_map_;
  ParamIndexMap after_cond_map_;
  ParamIndexMap prev_after_cond_map_;
  mindspore::HashMap<size_t, OperatorPtr> subgraph_input_cache_;

  std::set<size_t> while_used_input_index_;
  std::set<size_t> prev_while_used_input_index_;

  mindspore::HashMap<size_t, OutHandler> bypass_node_prev_handle_cache_;
  mindspore::HashMap<size_t, OutHandler> bypass_node_handle_cache_;
  size_t case_call_input_size_ = 0;
  bool is_kernel_graph_ = false;
};
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_CONVERT_H_
