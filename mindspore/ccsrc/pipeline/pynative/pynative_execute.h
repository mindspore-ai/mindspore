/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_EXECUTE_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_EXECUTE_H_

#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <stack>
#include <set>
#include <map>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include "pybind_api/ir/base_ref_py.h"
#include "pipeline/pynative/base.h"
#include "utils/ms_context.h"
#include "ir/anf.h"
#include "pipeline/jit/resource.h"
#include "frontend/operator/composite/composite.h"

namespace mindspore {
namespace pynative {
namespace py = pybind11;
using ResourcePtr = std::shared_ptr<pipeline::Resource>;
using GradOperationPtr = std::shared_ptr<prim::GradOperation>;

struct PrimAbsInfo {
  abstract::AbstractBasePtr abs;
  bool is_dynamic_shape = false;
  std::unordered_map<std::string, ValuePtr> attrs;
};

using AbstractListMap = std::unordered_map<abstract::AbstractBasePtrList, PrimAbsInfo,
                                           abstract::AbstractBasePtrListHasher, abstract::AbstractBasePtrListEqual>;
using OpIndexWithTensorId = std::unordered_map<std::string, std::vector<std::string>>;
using TensorIdWithTensor = std::unordered_map<std::string, std::vector<tensor::TensorPtr>>;

py::object RunOp(const py::args &args);

void ClearPyNativeSession();

struct GraphInfo {
  std::string cell_id;
  AnfNodePtr output;
  OrderedMap<std::string, ParameterPtr> params;  // hold input parameters and cell weights
  std::unordered_map<std::string, std::pair<AnfNodePtr, std::vector<int64_t>>> node_map;
  std::vector<std::string> objects;
  GraphInfo() = default;
  explicit GraphInfo(std::string id) : cell_id(std::move((id))) {}
};
using GraphInfoPtr = std::shared_ptr<GraphInfo>;

class CellInfo {
 public:
  CellInfo() = default;
  ~CellInfo() = default;
  CellInfo(bool custom_bprop, bool has_dynamic, FuncGraphPtr foward_graph, std::string cellid, std::string bprop_id)
      : is_custom_bprop_(custom_bprop),
        is_dynamic_(has_dynamic),
        fg_(std::move(foward_graph)),
        cell_id_(std::move(cellid)),
        bprop_cell_id_(std::move(bprop_id)) {}

  bool is_custom_bprop() const { return is_custom_bprop_; }
  void set_is_custom_bprop(bool is_custom_bprop) { is_custom_bprop_ = is_custom_bprop; }
  bool is_dynamic() const { return is_dynamic_; }
  void set_is_dynamic(bool is_dynamic) { is_dynamic_ = is_dynamic; }
  size_t call_times() const { return call_times_; }
  void set_call_times(size_t call_times) { call_times_ = call_times; }
  FuncGraphPtr fg() const { return fg_; }
  void set_fg(FuncGraphPtr fg) { fg_ = std::move(fg); }
  std::string &cell_id() { return cell_id_; }
  void set_cell_id(std::string cell_id) { cell_id_ = std::move(cell_id); }
  std::string &bprop_cell_id() { return bprop_cell_id_; }
  std::vector<std::string> &cell_ops_info() { return cell_ops_info_; }

 private:
  bool is_custom_bprop_{false};  // Custom bprop
  bool is_dynamic_{false};       // Set by has_dynamic_cell
  size_t call_times_{0};
  FuncGraphPtr fg_{nullptr};  // Forward graph
  std::string cell_id_;
  std::string bprop_cell_id_;
  std::vector<std::string> cell_ops_info_;  // All ops info
};
using CellInfoPtr = std::shared_ptr<CellInfo>;

class TopCellInfo {
 public:
  TopCellInfo() = default;
  ~TopCellInfo() = default;
  TopCellInfo(bool topest, ResourcePtr r, FuncGraphPtr df, std::string cellid)
      : is_topest_(topest), resource_(std::move(r)), df_builder_(std::move(df)), cell_id_(std::move(cellid)) {}

  bool is_grad() const { return is_grad_; }
  void set_is_grad(bool is_grad) { is_grad_ = is_grad; }
  bool is_topest() const { return is_topest_; }
  bool vm_compiled() const { return vm_compiled_; }
  void set_vm_compiled(bool vm_compiled) { vm_compiled_ = vm_compiled; }
  bool need_grad() const { return need_grad_; }
  void set_need_grad(bool need_grad) { need_grad_ = need_grad; }
  bool has_dynamic_cell() const { return has_dynamic_cell_; }
  bool is_real_dynamic() const { return is_real_dynamic_; }
  void set_is_real_dynamic(bool is_real_dynamic) { is_real_dynamic_ = is_real_dynamic; }
  bool forward_already_run() const { return forward_already_run_; }
  void set_forward_already_run(bool set_forward_already_run) { forward_already_run_ = set_forward_already_run; }
  ResourcePtr resource() { return resource_; }
  FuncGraphPtr df_builder() { return df_builder_; }
  std::string &cell_id() { return cell_id_; }
  std::string &sens_id() { return sens_id_; }
  void set_sens_id(std::string sens_id) { sens_id_ = std::move(sens_id); }
  std::string &weights_id() { return weights_id_; }
  void set_weights_id(std::string weights_id) { weights_id_ = std::move(weights_id); }
  std::string &input_args_id() { return input_args_id_; }
  void set_input_args_id(std::string input_args_id) { input_args_id_ = std::move(input_args_id); }
  std::vector<CellInfoPtr> &cell_graph_list() { return cell_graph_list_; }
  void set_cell_graph_list(const std::vector<CellInfoPtr> &cell_graph_list) { cell_graph_list_ = cell_graph_list; }
  OrderedMap<FuncGraphPtr, GraphInfoPtr> &graph_info_map() { return graph_info_map_; }
  void set_graph_info_map(const OrderedMap<FuncGraphPtr, GraphInfoPtr> &graph_info_map) {
    graph_info_map_ = graph_info_map;
  }
  void clear() {
    cell_graph_list_.clear();
    graph_info_map_.clear();
  }

 private:
  bool is_grad_{false};  // Derivative is calculated
  bool is_topest_{false};
  bool vm_compiled_{false};
  bool need_grad_{true};
  bool has_dynamic_cell_{false};
  bool is_real_dynamic_{false};
  bool forward_already_run_{false};
  ResourcePtr resource_{nullptr};
  FuncGraphPtr df_builder_{nullptr};
  std::string cell_id_;
  std::string sens_id_;
  std::string weights_id_;
  std::string input_args_id_;
  std::vector<CellInfoPtr> cell_graph_list_;
  OrderedMap<FuncGraphPtr, GraphInfoPtr> graph_info_map_;
};
using TopCellInfoPtr = std::shared_ptr<TopCellInfo>;

class DynamicAnalysis;
using DynamicAnalysisPtr = std::shared_ptr<DynamicAnalysis>;

class ForwardExecutor;
using ForwardExecutorPtr = std::shared_ptr<ForwardExecutor>;
using ForwardExecutorWeakPtr = std::weak_ptr<ForwardExecutor>;

class GradExecutor;
using GradExecutorPtr = std::shared_ptr<GradExecutor>;
using GradExecutorWeakPtr = std::weak_ptr<GradExecutor>;

class DynamicAnalysis {
 public:
  DynamicAnalysis() = default;
  ~DynamicAnalysis() = default;

  // Check cell struct
  bool IsDynamicCell(const py::object &cell);

 private:
  std::string GetCellInfo(const py::object &cell);
  void ParseInputArgs(const std::shared_ptr<parse::ParseAst> &ast, const py::object &fn_node);
  bool ParseBodyContext(const std::shared_ptr<parse::ParseAst> &ast, const py::object &fn_node,
                        const std::vector<std::string> &compare_prim = {});
  bool ParseIfWhileExprNode(const std::shared_ptr<parse::ParseAst> &ast, const py::object &node);
  bool ParseAssignExprNode(const std::shared_ptr<parse::ParseAst> &ast, const py::object &node);
  bool ParseAugAssignExprNode(const std::shared_ptr<parse::ParseAst> &ast, const py::object &node,
                              const std::vector<std::string> &compare_prim = {});
  bool ParseForExprNode(const std::shared_ptr<parse::ParseAst> &ast, const py::object &node);
  std::string ParseNodeName(const std::shared_ptr<parse::ParseAst> &ast, const py::object &node,
                            parse::AstMainType type);

  std::unordered_set<std::string> cell_input_args_;
};

class GradExecutor {
 public:
  GradExecutor() = default;
  ~GradExecutor() = default;
  explicit GradExecutor(const ForwardExecutorPtr &forward_executor = nullptr)
      : forward_executor_(ForwardExecutorWeakPtr(forward_executor)) {}

  std::function<void(py::object *, const py::object &, const py::args &)> InitGraph = [this](auto &&PH1, auto &&PH2,
                                                                                             auto &&PH3) {
    NewGraphInner(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2), std::forward<decltype(PH3)>(PH3));
  };
  std::function<void(py::object *, const py::object &, const py::object &, const py::args &)> LinkGraph =
    [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
      EndGraphInner(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2),
                    std::forward<decltype(PH3)>(PH3), std::forward<decltype(PH4)>(PH4));
    };
  std::function<void(py::object *, const GradOperationPtr &, const py::object &, const py::object &, const py::args &)>
    GradGraph = [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4, auto &&PH5) {
      GradNetInner(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2), std::forward<decltype(PH3)>(PH3),
                   std::forward<decltype(PH4)>(PH4), std::forward<decltype(PH5)>(PH5));
    };
  std::function<void(py::object *, const py::object &, const py::tuple &, const py::object &)> RunGraph =
    [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
      RunGradGraph(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2), std::forward<decltype(PH3)>(PH3),
                   std::forward<decltype(PH4)>(PH4));
    };

  FuncGraphPtr curr_g() const;
  TopCellInfoPtr top_cell() const;
  bool TopCellIsDynamic();
  void set_top_cell(TopCellInfoPtr top_cell) { top_cell_ = std::move(top_cell); }
  bool grad_flag() const { return grad_flag_; }
  void set_grad_flag(bool flag) { grad_flag_ = flag; }
  bool in_grad_process() const { return in_grad_process_; }
  std::string top_cell_id() { return top_cell()->cell_id(); }
  AnfNodePtr GetInput(const py::object &obj, bool op_mask);
  std::string GetCellId(const py::object &obj, const py::args &args);
  TopCellInfoPtr GetTopCell(const string &cell_id, bool find_nearest = false);
  void SaveOutputNodeMap(const std::string &obj_id, const py::object &out_real, const AnfNodePtr &cnode);
  void SaveAllResult(const OpExecInfoPtr &op_exec_info, const AnfNodePtr &node, const py::object &out_real);
  py::object CheckGraph(const py::object &cell, const py::args &args);
  void RunGradGraph(py::object *ret, const py::object &cell, const py::tuple &args, const py::object &phase);
  bool need_construct_graph() const { return !graph_stack_.empty() && grad_flag_; }
  void set_dynamic_analysis(DynamicAnalysisPtr dynamic_analysis) { dynamic_analysis_ = std::move(dynamic_analysis); }
  std::stack<FuncGraphPtr> &graph_stack() { return graph_stack_; }
  std::vector<TopCellInfoPtr> &top_cell_list() { return top_cell_list_; }
  bool need_replace_forward() const { return need_replace_forward_; }
  std::stack<std::string> &cell_op_info_stack() { return cell_op_info_stack_; }
  std::unordered_map<std::string, size_t> &op_index_map() { return op_index_map_; }
  std::unordered_map<std::string, std::string> &obj_to_forward_id() { return obj_to_forward_id_; }
  void ClearGrad(const py::object &cell, const py::args &args);
  void ClearRes();
  void ClearCellRes(const std::string &cell_id = "");

 private:
  ForwardExecutorPtr forward() const;
  DynamicAnalysisPtr dynamic_analysis() const;
  bool grad_running() const { return grad_is_running_; }
  void set_grad_runing(bool grad_runing) { grad_is_running_ = grad_runing; }
  void set_need_replace_forward(bool need_replace_forward) { need_replace_forward_ = need_replace_forward; }

  // Higher derivative
  bool IsNestedGrad() const;
  void AddNestedGradOrder() { ++grad_order_; }
  void SubNestedGradOrder();
  void ReplaceGraphParams(const FuncGraphPtr &df_builder, const FuncGraphPtr &forward_graph,
                          const std::string &cell_id);
  void SetNestedTopGraph(const py::object &cell, const py::args &args, const std::string &cell_id);
  void MakeNestedCnode(const std::string &cell_id, const py::args &args, const ResourcePtr &resource,
                       const py::object &out, bool has_sens);
  void RecoverGraphParams(const FuncGraphPtr &newfg, const std::string &cell_id, std::vector<AnfNodePtr> *inputs);
  bool MakeBpropNestedCnode(const py::object &cell, const py::object &out, const std::string &cell_id);

  // Dynamic
  bool CheckDynamicCell(const std::string &cell_id);
  bool CheckRealDynamicCell(const std::string &cell_id);
  void ClearDynamicTopRes(const std::string &cell_id);

  void PushCurrentGraphToStack();
  void PopGraphStack();
  void PushCurrentCellOpInfoToStack();
  void PopCurrentCellOpInfoFromStack();
  std::string GetCellOpInfo();
  void ReplaceCellOpInfoByCellId(const std::string &cell_id);

  FuncGraphPtr GetDfbuilder(const std::string &cell_id = "");
  ResourcePtr GetResource(const std::string &cell_id = "");
  bool IsFirstGradStep();
  bool IsTopGraph(const std::string &cell_id);
  bool IsTopestGraph(const std::string &cell_id);
  bool IsBpropGraph(const std::string &cell_id);
  bool IsGradBefore(const std::string &cell_id);
  bool CheckCellGraph(const std::string &cell_id);
  bool UpdateBpropCellGraph(const py::object &cell, const FuncGraphPtr &g, const std::string &cell_id, bool need_cloned,
                            bool is_grad);
  void UpdateCellGraph(const py::object &cell, const FuncGraphPtr &g, const std::string &cell_id,
                       bool need_cloned = false, bool is_grad = false);
  bool CheckCellChanged(const std::string &cell_id);
  void UpdateTopCellInfo(const std::string &cell_id, bool vm_compiled);
  void DumpGraphIR(const std::string &filename, const FuncGraphPtr &graph);
  void NewGraphInner(py::object *ret, const py::object &cell, const py::args &args);
  void MakeNewTopGraph(const string &cell_id, const py::args &args);
  void EndGraphInner(py::object *ret, const py::object &cell, const py::object &out, const py::args &args);
  void EndGraphByOutId(const py::object &cell, const std::string &cell_id, const py::object &out,
                       const std::string &out_id, const py::args &args);
  bool EndBpropGraph(const string &cell_id);
  FuncGraphPtr MakeGradGraph(const py::object &cell, const FuncGraphPtr &g, const ResourcePtr &r,
                             const std::string &cell_id, const py::args &args);
  std::string GetGradCellId(bool has_sens, const py::object &cell, const py::args &args, py::object *forward_args,
                            py::object *sens = nullptr);
  void GradNetInner(py::object *ret, const GradOperationPtr &grad, const py::object &cell, const py::object &weights,
                    const py::args &args);
  void SetTopCellTensorId(const std::string &cell_id);
  bool CheckGradParamsChanged(const std::string &cell_id, const py::object &weights, const py::object &sens);
  void SetGradGraphParams(const FuncGraphPtr &df_builder, const ResourcePtr &resource, size_t size);
  void SetGradGraph(const FuncGraphPtr &g, const GradOperationPtr &grad_op, const std::vector<AnfNodePtr> &weights,
                    size_t arg_size, const std::string &cell_id);
  std::vector<AnfNodePtr> GetWeightsArgs(const py::object &weights, const FuncGraphPtr &df_builder);
  abstract::AbstractBasePtrList GetArgsSpec(const py::args &args, const FuncGraphPtr &df_builder);
  void ClearUselessRes(const FuncGraphPtr &df_builder, const py::object &cell, const std::string &cell_id);
  void SetTupleItemArgsToGraphInfoMap(const FuncGraphPtr &g, const py::object &id, const AnfNodePtr &node,
                                      const std::vector<int64_t> &index_sequence, bool is_param = false);
  AnfNodePtr GetObjNode(const py::object &obj, const std::string &obj_id);
  AnfNodePtr MakeValueNode(const py::object &obj, const std::string &obj_id);

  // Memory clean between steps
  void ClearResidualRes(const std::string &cell_id);
  void ClearCnodeRes(const AnfNodePtr &node);
  void CleanPreMemoryInValueNode();
  void SaveTensorsInValueNode(const ResourcePtr &resource);
  void SaveAllValueNodeTensors(const FuncGraphPtr &graph);

  void SetPyObjInGraphInfoMap(const FuncGraphPtr &g, const std::string &obj) {
    top_cell()->graph_info_map()[g]->objects.push_back(obj);
  }
  void SetTupleArgsToGraphInfoMap(const FuncGraphPtr &g, const py::object &args, const AnfNodePtr &node,
                                  bool is_param = false);
  void SetParamNodeMapInGraphInfoMap(const FuncGraphPtr &g, const std::string &id, const ParameterPtr &param) {
    top_cell()->graph_info_map()[g]->params[id] = param;
  }
  void SetNodeMapInGraphInfoMap(const FuncGraphPtr &g, const std::string &id, const AnfNodePtr &node,
                                int64_t index = -1) {
    top_cell()->graph_info_map()[g]->node_map[id] = std::make_pair(node, std::vector<int64_t>{index});
  }
  void SetNodeMapInGraphInfoMap(const FuncGraphPtr &g, const std::string &id, const AnfNodePtr &node,
                                const std::vector<int64_t> &index) {
    top_cell()->graph_info_map()[g]->node_map[id] = std::make_pair(node, index);
  }

 private:
  size_t grad_order_{0};
  bool grad_flag_{false};
  bool in_bprop_process_{false};
  bool in_grad_process_{false};
  bool has_dynamic_cell_{false};
  bool need_replace_forward_{true};
  bool grad_is_running_{false};
  FuncGraphPtr curr_g_{nullptr};
  // For clear pre top res
  TopCellInfoPtr pre_top_cell_{nullptr};
  TopCellInfoPtr top_cell_{nullptr};
  std::unordered_map<std::string, size_t> op_index_map_;
  std::unordered_map<FuncGraphPtr, std::vector<std::pair<ParameterPtr, ParameterPtr>>> replace_weights_map_;
  std::unordered_set<tensor::TensorPtr> all_value_node_tensors_;
  std::unordered_map<std::string, std::string> obj_to_forward_id_;

  // Records forwrad graph, the bottom is top graph
  std::stack<FuncGraphPtr> graph_stack_;
  // Records op info of every cell, the bottom is op info of top cell
  std::stack<std::string> cell_op_info_stack_;

  // Use vector for keep order
  std::vector<TopCellInfoPtr> top_cell_list_;
  ForwardExecutorWeakPtr forward_executor_;
  DynamicAnalysisPtr dynamic_analysis_;
};

class ForwardExecutor {
 public:
  ForwardExecutor() = default;
  ~ForwardExecutor() = default;

  std::function<void(py::object *, const OpExecInfoPtr &)> RunOpS = [this](auto &&PH1, auto &&PH2) {
    RunOpInner(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2));
  };

  void RunOpInner(py::object *ret, const OpExecInfoPtr &op_exec_info);
  OpExecInfoPtr GenerateOpExecInfo(const py::args &args);
  void set_grad_executor(const GradExecutorPtr &grad_executor) { grad_executor_ = GradExecutorWeakPtr(grad_executor); }
  std::unordered_map<std::string, abstract::AbstractBasePtr> &node_abs_map() { return node_abs_map_; }
  std::unordered_map<std::string, OpIndexWithTensorId> &cell_op_index_with_tensor_id() {
    return cell_op_index_with_tensor_id_;
  }
  std::unordered_map<std::string, TensorIdWithTensor> &cell_tensor_id_with_tensor() {
    return cell_tensor_id_with_tensor_;
  }
  void ClearRes();

 private:
  GradExecutorPtr grad() const;
  MsBackendPolicy InitEnv(const OpExecInfoPtr &op_exec_info);
  py::tuple RunOpWithInitBackendPolicy(const OpExecInfoPtr &op_exec_info);
  py::object RunOpInVM(const OpExecInfoPtr &op_exec_info, PynativeStatusCode *status);
  py::object RunOpInMs(const OpExecInfoPtr &op_exec_info, PynativeStatusCode *status);
  py::object RunOpWithBackendPolicy(MsBackendPolicy backend_policy, const OpExecInfoPtr &op_exec_info,
                                    PynativeStatusCode *status);
  AnfNodePtr MakeCNode(const OpExecInfoPtr &op_exec_info, std::vector<int64_t> *op_masks,
                       abstract::AbstractBasePtrList *args_spec_list);
  bool FindOpMask(py::object obj, std::vector<int64_t> *op_masks, std::string id);
  void GetArgsSpec(const OpExecInfoPtr &op_exec_info, std::vector<int64_t> *op_masks, std::vector<AnfNodePtr> *inputs,
                   abstract::AbstractBasePtrList *args_spec_list);
  abstract::AbstractBasePtr CheckConstValue(const PrimitivePyPtr &prim, const py::object &obj,
                                            const abstract::AbstractBasePtr &abs, const std::string &id, size_t index);
  void GetOpOutputAbstract(const OpExecInfoPtr &op_exec_info, const abstract::AbstractBasePtrList &args_spec_list,
                           bool *is_find);
  // Update the abstract and device address info of value node and tensors in bprop graph
  void UpdateAbstractAndDeviceAddress(const OpExecInfoPtr &op_exec_info, const py::object &out_real);

  // Mix precision
  void RunParameterAutoMixPrecisionCast(const OpExecInfoPtr &op_exec_info);
  py::object DoParamMixPrecisionCast(bool *is_cast, const py::object &obj, const std::string &op_name, size_t index);
  py::object DoParamMixPrecisionCastTuple(bool *is_cast, const py::tuple &tuple, const std::string &op_name,
                                          size_t index);
  py::object DoAutoCast(const py::object &arg, const TypeId &type_id, const std::string &op_name, size_t index,
                        const std::string &obj_id);
  void DoSignatrueCast(const PrimitivePyPtr &prim, const std::map<SignatureEnumDType, TypeId> &dst_type,
                       const std::vector<SignatureEnumDType> &dtypes, const OpExecInfoPtr &op_exec_info);

 private:
  GradExecutorWeakPtr grad_executor_;
  std::unordered_map<std::string, AbstractListMap> prim_abs_list_;
  std::unordered_map<std::string, abstract::AbstractBasePtr> node_abs_map_;
  // Used for runop and replace forward result of grad graph
  std::unordered_map<std::string, OpIndexWithTensorId> cell_op_index_with_tensor_id_;
  std::unordered_map<std::string, TensorIdWithTensor> cell_tensor_id_with_tensor_;
  // Used to cache cast struct
  std::unordered_map<std::string, OpExecInfoPtr> cast_struct_map_;
  // Used to cache op_mask
  std::unordered_map<std::string, int64_t> op_mask_map_;
};

class PynativeExecutor : public std::enable_shared_from_this<PynativeExecutor> {
 public:
  static std::shared_ptr<PynativeExecutor> GetInstance() {
    std::lock_guard<std::mutex> i_lock(instance_lock_);
    if (executor_ == nullptr) {
      executor_ = std::shared_ptr<PynativeExecutor>(new (std::nothrow) PynativeExecutor());
      forward_executor_ = std::make_shared<ForwardExecutor>();
      grad_executor_ = std::make_shared<GradExecutor>(forward_executor_);
      grad_executor_->set_dynamic_analysis(std::make_shared<DynamicAnalysis>());
      forward_executor_->set_grad_executor(grad_executor_);
    }
    return executor_;
  }
  ~PynativeExecutor() = default;
  PynativeExecutor(const PynativeExecutor &) = delete;
  PynativeExecutor &operator=(const PynativeExecutor &) = delete;

  void EnterConstruct(const py::object &cell);
  void LeaveConstruct(const py::object &cell);
  GradExecutorPtr grad_executor();
  ForwardExecutorPtr forward_executor();

  void set_grad_flag(bool flag);
  void NewGraph(const py::object &cell, const py::args &args);
  void EndGraph(const py::object &cell, const py::object &out, const py::args &args);
  void GradNet(const GradOperationPtr &grad, const py::object &cell, const py::object &weights, const py::args &args);
  py::object CheckGraph(const py::object &cell, const py::args &args);
  py::object CheckAlreadyRun(const py::object &cell, const py::args &args);
  py::object Run(const py::object &cell, const py::tuple &args, const py::object &phase);

  // Used by graph clean
  bool GetIsDynamicCell();
  bool need_replace_forward() { return grad_executor()->need_replace_forward(); }
  // Cell destruct will call
  void ClearCell(const std::string &flag = "");
  void ClearGrad(const py::object &cell, const py::args &args);
  // Abnormal existed
  void ClearRes();
  // Sync stream
  void Sync();

 private:
  PynativeExecutor() = default;

  static std::shared_ptr<PynativeExecutor> executor_;
  static std::mutex instance_lock_;
  static ForwardExecutorPtr forward_executor_;
  static GradExecutorPtr grad_executor_;
  // The pointer of top python Cell object, which is always the network(inherit class Cell) ran in python test script,
  // such as Resnet50(Cell),LeNet(Cell).This pointer is used to distinguish temporary primitives from global
  // primitives to control memory release. Global primitives are always created in top cell's '__init__' function and
  // temporary primitives are always created in other place.Temporary primitives will be released after executing top
  // cell's 'construct' function but global primitives will not.
  PyObject *py_top_cell_{nullptr};
};

using PynativeExecutorPtr = std::shared_ptr<PynativeExecutor>;
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_EXECUTE_H_
