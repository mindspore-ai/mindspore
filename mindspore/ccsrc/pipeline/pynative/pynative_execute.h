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
#include "frontend/optimizer/ad/kpynative.h"
#include "frontend/operator/composite/composite.h"

namespace mindspore::pynative {
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
  GraphInfo() = default;
  explicit GraphInfo(std::string id) : cell_id(std::move((id))) {}
};
using GraphInfoPtr = std::shared_ptr<GraphInfo>;

class TopCellInfo {
 public:
  TopCellInfo() = default;
  ~TopCellInfo() = default;
  TopCellInfo(bool topest, ResourcePtr r, FuncGraphPtr df, std::string cellid)
      : is_topest_(topest), resource_(std::move(r)), df_builder_(std::move(df)), cell_id_(std::move(cellid)) {}

  bool is_init_kpynative() const { return is_init_kpynative_; }
  void set_init_kpynative(bool init) { is_init_kpynative_ = init; }
  bool is_topest() const { return is_topest_; }
  bool vm_compiled() const { return vm_compiled_; }
  void set_vm_compiled(bool vm_compiled) { vm_compiled_ = vm_compiled; }
  bool forward_already_run() const { return forward_already_run_; }
  void set_forward_already_run(bool set_forward_already_run) { forward_already_run_ = set_forward_already_run; }
  ResourcePtr resource() { return resource_; }
  FuncGraphPtr df_builder() { return df_builder_; }
  std::string &cell_id() { return cell_id_; }
  std::string &input_args_id() { return input_args_id_; }
  void set_input_args_id(std::string input_args_id) { input_args_id_ = std::move(input_args_id); }
  std::string &bprop_cell_id() { return bprop_cell_id_; }
  void set_bprop_cell_id(std::string &bprop_cell_id) { bprop_cell_id_ = bprop_cell_id; }
  OrderedMap<FuncGraphPtr, GraphInfoPtr> &graph_info_map() { return graph_info_map_; }
  void set_graph_info_map(const OrderedMap<FuncGraphPtr, GraphInfoPtr> &graph_info_map) {
    graph_info_map_ = graph_info_map;
  }
  ad::KPynativeCellPtr k_pynative_cell_ptr() const { return k_pynative_cell_ptr_; }
  void set_k_pynative_cell_ptr(const ad::KPynativeCellPtr &k_pynative_cell_ptr) {
    k_pynative_cell_ptr_ = k_pynative_cell_ptr;
  }
  void clear() { graph_info_map_.clear(); }

 private:
  bool is_topest_{false};
  bool vm_compiled_{false};
  bool is_init_kpynative_{false};
  bool forward_already_run_{false};
  ResourcePtr resource_{nullptr};
  FuncGraphPtr df_builder_{nullptr};
  ad::KPynativeCellPtr k_pynative_cell_ptr_{nullptr};
  std::string cell_id_;
  std::string bprop_cell_id_;
  std::string input_args_id_;
  OrderedMap<FuncGraphPtr, GraphInfoPtr> graph_info_map_;
};
using TopCellInfoPtr = std::shared_ptr<TopCellInfo>;

class ForwardExecutor;
using ForwardExecutorPtr = std::shared_ptr<ForwardExecutor>;
using ForwardExecutorWeakPtr = std::weak_ptr<ForwardExecutor>;

class GradExecutor;
using GradExecutorPtr = std::shared_ptr<GradExecutor>;
using GradExecutorWeakPtr = std::weak_ptr<GradExecutor>;

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
  TopCellInfoPtr top_cell_direct() const { return top_cell_; }
  bool TopCellIsDynamic();
  void set_top_cell(TopCellInfoPtr top_cell) { top_cell_ = std::move(top_cell); }
  bool grad_flag() const { return grad_flag_; }
  void set_grad_flag(bool flag) { grad_flag_ = flag; }
  bool in_grad_process() const { return in_grad_process_; }
  AnfNodePtr GetInput(const py::object &obj, bool op_mask);
  std::string GetCellId(const py::object &obj, const py::args &args);
  void SaveOutputNodeMap(const std::string &obj_id, const py::object &out_real, const AnfNodePtr &cnode);
  void DoOpGrad(const OpExecInfoPtr &op_exec_info, const AnfNodePtr &node, const py::object &op_out);
  py::object CheckGraph(const py::object &cell, const py::args &args);
  void RunGradGraph(py::object *ret, const py::object &cell, const py::tuple &args, const py::object &phase);
  bool need_construct_graph() const { return !cell_stack_.empty() && grad_flag_; }
  std::stack<std::string> &cell_stack() { return cell_stack_; }
  std::vector<TopCellInfoPtr> &top_cell_list() { return top_cell_list_; }
  void ClearGrad(const py::object &cell, const py::args &args);
  void ClearRes();
  void ClearCellRes(const std::string &cell_id = "");

  void InitGradForPrimBpOpt();

 private:
  ForwardExecutorPtr forward() const;
  bool grad_running() const { return grad_is_running_; }
  void set_grad_runing(bool grad_runing) { grad_is_running_ = grad_runing; }

  // Higher derivative
  bool IsNestedGrad() const;
  size_t cell_nums() const { return cell_nums_; }
  void set_cell_nums(size_t cell_nums) { cell_nums_ = cell_nums; }
  void AddNestedGradOrder() { ++grad_order_; }
  void SubNestedGradOrder();
  void MakeNestedCnode(const std::string &cell_id, const py::args &args, const ResourcePtr &resource,
                       const py::object &out, bool has_sens);
  bool MakeBpropNestedCnode(const py::object &cell, const py::object &out, const std::string &cell_id);
  void PushCellStack(const std::string &cell_id);
  void PopCellStack();
  void PushGraphStack();
  void PopGraphStack();

  FuncGraphPtr GetDfbuilder(const std::string &cell_id = "");
  ResourcePtr GetResource(const std::string &cell_id = "");
  bool IsCellObjIdEq(const std::string &l_cell_id, const std::string &r_cell_id);
  bool IsTopGraph(const std::string &cell_id);
  bool IsBpropGraph(const std::string &cell_id);
  void UpdateBpropCellGraph(const py::object &cell, const std::string &cell_id);
  void UpdateTopCellInfo(const std::string &cell_id, bool vm_compiled);
  void DumpGraphIR(const std::string &filename, const FuncGraphPtr &graph);
  void NewGraphInner(py::object *ret, const py::object &cell, const py::args &args);
  void MakeNewTopGraph(const string &cell_id, const py::args &args, bool is_topest);
  void EndGraphInner(py::object *ret, const py::object &cell, const py::object &out, const py::args &args);
  std::string GetGradCellId(bool has_sens, const py::object &cell, const py::args &args);
  void GradNetInner(py::object *ret, const GradOperationPtr &grad, const py::object &cell, const py::object &weights,
                    const py::args &args);
  FuncGraphPtr GetBpropGraph(const GradOperationPtr &grad, const std::vector<AnfNodePtr> &weights, size_t arg_size);
  std::vector<AnfNodePtr> GetWeightsArgs(const py::object &weights, const FuncGraphPtr &df_builder);
  abstract::AbstractBasePtrList GetArgsSpec(const py::args &args, const FuncGraphPtr &bprop_graph);
  void SetTupleItemArgsToGraphInfoMap(const FuncGraphPtr &g, const py::object &id, const AnfNodePtr &node,
                                      const std::vector<int64_t> &index_sequence, bool is_param = false);
  AnfNodePtr GetObjNode(const py::object &obj, const std::string &obj_id);
  AnfNodePtr MakeValueNode(const py::object &obj, const std::string &obj_id);

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
  size_t cell_nums_{0};
  bool grad_flag_{false};
  bool in_bprop_process_{false};
  bool in_grad_process_{false};
  bool grad_is_running_{false};

  FuncGraphPtr curr_g_{nullptr};
  // For clear pre top res
  TopCellInfoPtr top_cell_{nullptr};
  // Records forwrad cell, the bottom is top cell
  std::stack<std::string> cell_stack_;
  // For high grad order
  std::stack<FuncGraphPtr> high_order_stack_;
  // Use vector for keep order
  std::vector<TopCellInfoPtr> top_cell_list_;
  // Use vector for keep order
  ForwardExecutorWeakPtr forward_executor_;
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
  void ClearRes();

  AnfNodePtr MakeCNodeForPrimBpOpt(const OpExecInfoPtr &op_exec_info, std::vector<bool> *op_masks,
                                   abstract::AbstractBasePtrList *args_spec_list);

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
  void GetArgsSpec(const OpExecInfoPtr &op_exec_info, std::vector<int64_t> *op_masks, std::vector<AnfNodePtr> *inputs,
                   abstract::AbstractBasePtrList *args_spec_list);
  abstract::AbstractBasePtr CheckConstValue(const PrimitivePyPtr &prim, const py::object &obj,
                                            const abstract::AbstractBasePtr &abs, const std::string &id, size_t index);
  void GetOpOutputAbstract(const OpExecInfoPtr &op_exec_info, const abstract::AbstractBasePtrList &args_spec_list,
                           bool *is_find);
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
  // Used to cache cast struct
  std::unordered_map<std::string, OpExecInfoPtr> cast_struct_map_;
};

class PynativeExecutor : public std::enable_shared_from_this<PynativeExecutor> {
 public:
  static std::shared_ptr<PynativeExecutor> GetInstance() {
    std::lock_guard<std::mutex> i_lock(instance_lock_);
    if (executor_ == nullptr) {
      executor_ = std::shared_ptr<PynativeExecutor>(new (std::nothrow) PynativeExecutor());
      forward_executor_ = std::make_shared<ForwardExecutor>();
      grad_executor_ = std::make_shared<GradExecutor>(forward_executor_);
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
  // Cell destruct will call
  void ClearCell(const std::string &cell_id);
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
}  // namespace mindspore::pynative

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_EXECUTE_H_
