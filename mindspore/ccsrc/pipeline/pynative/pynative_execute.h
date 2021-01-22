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

#include <vector>
#include <utility>
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

class CellInfo {
 public:
  CellInfo() = default;
  CellInfo(bool custom_bprop, bool has_dynamic, FuncGraphPtr foward_graph, std::string cellid, std::string bprop_id)
      : is_custom_bprop(custom_bprop),
        is_dynamic(has_dynamic),
        fg(std::move(foward_graph)),
        cell_id(std::move(cellid)),
        bprop_cell_id(std::move(bprop_id)) {}

  bool is_grad{false};          // Derivative is calculated
  bool is_custom_bprop{false};  // Custom bprop
  bool is_dynamic{false};       // Set by has_dynamic_cell
  bool is_real_dynamic{false};  // Set by ops order
  size_t call_times{0};
  FuncGraphPtr fg{nullptr};  // Forward graph
  std::string cell_id;
  std::string bprop_cell_id;
  std::vector<std::string> cell_ops_info;  // All ops info
};

class TopCellInfo {
 public:
  TopCellInfo() = default;
  TopCellInfo(bool topest, ResourcePtr r, FuncGraphPtr df, std::string cellid)
      : is_topest(topest), resource(std::move(r)), df_builder(std::move(df)), cell_id(std::move(cellid)) {}

  bool is_topest{false};
  bool do_vm_compiled{false};
  bool forward_already_run{false};
  ResourcePtr resource{nullptr};
  FuncGraphPtr df_builder{nullptr};
  FuncGraphPtr bg{nullptr};  // Backward graph
  std::string cell_id;
  std::string sens_id;
  std::string weights_id;
};

using GraphInfoPtr = std::shared_ptr<GraphInfo>;
using CellInfoPtr = std::shared_ptr<CellInfo>;
using TopCellInfoPtr = std::shared_ptr<TopCellInfo>;

class PynativeExecutor : public std::enable_shared_from_this<PynativeExecutor> {
 public:
  static std::shared_ptr<PynativeExecutor> GetInstance() {
    std::lock_guard<std::mutex> i_lock(instance_lock_);
    if (executor_ == nullptr) {
      executor_ = std::shared_ptr<PynativeExecutor>(new (std::nothrow) PynativeExecutor());
    }
    return executor_;
  }
  ~PynativeExecutor();
  PynativeExecutor(const PynativeExecutor &) = delete;
  PynativeExecutor &operator=(const PynativeExecutor &) = delete;

  bool need_replace_forward() const { return need_replace_forward_; }
  bool grad_flag() const { return grad_flag_; }
  void set_grad_flag(bool flag) { grad_flag_ = flag; }
  void EnterConstruct(const py::object &cell);
  void LeaveConstruct(const py::object &cell);

  py::object RunOpInner(const OpExecInfoPtr &op_exec_info);
  OpExecInfoPtr GenerateOpExecInfo(const py::args &args);
  void NewGraph(const py::object &cell, const py::args &args);
  py::object Run(const py::object &cell, const py::tuple &args, const py::object &phase);
  py::object CheckGraph(const py::object &cell, const py::args &args);
  py::object CheckAlreadyRun(const py::object &cell, const py::args &args);
  void EndGraph(const py::object &cell, const py::object &out, const py::args &args);
  void GradNet(const GradOperationPtr &grad, const py::object &cell, const py::object &weights, const py::args &args);

  // Get info
  bool GetIsDynamicCell() { return CheckRealDynamicCell(top_cell_id_); }
  // Call by python
  void Clear(const std::string &flag = "");
  void Clean();
  // Abnormal existed
  void ClearRes();
  // Sync stream
  void Sync();

 private:
  PynativeExecutor() = default;

  template <typename T>
  void MapClear(T *map, const std::string &cell_id) {
    for (auto it = map->begin(); it != map->end();) {
      if (it->first.find(cell_id) != std::string::npos) {
        it = map->erase(it);
      } else {
        it++;
      }
    }
  }

  template <typename T>
  void VectorClear(T *vec, const std::string &cell_id) {
    for (auto it = vec->begin(); it != vec->end();) {
      if ((*it)->cell_id.find(cell_id) != std::string::npos) {
        it = vec->erase(it);
      } else {
        it++;
      }
    }
  }

  // Check cell struct
  bool IsDynamicCell(const py::object &cell);
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
  py::object DoParamMixPrecisionCast(bool *is_cast, const py::object obj, const std::string &op_name, size_t index);
  py::object DoParamMixPrecisionCastTuple(bool *is_cast, const py::tuple tuple, const std::string &op_name,
                                          size_t index);
  py::object DoAutoCast(const py::object &arg, const TypeId &type_id, const std::string &op_name, size_t index);
  void DoSignatrueCast(const PrimitivePyPtr &prim, const std::map<SignatureEnumDType, TypeId> &dst_type,
                       const std::vector<SignatureEnumDType> &dtypes, const OpExecInfoPtr &op_exec_info);
  // Run op
  AnfNodePtr GetInput(const py::object &obj, bool op_mask);
  MsBackendPolicy InitEnv(const OpExecInfoPtr &op_exec_info);
  py::tuple RunOpWithInitBackendPolicy(const OpExecInfoPtr &op_exec_info);
  void RunParameterAutoMixPrecisionCast(const OpExecInfoPtr &op_exec_info);
  py::object RunOpInVM(const OpExecInfoPtr &op_exec_info, PynativeStatusCode *status);
  py::object RunOpInMs(const OpExecInfoPtr &op_exec_info, PynativeStatusCode *status);
  py::object RunOpWithBackendPolicy(MsBackendPolicy backend_policy, const OpExecInfoPtr &op_exec_info,
                                    PynativeStatusCode *const status);
  AnfNodePtr GetObjNode(const py::object &obj, const std::string &obj_id);
  AnfNodePtr MakeValueNode(const py::object &obj, const std::string &obj_id);
  AnfNodePtr MakeCNode(const OpExecInfoPtr &op_exec_info, std::vector<bool> *op_masks,
                       abstract::AbstractBasePtrList *args_spec_list);
  void GetOpOutputAbstract(const OpExecInfoPtr &op_exec_info, const abstract::AbstractBasePtrList &args_spec_list,
                           bool *is_find);
  void SaveOutputNodeMap(const std::string &obj_id, const py::object &out_real, const AnfNodePtr &cnode);

  // Replace for grad graph
  ValuePtr CleanTupleAddr(const ValueTuplePtr &tuple);
  void GenTupleMap(const ValueTuplePtr &tuple, std::map<std::string, tensor::TensorPtr> *t_map);
  void SaveAllResult(const OpExecInfoPtr &op_exec_info, const AnfNodePtr &node, const py::object &out_real);
  // Update the abstract and device address info of value node and tensors in bprop graph
  void UpdateAbstractAndDeviceAddress(const OpExecInfoPtr &op_exec_info, const py::object &out_real);
  void SaveTensorsInValueNode(const ResourcePtr &resource);
  void SaveAllValueNodeTensors(const FuncGraphPtr &graph);
  void CleanPreMemoryInValueNode();

  // Construct grad graph
  void PushCurrentGraphToStack();
  void PopGraphStack();
  void PushCurrentCellOpInfoToStack();
  void PopCurrentCellOpInfoFromStack();
  FuncGraphPtr GetDfbuilder(const std::string &cell_id = "");
  ResourcePtr GetResource(const std::string &cell_id = "");
  void AddNestedGradOrder() { ++grad_order_; }
  void SubNestedGradOrder();
  bool IsNestedGrad() const;
  bool IsTopGraph(const std::string &cell_id);
  bool IsTopestGraph(const std::string &cell_id);
  bool IsBpropGraph(const std::string &cell_id);
  bool IsFirstGradStep(const std::string &cell_id);
  bool grad_running() const { return grad_is_running_; }
  void set_grad_runing(bool grad_runing) { grad_is_running_ = grad_runing; }
  void set_need_replace_forward(bool need_replace_forward) { need_replace_forward_ = need_replace_forward; }
  bool need_construct_graph() { return !graph_stack_.empty() && grad_flag_; }
  bool CheckCellGraph(const std::string &cell_id, bool is_grad = false);
  bool CheckDynamicCell(const std::string &cell_id);
  bool CheckRealDynamicCell(const std::string &cell_id);
  void UpdateCellGraph(const py::object &cell, const FuncGraphPtr &g, const std::string &cell_id,
                       bool need_cloned = false, bool is_grad = false);
  void ClearCnodeRes(const AnfNodePtr &node, std::unordered_set<AnfNodePtr> *node_set);
  void UpdateCellDynamic(const std::string &cell_id);
  bool CheckCellChanged(const std::string &cell_id);
  void UpdateTopCellInfo(const std::string &cell_id, bool vm_compiled);
  void ClearResidualRes(const std::string &cell_id);
  void DumpGraphIR(const std::string &filename, const FuncGraphPtr &graph);
  void NewGraphInner(const py::object &cell, const py::args &args);
  void MakeNewTopGraph(const string &cell_id, const py::args &args);
  TopCellInfoPtr GetTopCell(const string &cell_id, bool find_nearest = false);
  void EndGraphInner(const py::object &cell, const py::object &out, const py::args &args);
  void EndGraphByOutId(const py::object &cell, const std::string &cell_id, const py::object &out,
                       const std::string &out_id, const py::args &args);
  bool EndBpropGraph(const string &cell_id);
  FuncGraphPtr MakeGradGraph(const py::object &cell, const FuncGraphPtr &g, const ResourcePtr &r,
                             const std::string &cell_id, const py::args &args);
  std::string GetGradCellId(bool has_sens, const py::object &cell, const py::args &args, py::object *forward_args,
                            py::object *sens = nullptr);
  void ClearDynamicTopRes(const std::string &cell_id);
  void GradNetInner(const GradOperationPtr &grad, const py::object &cell, const py::object &weights,
                    const py::args &args);
  std::string GetCellId(const py::object &obj, const py::args &args);
  std::string GetTensorCellId(const std::string &cell_id);
  bool CheckGradParamsChanged(const std::string &cell_id, const py::object &weights, const py::object &sens);
  void SetGradGraphParams(const FuncGraphPtr &df_builder, const ResourcePtr &resource, size_t size);
  void GradGraph(const FuncGraphPtr &g, const GradOperationPtr &grad_op, const std::vector<AnfNodePtr> &weights,
                 size_t arg_size, const std::string &cell_id);
  std::vector<AnfNodePtr> GetWeightsArgs(const py::object &weights, const FuncGraphPtr &df_builder);
  abstract::AbstractBasePtrList GetArgsSpec(const py::args &args, const FuncGraphPtr &df_builder);
  void ClearUselessRes(const FuncGraphPtr &df_builder, const py::object &cell, const std::string &cell_id);
  void ReplaceGraphParams(const FuncGraphPtr &df_builder, const FuncGraphPtr &forward_graph,
                          const std::string &cell_id);
  void SetNestedTopGraph(const py::object &cell, const py::args &args, const std::string &cell_id);
  void MakeNestedCnode(const std::string &cell_id, const py::args &args, const ResourcePtr &resource,
                       const py::object &out, bool has_sens);
  void RecoverGraphParams(const FuncGraphPtr &newfg, const std::string &cell_id, std::vector<AnfNodePtr> *inputs);
  bool MakeBpropNestedCnode(const py::object &cell, const py::object &out, const std::string &cell_id);

  // Hold graph(forward and grad) info
  std::string GetCellOpInfo();
  void ReplaceCellOpInfoByCellId(const std::string &cell_id);
  void SetPyObjInGraphInfoMap(const FuncGraphPtr &g, const std::string &obj) {
    graph_info_map_[g]->objects.push_back(obj);
  }
  void SetTupleArgsToGraphInfoMap(const FuncGraphPtr &g, const py::object &args, const AnfNodePtr &node,
                                  bool is_param = false);
  void SetParamNodeMapInGraphInfoMap(const FuncGraphPtr &g, const std::string &id, const ParameterPtr &param) {
    graph_info_map_[g]->params[id] = param;
  }
  void SetNodeMapInGraphInfoMap(const FuncGraphPtr &g, const std::string &id, const AnfNodePtr &node,
                                int64_t index = -1) {
    graph_info_map_[g]->node_map[id] = std::make_pair(node, std::vector<int64_t>{index});
  }
  void SetNodeMapInGraphInfoMap(const FuncGraphPtr &g, const std::string &id, const AnfNodePtr &node,
                                const std::vector<int64_t> &index) {
    graph_info_map_[g]->node_map[id] = std::make_pair(node, index);
  }
  void SetTupleItemArgsToGraphInfoMap(const FuncGraphPtr &g, const py::object &id, const AnfNodePtr &node,
                                      const std::vector<int64_t> &index_sequence, bool is_param = false);

  static std::shared_ptr<PynativeExecutor> executor_;
  static std::mutex instance_lock_;
  static int64_t graph_id_;
  size_t grad_order_{0};
  std::string top_cell_id_;
  bool grad_flag_{false};
  bool in_grad_process_{false};
  bool has_dynamic_cell_{false};
  bool grad_is_running_{false};
  bool need_replace_forward_{true};
  // The pointer of top python Cell object, which is always the network(inherit class Cell) ran in python test script,
  // such as Resnet50(Cell),LeNet(Cell).This pointer is used to distinguish temporary primitives from global
  // primitives to control memory release. Global primitives are always created in top cell's '__init__' function and
  // temporary primitives are always created in other place.Temporary primitives will be released after executing top
  // cell's 'construct' function but global primitives will not.
  PyObject *top_cell_{nullptr};

  // Used for construct grad graph
  FuncGraphPtr curr_g_{nullptr};
  // Records forwrad graph, the bottom is top graph
  std::stack<FuncGraphPtr> graph_stack_;
  // Records op info of every cell, the bottom is op info of top cell
  std::stack<std::string> cell_op_info_stack_;

  // Use vector for keep order
  std::vector<CellInfoPtr> cell_graph_list_;
  std::vector<TopCellInfoPtr> top_cell_list_;
  std::unordered_set<std::string> cell_input_args_;
  // Record all info for all cells
  OrderedMap<FuncGraphPtr, GraphInfoPtr> graph_info_map_;
  std::unordered_map<FuncGraphPtr, std::vector<std::pair<ParameterPtr, ParameterPtr>>> replace_weights_map_;

  // Used for runop and replace forward result of grad graph
  std::unordered_map<std::string, size_t> op_index_map_;
  std::unordered_map<std::string, std::string> obj_to_forward_id_;
  std::unordered_map<std::string, OpIndexWithTensorId> cell_op_index_with_tensor_id_;
  std::unordered_map<std::string, TensorIdWithTensor> cell_tensor_id_with_tensor_;
  std::unordered_map<std::string, abstract::AbstractBasePtr> node_abs_map_;
  std::unordered_map<std::string, AbstractListMap> prim_abs_list_;
  std::unordered_set<tensor::TensorPtr> all_value_node_tensors_;
};

using PynativeExecutorPtr = std::shared_ptr<PynativeExecutor>;
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_EXECUTE_H_
