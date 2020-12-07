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

py::object RunOpInVM(const OpExecInfoPtr &op_exec_info, PynativeStatusCode *status);

py::tuple RunOp(const py::args &args);

void ClearPyNativeSession();

struct GraphInfo {
  std::unordered_set<std::string> params;  // hold input parameters and cell weigths
  std::unordered_map<std::string, std::pair<AnfNodePtr, std::vector<int64_t>>> node_map;
  AnfNodePtr output;
  std::vector<std::string> objects;
};

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

  bool grad_flag() { return grad_flag_; }
  void set_grad_flag(bool flag) { grad_flag_ = flag; }

  py::tuple RunOpInner(const OpExecInfoPtr &op_exec_info);
  OpExecInfoPtr GenerateOpExecInfo(const py::args &args);
  void NewGraph(const py::object &cell, const py::args &args);
  py::object Run(const py::tuple &args, const py::object &phase);
  py::object CheckGraph(const py::object &cell, const py::args &args);
  void EndGraph(const py::object &cell, const py::object &out, const py::args &args);
  void GradNet(const GradOperationPtr &grad, const py::object &cell, const py::object &weights, const py::args &args);

  // Call by python
  void Clear(const std::string &flag = "");
  // Abnormal existed
  void Clean();
  // Destrcut call
  void ClearRes();
  // Sync stream
  void Sync();

 private:
  PynativeExecutor() = default;

  // check cell struct
  bool IsDynamicCell(const py::object &cell);
  std::string GetCellInfo(const py::object &cell);
  void ParseInputArgs(const std::shared_ptr<parse::ParseAst> &ast, const py::object &fn_node);
  bool ParseBodyContext(const std::shared_ptr<parse::ParseAst> &ast, const py::object &fn_node);
  bool ParseIfWhileExprNode(const std::shared_ptr<parse::ParseAst> &ast, const py::object &node);
  bool ParseAssignExprNode(const std::shared_ptr<parse::ParseAst> &ast, const py::object &node);
  bool ParseForExprNode(const std::shared_ptr<parse::ParseAst> &ast, const py::object &node);
  std::string ParseNodeName(const std::shared_ptr<parse::ParseAst> &ast, const py::object &node,
                            parse::AstMainType type);

  py::object DoParamMixPrecisionCast(bool *is_cast, const py::object obj, const std::string &op_name, size_t index);
  py::object DoParamMixPrecisionCastTuple(bool *is_cast, const py::tuple tuple, const std::string &op_name,
                                          size_t index);
  py::object DoAutoCast(const py::object &arg, const TypeId &type_id, const std::string &op_name, size_t index);
  void DoSignatrueCast(const PrimitivePyPtr &prim, const std::map<SignatureEnumDType, TypeId> &dst_type,
                       const std::vector<SignatureEnumDType> &dtypes, const OpExecInfoPtr &op_exec_info);
  // run op
  AnfNodePtr GetInput(const py::object &obj, bool op_mask);
  MsBackendPolicy InitEnv(const OpExecInfoPtr &op_exec_info);
  py::tuple RunOpWithInitBackendPolicy(const OpExecInfoPtr &op_exec_info);
  void RunParameterAutoMixPrecisionCast(const OpExecInfoPtr &op_exec_info);
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

  // replace for grad graph
  ValuePtr CleanTupleAddr(const ValueTuplePtr &tuple);
  void GenTupleMap(const ValueTuplePtr &tuple, std::map<std::string, tensor::TensorPtr> *t_map);
  void SaveAllResult(const OpExecInfoPtr &op_exec_info, const AnfNodePtr &node, const py::object &out_real);
  // Update the abstract and device address info of value node and tensors in bprop graph
  void UpdateAbstractAndDeviceAddress(const OpExecInfoPtr &op_exec_info, const py::object &out_real);
  void SaveTensorsInValueNode(const ResourcePtr &resource);

  // construct grad graph
  void PushCurrentGraphToStack();
  void PopGraphStack();
  void NewGraphInner(const py::object &cell, const py::args &args);
  void MakeNewTopGraph(const string &cell_id, const py::args &args, const FuncGraphPtr &g);
  void EndGraphInner(const py::object &cell, const py::object &out, const py::args &args);
  void EndGraphByOutId(const std::string &out_id, const py::object &cell, const py::object &out, const py::args &args);
  FuncGraphPtr MakeGradGraph(const py::object &cell, const py::args &args);
  void GradNetInner(const GradOperationPtr &grad, const py::object &cell, const py::object &weights,
                    const py::args &args);
  std::string GetCellId(const py::object &obj, const py::args &args);
  std::string CheckCellChanged(const GradOperationPtr &grad, const py::object &cell, const py::object &weights,
                               const py::args &args, std::pair<bool, bool> *sens_weights_changed);
  void SetGradGraphParams(size_t size, const std::string &cell_id, const std::pair<bool, bool> &sens_weights_changed);
  void GradGraph(FuncGraphPtr g, const GradOperationPtr &grad_op, const std::vector<AnfNodePtr> &weights,
                 size_t arg_size);
  std::vector<AnfNodePtr> GetWeightsArgs(const py::object &weights);
  abstract::AbstractBasePtrList GetArgsSpec(const py::args &args);

  // hold graph(forward and grad) info
  void SetPyObjInGraphInfoMap(FuncGraphPtr g, const std::string obj) { graph_info_map_[g].objects.push_back(obj); }
  void SetTupleArgsToGraphInfoMap(const FuncGraphPtr &g, const py::object &args, const AnfNodePtr &node,
                                  bool is_param = false);
  void SetNodeMapInGraphInfoMap(FuncGraphPtr g, const std::string id, AnfNodePtr node, int64_t index = -1) {
    graph_info_map_[g].node_map[id] = std::make_pair(node, std::vector<int64_t>{index});
  }
  void SetNodeMapInGraphInfoMap(FuncGraphPtr g, const std::string id, AnfNodePtr node, std::vector<int64_t> index) {
    graph_info_map_[g].node_map[id] = std::make_pair(node, index);
  }
  void SetTupleItemArgsToGraphInfoMap(const FuncGraphPtr &g, const py::object &id, const AnfNodePtr &node,
                                      const std::vector<int64_t> &index_sequence, bool is_param = false);

  static std::shared_ptr<PynativeExecutor> executor_;
  static std::mutex instance_lock_;
  static int64_t graph_id_;
  bool grad_flag_{false};
  bool dynamic_cell_{false};
  bool grad_is_running{false};

  // Used for construct grad graph
  FuncGraphPtr top_g_{nullptr};
  FuncGraphPtr curr_g_{nullptr};
  FuncGraphPtr df_builder_{nullptr};
  ResourcePtr resource_{nullptr};
  // Records forwrad graph, the bottom is top graph
  std::stack<FuncGraphPtr> graph_stack_;
  std::unordered_set<std::string> top_graph_cells_;

  // record all info of a graph
  std::unordered_map<FuncGraphPtr, GraphInfo> graph_info_map_;
  std::unordered_set<std::string> cell_input_args_;
  std::unordered_map<std::string, bool> cell_dynamic_map_;
  std::unordered_map<std::string, ResourcePtr> cell_resource_map_;
  std::unordered_map<std::string, std::pair<FuncGraphPtr, bool>> cell_graph_map_;
  // key: cell_id, value: (send_id, weigths_id), cache for sens and weight change
  std::unordered_map<std::string, std::pair<std::string, std::string>> cell_sw_map_;
  // key: cell_id, value: (forward graph, grad graph)
  std::unordered_map<std::string, std::pair<FuncGraphPtr, FuncGraphPtr>> df_builder_map_;

  // used for runop and replace forward result of grad graph
  std::unordered_map<std::string, size_t> op_index_map_;
  std::unordered_map<std::string, std::string> obj_to_forward_id_;
  std::unordered_map<std::string, std::vector<std::string>> op_index_with_tensor_id_;
  std::unordered_map<std::string, std::vector<tensor::TensorPtr>> tensor_id_with_tensor_;
  std::unordered_map<std::string, abstract::AbstractBasePtr> node_abs_map_;
  std::unordered_map<std::string, AbstractListMap> prim_abs_list_;
  const inline static std::string kOpsFunctionModelName = "mindspore.ops.functional";
  const inline static std::string kMSDtypeModelName = "mindspore.common.dtype";
};

using PynativeExecutorPtr = std::shared_ptr<PynativeExecutor>;
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_EXECUTE_H_
