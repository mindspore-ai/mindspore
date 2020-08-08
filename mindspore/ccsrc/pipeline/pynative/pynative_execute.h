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
#include <mutex>
#include <stack>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

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
  std::unordered_map<std::string, ValuePtr> attrs;
};

using AbstractListMap = std::unordered_map<abstract::AbstractBasePtrList, PrimAbsInfo,
                                           abstract::AbstractBasePtrListHasher, abstract::AbstractBasePtrListEqual>;

py::object RunOpInVM(const OpExecInfoPtr &op_exec_info, PynativeStatusCode *status);

py::tuple RunOp(const py::args &args);

void ConvertInputs(const PrimitivePyPtr &prim, const py::list &py_args, py::tuple *const out_args,
                   py::list *const out_args_list);

void ClearPyNativeSession();

struct GraphInfo {
  std::unordered_map<std::string, AnfNodePtr> param_map;
  std::unordered_map<std::string, std::pair<AnfNodePtr, std::vector<int>>> obj_node_map;
  AnfNodePtr output;
  std::vector<std::string> objects;
};

class PynativeExecutor : public std::enable_shared_from_this<PynativeExecutor> {
 public:
  static std::shared_ptr<PynativeExecutor> GetInstance() {
    std::lock_guard<std::mutex> i_lock(instance_lock_);
    if (executor_ == nullptr) {
      executor_ = std::shared_ptr<PynativeExecutor>(new (std::nothrow) PynativeExecutor());
      resource_ = std::make_shared<pipeline::Resource>();
    }
    return executor_;
  }
  void NewGraph(const py::object &cell, const py::args &args);
  void NewGraphInner(const py::object &cell, const py::args &args);
  void EndGraph(const py::object &cell, const py::object &out, const py::args &args);
  void EndGraphInner(const py::object &cell, const py::object &out, const py::args &args);
  void EndGraphByOutId(const std::string &out_id, const py::object &cell, const py::object &out, const py::args &args);
  std::vector<AnfNodePtr> GetWeightsArgs(const py::object &weights);
  abstract::AbstractBasePtrList GetArgsSpec(const py::args &args);
  void GradNet(const GradOperationPtr &grad, const py::object &cell, const py::object &weights, const py::args &args);
  void GradNetInner(const GradOperationPtr &grad, const py::object &cell, const py::object &weights,
                    const py::args &args);
  void Clear(const std::string &flag = "");
  void Clean();
  void ClearRes();
  bool grad_flag() { return grad_flag_; }
  void set_grad_flag(bool flag) { grad_flag_ = flag; }
  AnfNodePtr GetInput(const py::object &obj, bool op_mask);
  AnfNodePtr GetObjNode(const py::object &obj);
  std::string GetCellId(const py::object &obj, const py::args &args);
  FuncGraphPtr curr_g() { return curr_g_; }
  void set_pyobj(FuncGraphPtr g, const std::string obj) { graph_info_map_[g].objects.push_back(obj); }
  void set_obj_node_map(FuncGraphPtr g, const std::string obj, AnfNodePtr node) {
    graph_info_map_[g].obj_node_map[obj] = std::make_pair(node, std::vector<int>{-1});
  }
  void set_obj_node_map(FuncGraphPtr g, const std::string obj, AnfNodePtr node, int index) {
    graph_info_map_[g].obj_node_map[obj] = std::make_pair(node, std::vector<int>{index});
  }
  void set_obj_node_map(FuncGraphPtr g, const std::string obj, AnfNodePtr node, std::vector<int> index) {
    graph_info_map_[g].obj_node_map[obj] = std::make_pair(node, index);
  }
  AnfNodePtr MakeCNode(const OpExecInfoPtr &op_exec_info, std::vector<bool> *op_masks,
                       abstract::AbstractBasePtrList *args_spec_list);
  void MakeCNode(const OpExecInfoPtr &op_exec_info, const py::object &out, const AnfNodePtr &cnode);
  ValuePtr GetForwardValue(const OpExecInfoPtr &op_exec_info);
  void SaveOpForwardValue(const OpExecInfoPtr &op_exec_info, const ValuePtr &value);
  void SaveForwardResult(const CNodePtr &cnode, const py::object &out);
  void SaveAllResult(const OpExecInfoPtr &op_exec_info, const CNodePtr &cnode, const py::tuple &out);

  py::object Run(const py::tuple &args, const py::object &phase);

  void Pushp();
  void Popp();
  FuncGraphPtr GradGraph(FuncGraphPtr g, const GradOperationPtr &grad_op, const std::vector<AnfNodePtr> &weights,
                         size_t arg_size);
  void SetTupleOutput(const py::object &obj, const AnfNodePtr &cnode, std::vector<int> idx);
  AnfNodePtr MakeValueNode(const py::object &obj, const std::string &obj_id);
  py::tuple RunOpInner(const py::args &args);
  py::tuple RunOpInner(const OpExecInfoPtr &op_exec_info);

  ~PynativeExecutor();

 private:
  PynativeExecutor();
  static std::shared_ptr<PynativeExecutor> executor_;
  static std::mutex instance_lock_;
  static ResourcePtr resource_;
  bool grad_flag_;
  std::unordered_map<std::string, FuncGraphPtr> graph_map_;
  std::unordered_map<std::string, FuncGraphPtr> cell_graph_map_;
  std::unordered_map<std::string, ResourcePtr> cell_resource_map_;
  std::unordered_map<FuncGraphPtr, GraphInfo> graph_info_map_;
  std::unordered_map<std::string, ValuePtr> op_forward_map_;
  std::unordered_map<std::string, size_t> op_id_map_;
  std::unordered_map<std::string, abstract::AbstractBasePtr> node_abs_map_;
  std::stack<FuncGraphPtr> graph_p_;
  FuncGraphPtr top_g_;
  FuncGraphPtr df_builder_;
  FuncGraphPtr curr_g_;
  std::unordered_map<std::string, AbstractListMap> prim_abs_list_;
};

using PynativeExecutorPtr = std::shared_ptr<PynativeExecutor>;

}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_EXECUTE_H_
