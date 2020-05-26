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

#ifndef MINDSPORE_CCSRC_PYNATIVE_PYNATIVE_EXECUTE_H_
#define MINDSPORE_CCSRC_PYNATIVE_PYNATIVE_EXECUTE_H_

#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <stack>

#include "pybind11/pybind11.h"

#include "pynative/base.h"
#include "utils/context/ms_context.h"
#include "ir/anf.h"
#include "pipeline/resource.h"
#include "operator/composite/composite.h"

namespace mindspore {
namespace pynative {

namespace py = pybind11;
using ResourcePtr = std::shared_ptr<pipeline::Resource>;
using GradOperationPtr = std::shared_ptr<prim::GradOperation>;

py::object RunOpInVM(const OpExecInfoPtr &op_exec_info, PynativeStatusCode *status);

py::tuple RunOp(const py::args &args);

py::list ConvertInputs(const PrimitivePyPtr &prim, const py::list &py_args);

void ClearPyNativeSession();

struct GraphInfo {
  std::unordered_map<std::string, AnfNodePtr> param_map;
  std::unordered_map<std::string, std::pair<AnfNodePtr, int>> obj_node_map;
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
  void EndGraph(const py::object &cell, const py::object &out, const py::args &args);
  void GradNet(const GradOperationPtr &grad, const py::object &cell, const py::object &weights, const py::args &args);
  void Clear();
  void Clean();
  bool grad_flag() { return grad_flag_; }
  void set_grad_flag(bool flag) { grad_flag_ = flag; }
  AnfNodePtr GetInput(const py::object &obj, const py::object &op_mask);
  AnfNodePtr GetObjNode(const py::object &obj);
  FuncGraphPtr curr_g() { return curr_g_; }
  void set_pyobj(FuncGraphPtr g, const std::string obj) { graph_info_map_[g].objects.push_back(obj); }
  void set_obj_node_map(FuncGraphPtr g, const std::string obj, AnfNodePtr node) {
    graph_info_map_[g].obj_node_map[obj] = std::make_pair(node, -1);
  }
  void set_obj_node_map(FuncGraphPtr g, const std::string obj, AnfNodePtr node, int index) {
    graph_info_map_[g].obj_node_map[obj] = std::make_pair(node, index);
  }
  AnfNodePtr MakeCNode(const py::args &args, const py::tuple &out);
  py::object Run(const py::tuple &args, const py::object &phase);

  void Pushp();
  void Popp();
  FuncGraphPtr GradGraph(FuncGraphPtr g, const GradOperationPtr &grad_op, const std::vector<AnfNodePtr> &weights,
                         size_t arg_size);

  ~PynativeExecutor();

 private:
  PynativeExecutor();
  static std::shared_ptr<PynativeExecutor> executor_;
  static std::mutex instance_lock_;
  static ResourcePtr resource_;
  bool grad_flag_;
  std::unordered_map<std::string, FuncGraphPtr> graph_map_;
  std::unordered_map<std::string, FuncGraphPtr> cell_graph_map_;
  std::unordered_map<FuncGraphPtr, GraphInfo> graph_info_map_;
  std::stack<FuncGraphPtr> graph_p_;
  FuncGraphPtr top_g_;
  FuncGraphPtr df_builder_;
  FuncGraphPtr curr_g_;
};

using PynativeExecutorPtr = std::shared_ptr<PynativeExecutor>;

}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PYNATIVE_PYNATIVE_EXECUTE_H_
