/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_EXECUTOR_GRAPH_EXECUTOR_PY_H_
#define MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_EXECUTOR_GRAPH_EXECUTOR_PY_H_

#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <map>
#include <mutex>
#include <unordered_map>
#include <list>

#include "pybind11/pybind11.h"

#include "include/frontend/jit/ps/executor/executor_py.h"

#include "base/base.h"
#include "ir/tensor.h"
#include "frontend/parallel/strategy.h"
#include "include/utils/visible.h"

namespace mindspore {
// namespace to support pipeline structures definition
namespace pipeline {
// A function pipeline.
class FRONTEND_EXPORT GraphExecutorPy : public ExecutorPy {
 public:
  static std::shared_ptr<GraphExecutorPy> GetInstance() {
    std::lock_guard<std::mutex> i_lock(instance_lock_);
    if (executor_ == nullptr) {
      executor_ = std::shared_ptr<GraphExecutorPy>(new (std::nothrow) GraphExecutorPy());
    }
    executor_->set_process_id();
    return executor_;
  }

  ~GraphExecutorPy() override;

  bool CompileInner(const FuncGraphPtr &graph, const py::tuple &args, const py::dict &kwargs, const std::string &phase,
                    bool trace_flag) override;

  void ConvertArgs(const py::tuple &args, const py::dict &kwargs, const ResourcePtr &resource, bool is_auto_parallel,
                   abstract::AbstractBasePtrList *args_abs, std::vector<ValuePtr> *arguments);
  py::bytes GetOptimizeGraphProto(const std::string &phase);

  void BuildGraph(const py::dict &init_params, const std::string &phase) const;
  void ExportGraph(const std::string &file_name, const std::string &phase, const py::object encrypt = py::none(),
                   char *key = nullptr);
  py::bytes GetRandomStatus(const std::string &phase) const;
  void UpdataParamNodeDefaultInput(const std::string &phase,
                                   const std::unordered_map<std::string, py::object> &params_value);
  void PyExePath(const py::object &py_exe_path) const;
  void KernelBuildServerDir(const py::object &kernel_build_server_dir) const;
  py::dict GetParameterLayout(const std::string &phase);
  py::tuple FlopsCollection(const std::string &phase);
  // Get CNode name, input node name and attribute from each graph
  py::dict GetParallelGraphInfo(const std::string &phase);
  py::dict GetCNodeStrategy(const std::string &phase);
  py::list GetParallelParameterNameList(const std::string &phase);
  void SetCNodeStrategy(const std::string &name, const parallel::Strategies &strategy);
  size_t GetNumOpsInfo(const std::string &phase);
  void SetNumOpsInfo(size_t num_ops);
  py::dict GetAllreduceFusion(const std::string &phase);
  static void ClearRes();
  void SetOptimizeConfig(const py::list &optimize_cfg);
  std::string GetOptimizeConfig();
  void SetConfigPasses(const py::list &passes);
  py::list GetRunningPasses();

  void ParentBeforeFork();
  void ParentAfterFork();
  void ChildAfterFork();

  void CleanCompileRes(const ResourcePtr &resource) override;

 private:
  GraphExecutorPy() = default;
  void ParallelPostProcess(const string &phase, bool use_compile_cache);

  void ConvertObjectToTensors(const py::dict &dict,
                              std::map<std::string, std::shared_ptr<tensor::Tensor>> *const tensors,
                              const FuncGraphPtr &anf_graph) const;

  void DelOneNetRes(const py::handle &py_phase) override;
  bool CompileInner(const py::object &source, const py::tuple &args, const py::dict &kwargs,
                    const py::object &phase) override;
  py::object RunInner(const py::tuple &args, const py::object &phase) override;
  void SaveCompiledGraph(const std::string &phase) override;

  static std::shared_ptr<GraphExecutorPy> executor_;
  static std::mutex instance_lock_;
  std::map<std::string, py::dict> stra_dict_;
  std::map<std::string, size_t> phase_to_num_op_info_;
};
using GraphExecutorPyPtr = std::shared_ptr<GraphExecutorPy>;
}  // namespace pipeline
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_EXECUTOR_GRAPH_EXECUTOR_PY_H_
