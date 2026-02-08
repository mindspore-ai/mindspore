/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_RESOURCE_INTERFACE_H_
#define MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_RESOURCE_INTERFACE_H_

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <future>
#include <mutex>
#include <utility>
#include <functional>

#include "utils/hash_map.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "utils/any.h"
#include "utils/profile.h"

#include "frontend/jit/ps/resource_base.h"
#include "frontend/jit/ps/static_analysis/prim.h"
#include "frontend/jit/ps/static_analysis/static_analysis.h"
#include "load_mindir/load_model.h"
#include "frontend/jit/ps/compile_cache_manager.h"

namespace mindspore {
namespace pipeline {
namespace py = pybind11;
enum PiplineLevel : int {
  // Not running in jit pipeline or graph pipeline.
  kLevelNone = 0,
  // Running in a simple pipeline which contains only the necessary passes and no parallel passes.
  kLevelJit,
  // Running in a whole pipeline which contains the necessary passes and all the parallel passes.
  kLevelGraph,
};

class FRONTEND_EXPORT Resource : public ResourceBase {
 public:
  explicit Resource(const py::object &obj = py::none());

  ~Resource() override;

  abstract::AnalysisEnginePtr engine() { return engine_; }

  static bool IsTypeInBuiltInMap(const TypeId &type);

  static Any GetMethodPtr(const TypeId &type, const std::string &name);

  static Any GetAttrPtr(const TypeId &type, const std::string &name);

  const py::object &source_input() const { return source_input_; }

  FuncGraphPtr func_graph() const { return func_graph_; }
  void set_func_graph(const FuncGraphPtr &func_graph) { func_graph_ = func_graph; }

  FuncGraphPtr optimize_graph() const { return optimize_graph_; }
  void set_optimize_graph(const FuncGraphPtr &optimize_graph) { optimize_graph_ = optimize_graph; }

  const abstract::AbstractBasePtrList &args_abs() const { return args_abs_; }
  void set_args_abs(const abstract::AbstractBasePtrList &args_abs) { args_abs_ = args_abs; }

  const ValuePtrList &arguments() const { return arguments_; }
  void set_arguments(const ValuePtrList &arguments) { arguments_ = arguments; }

  const ValuePtrList &real_arguments() const { return real_arguments_; }
  void set_real_arguments(const ValuePtrList &args_list) { real_arguments_ = args_list; }

  void set_vm_loop(const bool &flag, const int64_t size) {
    vm_loop_flag_ = flag;
    loop_size_ = size;
  }
  void set_is_load(bool flag) { is_load_ = flag; }
  bool is_load() const { return is_load_; }
  bool vm_loop_flag() const { return vm_loop_flag_; }
  int64_t loop_size() const { return loop_size_; }

  const LayoutMap &layout_map() const { return layout_map_; }

  // Get the cached func_graph and parameters layout map.
  void GetCompileCacheResource(const py::list &compile_cache_dep_files, const py::dict &weights,
                               const std::string &queue_name, size_t compile_cache_id, bool *compile_cache_consistent,
                               bool has_python_script);
  void CacheFuncGraph() const;
  bool EnableCompileCache() const { return compile_cache_manager_ != nullptr; }

  // Reclaim resource and clear the cache.
  // ExecutorPy::Compile() can be called multiple times, so cache
  // should be cleared.
  void Clean();

  // Get the mutex for backend initializing.
  static std::mutex &GetBackendInitMutex() { return backend_init_mutex_; }

  void set_is_pynative_grad_view_inplace(bool is_pynative_grad_view_inplace) {
    is_pynative_grad_view_inplace_ = is_pynative_grad_view_inplace;
  }
  bool is_pynative_grad_view_inplace() const { return is_pynative_grad_view_inplace_; }

  PiplineLevel pipeline_level() const { return pipeline_level_; }
  void set_pipeline_level(PiplineLevel pipeline_level) { pipeline_level_ = pipeline_level; }

 private:
  abstract::AnalysisEnginePtr engine_;
  FuncGraphPtr func_graph_;
  FuncGraphPtr optimize_graph_;
  // The arguments may contain a Parameter, we need connect it to the Parameter default value of func graph.
  // We keep all arguments inputs here for subsequent procedure.
  std::vector<ValuePtr> arguments_;
  ValuePtrList real_arguments_;
  abstract::AbstractBasePtrList args_abs_;
  // The source obj to compile, usually a `Cell` or `jit` decorated function.
  py::object source_input_;
  bool is_cleaned_;
  // The func_graph_ is loaded from mindir
  bool is_load_{false};
  bool vm_loop_flag_{false};
  int64_t loop_size_{1};
  LayoutMap layout_map_{};
  CompileCacheManagerPtr compile_cache_manager_{nullptr};
  // The backend related fields for async initializing.
  static std::mutex backend_init_mutex_;
  bool is_pynative_grad_view_inplace_{false};
  PiplineLevel pipeline_level_{kLevelNone};
};

using ResourcePtr = std::shared_ptr<pipeline::Resource>;

// Clear resource
FRONTEND_EXPORT void ClearAttrAndMethodMap();
FRONTEND_EXPORT void ClearPassConfigure();
FRONTEND_EXPORT void ClearOpPrimPyRegister();
FRONTEND_EXPORT void CleanParserResource();
FRONTEND_EXPORT void ClearAnalysisSchedule();
FRONTEND_EXPORT void ClearAnalysisResultCacheMgr();
FRONTEND_EXPORT void ClearPrimitiveEvaluatorMap();
FRONTEND_EXPORT void ClearActionConfigure();
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_RESOURCE_INTERFACE_H_
