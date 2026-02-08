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

#ifndef MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_EXECUTOR_EXECUTOR_PY_H_
#define MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_EXECUTOR_EXECUTOR_PY_H_

#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <map>
#include <list>

#include "pybind11/pybind11.h"

#include "base/base.h"
#include "frontend/jit/ps/base.h"
#include "include/utils/visible.h"

namespace mindspore {
// namespace to support pipeline structures definition
namespace pipeline {
class FRONTEND_EXPORT ExecutorPy : public std::enable_shared_from_this<ExecutorPy> {
 public:
  ExecutorPy() = default;
  virtual ~ExecutorPy() = default;
  bool Compile(const py::object &source, const py::tuple &args, const py::dict &kwargs, const py::object &phase,
               const py::dict &config);
  py::object Run(const py::tuple &args, const py::object &phase);
  void set_enable_tuple_broaden(bool enable_tuple_broaden) { enable_tuple_broaden_ = enable_tuple_broaden; }
  // Generate a key for mapping function graph
  py::object GenerateArgumentsKey(const py::object &obj, const py::tuple &args, const py::dict &kwargs,
                                  bool enable_tuple_broaden = false);
  void ClearCompileArgumentsResource();
  void SetJitConfig(const py::dict &jit_config);
  std::map<std::string, std::string> GetJitConfig();
  virtual void CleanCompileRes(const ResourcePtr &resource) = 0;
  FuncGraphPtr GetFuncGraph(const std::string &phase);
  std::vector<bool> CheckFuncGraphSequenceParamAbstract(const std::string &phase);
  void SetJitPrimalFuncGraph(const FuncGraphPtr &primal_func_graph, const std::string &phase);
  FuncGraphPtr GetJitPrimalFuncGraph(const std::string &phase);
  FuncGraphPtr GetJitGradGraph(const std::string &phase);
  void SetJitGradGraph(const FuncGraphPtr &grad_graph, const std::string &phase);
  py::dict GetParams(const std::string &phase);
  bool HasCompiled(const std::string &phase) const;
  void DelNetRes(const py::object &source, const py::set &id);
  const std::string &phase() const { return phase_; }
  void set_queue_name(const std::string &queue_name) { queue_name_ = queue_name; }
  std::string get_queue_name(const std::string &dataset_phase);
  void set_compile_cache_dep_files(const py::list &compile_cache_dep_files) {
    compile_cache_dep_files_ = compile_cache_dep_files;
  }
  py::list compile_cache_dep_files() const { return compile_cache_dep_files_; }
  void set_weights_values(const py::dict &weights) { weights_ = weights; }
  py::dict weights() const { return weights_; }
  // Check consistency of two arguments for mapping function graph
  void CheckArgumentsConsistency(const py::tuple &compile_args, const py::tuple &args_list, const py::object &target);
  py::bytes GetFuncGraphProto(const std::string &phase, const std::string &ir_type, const bool &incremental);
  py::bytes GetOnnxFuncGraphProto(const std::string &phase, const std::vector<std::string> &input_names,
                                  const std::vector<std::string> &outputs_names, const int &opset_version,
                                  const bool &export_params, const bool &keep_initializers_as_inputs,
                                  const py::dict &dynamic_axes, const bool &extra_save_params,
                                  const std::string &save_file_dir);
  virtual bool CompileInner(const FuncGraphPtr &graph, const py::tuple &args, const py::dict &kwargs,
                            const std::string &phase, bool trace_flag) = 0;
  bool executor_running() const { return executor_running_; }
  const std::string &obj_desc() const { return obj_desc_; }
  int32_t max_call_depth() const { return max_call_depth_; }
  void set_max_call_depth(int32_t max_call_depth) { max_call_depth_ = max_call_depth; }
  const ValuePtrList &real_arguments() const { return real_arguments_; }
  void SetRealArguments(const py::tuple &args, const py::dict &kwargs);
  bool compile_cache_consistent() const { return compile_cache_consistent_; }

 protected:
  virtual bool CompileInner(const py::object &source, const py::tuple &args, const py::dict &kwargs,
                            const py::object &phase) = 0;
  virtual py::object RunInner(const py::tuple &args, const py::object &phase) = 0;
  virtual void DelOneNetRes(const py::handle &py_phase) = 0;
  virtual void SaveCompiledGraph(const std::string &phase) = 0;
  ResourcePtr GetResource(const std::string &phase);
  void ProcessVmArg(const py::tuple &args, const std::string &phase, VectorRef *const arg_list);
  std::shared_ptr<std::function<BaseRef(const VectorRef &)>> GetVmEvalFunc(const std::string &phase,
                                                                           const std::string &kind = kOutput);
  void ClearRunArgumentsResource(size_t input_arg_size, VectorRef *arg_list);
  // If enable compile cache, get the compile cache resource.
  void InitCompileCacheInfo(const ResourcePtr &resource, const std::string &phase);
  void InitCompileCacheResource(const ResourcePtr &resource, const std::string &phase);
  void set_process_id();
  void ConvertSymbolicShape(const py::tuple &args, AbstractBasePtrList *args_abs);

  std::map<std::string, ExecutorInfoPtr> info_;
  std::string phase_;
  std::string source_;
  std::string obj_desc_;
  bool enable_tuple_broaden_{false};
  ValuePtrList real_arguments_;
  std::map<PyObject *, std::pair<ValuePtr, AbstractBasePtr>> cur_convert_input_;

 private:
  void ClearCurConvertInput();
  void ReleaseResourceOnException(const py::object &phase);

  std::string queue_name_;
  py::list compile_cache_dep_files_;
  py::dict weights_;
  bool executor_running_{false};
  bool compile_cache_consistent_{true};
  int32_t max_call_depth_{-1};
  pid_t process_id_{0};
};
using ExecutorPyPtr = std::shared_ptr<ExecutorPy>;

FRONTEND_EXPORT ExecutorPyPtr GetExecutor(const std::string &phase = "");

FRONTEND_EXPORT void CleanCache();

}  // namespace pipeline
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_EXECUTOR_EXECUTOR_PY_H_
