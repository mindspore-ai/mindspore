/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_EXECUTOR_JIT_EXECUTOR_PY_H_
#define MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_EXECUTOR_JIT_EXECUTOR_PY_H_

#include <string>
#include <memory>

#include "include/frontend/jit/ps/executor/executor_py.h"
#include "include/utils/visible.h"

namespace mindspore {
namespace pipeline {
class FRONTEND_EXPORT JitExecutorPy : public ExecutorPy {
 public:
  static std::shared_ptr<JitExecutorPy> GetInstance() {
    std::lock_guard<std::mutex> i_lock(instance_lock_);
    if (executor_ == nullptr) {
      executor_ = std::shared_ptr<JitExecutorPy>(new (std::nothrow) JitExecutorPy());
    }
    executor_->set_process_id();
    return executor_;
  }

  ~JitExecutorPy() override = default;

  py::tuple SplitGraph(const py::object &func_graph_obj);
  static void ClearRes();
  void CleanCompileRes(const ResourcePtr &resource) override;
  // For pi jit compiler.
  bool CompileInner(const FuncGraphPtr &graph, const py::tuple &args, const py::dict &kwargs, const std::string &phase,
                    bool) override;

 private:
  JitExecutorPy() = default;

  bool CompileInner(const py::object &source, const py::tuple &args, const py::dict &kwargs,
                    const py::object &phase) override;
  py::object RunInner(const py::tuple &args, const py::object &phase) override;
  void DelOneNetRes(const py::handle &py_phase) override;
  void SaveCompiledGraph(const std::string &phase) override;
  void ConvertArgs(const py::tuple &args, const py::dict &kwargs, const ResourcePtr &resource,
                   const ExecutorInfoPtr &executor_info);
  bool SetSource(const py::object &source);
  bool SetPhase(const py::object &phase);

  static std::shared_ptr<JitExecutorPy> executor_;
  static std::mutex instance_lock_;
};
using JitExecutorPyPtr = std::shared_ptr<JitExecutorPy>;
}  // namespace pipeline
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_EXECUTOR_JIT_EXECUTOR_PY_H_
