/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_PI_JIT_GRAD_EXECUTOR_H_
#define MINDSPORE_PI_JIT_GRAD_EXECUTOR_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include "pipeline/jit/pi/auto_grad/bprop_func_graph_manager.h"
#include "pybind11/stl.h"

namespace mindspore {
namespace pijit {
namespace grad {
namespace py = pybind11;

class FunctionNode;
using FunctionNodePtr = std::shared_ptr<FunctionNode>;

class GradExecutor;
using GradExecutorPtr = std::shared_ptr<GradExecutor>;

class GradExecutor {
 public:
  GradExecutor() : manager_(std::make_shared<BpropFuncGraphManager>()) {}
  virtual ~GradExecutor() = default;

  static GradExecutorPtr GetInstance() { return grad_executor_; }

  FuncGraphPtr PrimBpropGraphPass(const FuncGraphPtr &prim_grad_graph);
  FuncGraphPtr GetAccumulateGraph(const py::object &tensor);
  FuncGraphPtr GetBpropGraph(const AnfNodePtr &func, const ValuePtrList &inputs, const ValuePtr &out,
                             const ValuePtr &dout);
  FuncGraphPtr GetFuncGraphBpropGraph(const std::string &phase, const py::tuple &args);
  ValuePtr RunGraph(const FuncGraphPtr &func_graph, const ValuePtrList &inputs, const ValuePtr &out,
                    const ValuePtr &dout);
  ValuePtr RunGraph(const FuncGraphPtr &func_graph, const ValuePtrList &inputs);
  ValuePtr RunGraph(const FuncGraphPtr &func_graph, const VectorRef &inputs);

 private:
  static GradExecutorPtr grad_executor_;
  BpropFuncGraphManagerPtr manager_;
};
}  // namespace grad
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_GRAD_EXECUTOR_H_
