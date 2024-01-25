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
#include "pipeline/jit/pi/auto_grad/grad_executor.h"
#include <algorithm>
#include <iterator>
#include <memory>
#include "backend/graph_compiler/backend.h"
#include "include/common/utils/contract.h"
#include "pipeline/pynative/pynative_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace pijit {
namespace grad {
GradExecutorPtr GradExecutor::grad_executor_ = std::make_shared<GradExecutor>();

FuncGraphPtr GradExecutor::PrimBpropGraphPass(const FuncGraphPtr &prim_grad_graph) {
  return manager_->PrimBpropGraphPass(prim_grad_graph);
}

FuncGraphPtr GradExecutor::GetAccumulateGraph(const py::object &tensor) {
  auto value = pynative::PyNativeAlgo::DataConvert::PyObjToValue(tensor);
  return manager_->GetAccumulateGraph(value, value);
}

FuncGraphPtr GradExecutor::GetBpropGraph(const AnfNodePtr &func, const ValuePtrList &inputs, const ValuePtr &out,
                                         const ValuePtr &dout) {
  if (IsValueNode<Primitive>(func)) {
    return manager_->GetPrimBpropGraph(GetValueNode<PrimitivePtr>(func), inputs, out, dout);
  }
  MS_EXCEPTION_IF_CHECK_FAIL(IsValueNode<FuncGraph>(func), "");
  return manager_->GetFuncGraphBpropGraph(GetValueNode<FuncGraphPtr>(func), inputs, out, dout);
}

ValuePtr GradExecutor::RunGraph(const FuncGraphPtr &func_graph, const ValuePtrList &inputs, const ValuePtr &out,
                                const ValuePtr &dout) {
  VectorRef args;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(args),
                 [](const ValuePtr &input) { return BaseRef(input); });
  args.push_back(BaseRef(out));
  args.push_back(BaseRef(dout));
  return RunGraph(func_graph, args);
}

ValuePtr GradExecutor::RunGraph(const FuncGraphPtr &func_graph, const ValuePtrList &inputs) {
  VectorRef args;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(args),
                 [](const ValuePtr &input) { return BaseRef(input); });
  return RunGraph(func_graph, args);
}

ValuePtr GradExecutor::RunGraph(const FuncGraphPtr &func_graph, const VectorRef &inputs) {
  auto mgr = func_graph->manager();
  if (mgr == nullptr) {
    mgr = Manage(func_graph, true);
  }
  func_graph->set_manager(mgr);
  VectorRef outputs;
  auto context_ptr = MsContext::GetInstance();
  auto backend = compile::CreateBackend();
  if (context_ptr->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    compile::MindRTBackendBase *rtbc_ptr = std::dynamic_pointer_cast<compile::MindRTBackendBase>(backend).get();
    auto actor_info = rtbc_ptr->CompileGraphs(func_graph);
    rtbc_ptr->RunGraph(actor_info, inputs, &outputs);
  } else {
    compile::MsBackend *msbc_ptr = std::dynamic_pointer_cast<compile::MsBackend>(backend).get();
    auto graph_id = backend->CompileGraph(NOT_NULL(func_graph));
    outputs = msbc_ptr->RunGraph(graph_id, inputs);
  }
  auto ret = pynative::PyNativeAlgo::DataConvert::VectorRefToValue(outputs, true, true);
  MS_EXCEPTION_IF_CHECK_FAIL(ret->isa<ValueTuple>(), "Return value should be a tuple.");
  auto tuple = ret->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_CHECK_FAIL(tuple->size() == 1, "Size should be 1.");
  return tuple->value()[0];
}

FuncGraphPtr GradExecutor::GetFuncGraphBpropGraph(const std::string &phase, const py::tuple &args) {
  auto graph_executor = pipeline::GraphExecutorPy::GetInstance();
  MS_EXCEPTION_IF_NULL(graph_executor);
  auto forward_graph = graph_executor->GetFuncGraph(phase);
  ValuePtrList inputs;
  std::transform(args.begin(), args.end(), std::back_inserter(inputs), [](const py::handle &arg) {
    return pynative::PyNativeAlgo::DataConvert::PyObjToValue(py::cast<py::object>(arg));
  });
  return manager_->GetFuncGraphBpropGraph(forward_graph, inputs, nullptr, nullptr);
}
}  // namespace grad
}  // namespace pijit
}  // namespace mindspore
