/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/graph_jit/graph_compiler/compiler.h"
#include <memory>
#include <string>
#include "pipeline/jit/graph_jit/graph_compiler/func_graph_builder.h"
#include "pipeline/jit/graph_jit/graph_compiler/utils.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/debug/anf_ir_dump.h"
#include "ir/func_graph.h"
#include "pipeline/jit/graph_jit/graph_compiler/parser/byte_code_parser.h"
#include "pipeline/jit/ps/pipeline.h"
#include "pipeline/pynative/pynative_execute.h"

namespace mindspore {
namespace jit {
namespace graph {
namespace {
// Reference : method _generate_run_args of _MindsporeFunctionExecutor in api.py
// Parameters should be eliminated in the following case：
// 1.Constant Tensor, reason : constant folding
// 2.Constant Scalar(exclude those will be broaden), reason : constant folding
// 3.None, reason : reason : constant folding or not use
// 4.Other(Graph Not Support)
bool IsValidRunArg(const py::object &obj, bool enable_tuple_broaden) {
  if (GraphUtils::IsTensor(obj)) {
    if (GraphUtils::HasInit(obj)) {
      (void)python_adapter::CallPyObjMethod(obj, "init_data");
    }
    return !GraphUtils::IsConst(obj);
  }
  return GraphUtils::IsMutable(obj) || GraphUtils::IsGradForScalar(obj) ||
         (enable_tuple_broaden && GraphUtils::IsTupleCanBroaden(obj));
}

py::tuple MergeAllArgments(PyObject *args, PyObject *kwargs) {
  if (kwargs == nullptr) {
    return py::cast<py::tuple>(args);
  }
  py::list new_args;
  for (const auto &value : py::cast<py::tuple>(args)) {
    new_args.append(value);
  }
  for (const auto &[key, value] : py::cast<py::dict>(kwargs)) {
    (void)key;
    new_args.append(value);
  }
  return py::cast<py::tuple>(new_args);
}

py::tuple EliminateStubTensor(const py::tuple &args) {
  py::tuple new_args = py::reinterpret_steal<py::tuple>(PyTuple_New(args.size()));
  for (size_t idx = 0; idx < args.size(); idx++) {
    new_args[idx] = IsStubTensor(args[idx]) ? python_adapter::CallPyObjMethod(args[idx], "stub_sync") : args[idx];
  }
  return new_args;
}

py::tuple EliminateSelf(const py::tuple &args, const std::string &name) {
  if (!args.empty() && !GraphUtils::IsTensor(args[0]) && py::hasattr(args[0], common::SafeCStr(name))) {
    return py::reinterpret_steal<py::tuple>(PyTuple_GetSlice(args.ptr(), 1, args.size()));
  }
  return args;
}

py::tuple EliminateInvalidArgs(const py::tuple &args, bool enable_tuple_broaden) {
  py::list new_args;
  for (size_t idx = 0; idx < args.size(); idx++) {
    if (IsValidRunArg(args[idx], enable_tuple_broaden)) {
      new_args.append(args[idx]);
    }
  }
  return py::cast<py::tuple>(new_args);
}

py::tuple ExpandVariableArgs(const py::tuple &args, const PyCodeObject &code) {
  if ((code.co_flags & CO_VARARGS) == 0x0) {
    return args;
  }
  py::tuple var_args = py::cast<py::tuple>(args[code.co_argcount]);
  py::list new_args;
  for (int index = 0; index < code.co_argcount; index++) {
    new_args.append(args[index]);
  }
  for (const auto &var_arg : var_args) {
    new_args.append(var_arg);
  }
  for (size_t index = code.co_argcount + 1; index < args.size(); index++) {
    new_args.append(args[index]);
  }
  return py::cast<py::tuple>(new_args);
}
}  // namespace

CallableGraph Compiler::Compile(const PyFunctionObject &func, const PyFrameObject &frame, const std::string &phase) {
  const PyCodeObject *code = frame.f_code;
  std::string name = py::cast<std::string>(code->co_name);
  MS_EXCEPTION_IF_CHECK_FAIL(!phase.empty(), "Phase name should not be empty for function " + name + ".");

  PyObject *f = reinterpret_cast<PyObject *>(const_cast<PyFunctionObject *>(&func));
  bool enable_tuple_broaden = GraphUtils::IsTupleBroadenEnable(py::cast<py::object>(f));
  CallableGraph callable = [code, enable_tuple_broaden, phase](PyObject *args, PyObject *kwargs) -> PyObject * {
    MS_EXCEPTION_IF_CHECK_FAIL(PyTuple_Check(args), "Excepted a Tuple Object for run args.");
    MS_EXCEPTION_IF_CHECK_FAIL(((kwargs == nullptr) || PyDict_Check(kwargs)),
                               "Excepted nullptr or a Dict Object for run kwargs.");

    py::tuple tuple = MergeAllArgments(args, kwargs);
    tuple = ExpandVariableArgs(tuple, *code);
    std::string name = py::cast<std::string>(code->co_name);
    tuple = EliminateSelf(tuple, name);
    tuple = EliminateStubTensor(tuple);
    tuple = EliminateInvalidArgs(tuple, enable_tuple_broaden);
    py::object ret = mindspore::pipeline::GraphExecutorPy::GetInstance()->Run(tuple, py::str(phase));
    int mode = MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE);
    auto executor = pynative::PyNativeExecutor::GetInstance();
    if (mode == kPynativeMode && executor->grad_flag()) {
      executor->grad_executor()->jit()->set_graph_phase(phase);
      executor->GradJit(ret, tuple);
    }
    ret = python_adapter::CallPyFn("mindspore.common.api", "_convert_python_data", ret);
    ret.inc_ref();
    return ret.ptr();
  };

  auto graph_executor = mindspore::pipeline::GraphExecutorPy::GetInstance();
  if (graph_executor->HasCompiled(phase)) {
    return callable;
  }

  int arg_cnt = code->co_argcount + code->co_kwonlyargcount;
  if (code->co_flags & CO_VARARGS) {
    arg_cnt++;
  }
  py::list locals = py::reinterpret_steal<py::list>(PyDict_Values(frame.f_locals));
  py::tuple args = py::reinterpret_steal<py::tuple>(PyList_AsTuple(PyList_GetSlice(locals.ptr(), 0, arg_cnt)));
  py::dict kwargs = (code->co_flags & CO_VARKEYWORDS) == 0x0 ? py::dict() : py::cast<py::dict>(locals[arg_cnt]);
  args = EliminateStubTensor(args);
  auto byteCodeParser = std::make_shared<ByteCodeParser>(func);
  ir::FunctionNodePtr func_node = byteCodeParser->Parse();
  bool is_method = !args.empty() && !GraphUtils::IsTensor(args[0]) && py::hasattr(args[0], common::SafeCStr(name));
  if (is_method) {
    func_node->MarkMethod();
  }
  FuncGraphPtr graph = FuncGraphBuilder::BuildFuncGraph(func_node, args, kwargs);
  if (graph == nullptr) {
    return nullptr;
  }
  args = EliminateSelf(args, name);
  args = ExpandVariableArgs(args, *code);
  (void)graph_executor->CompileInner(graph, args, kwargs, phase, true);

  return callable;
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
