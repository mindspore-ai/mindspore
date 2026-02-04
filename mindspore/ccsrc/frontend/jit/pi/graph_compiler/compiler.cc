/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#include "frontend/jit/pi/graph_compiler/compiler.h"
#include <memory>
#include <string>
#include <algorithm>
#include <utility>
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "include/utils/convert_utils_py.h"
#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"
#include "frontend/jit/pi/graph_compiler/utils.h"
#include "frontend/jit/pi/graph_compiler/parser/byte_code_parser.h"
#include "include/frontend/jit/ps/executor/jit_executor_py.h"
#include "frontend/jit/ps/pipeline.h"
#include "frontend/jit/pi/utils/utils.h"
#include "include/utils/pynative/grad_state.h"
#include "include/utils/pynative/adapter.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace pijit {
namespace {
// Reference : method _generate_run_args of _JitExecutor in api.py
// Parameters should be eliminated in the following case：
// 1.Constant Tensor, reason : constant folding
// 2.Constant Scalar(exclude those will be broaden), reason : constant folding
// 3.None, reason : reason : constant folding or not use
// 4.Empty constant length container(tuple/list/dict): constant folding or not use
// 5.Other(Graph Not Support)
bool IsValidRunArg(const py::object &obj, bool enable_tuple_broaden) {
  if (GraphUtils::IsTensor(obj)) {
    if (GraphUtils::HasInit(obj)) {
      (void)python_adapter::CallPyObjMethod(obj, "init_data");
    }
    return !GraphUtils::IsConst(obj);
  }
  // If the container input is empty and not variable length, graph treat it as constant, it should be erased in inputs.
  if (!GraphUtils::IsDynamicLength(obj) && GraphUtils::IsEmptyContainer(obj)) {
    return false;
  }
  return GraphUtils::IsMutable(obj) || GraphUtils::IsGradForScalar(obj) ||
         (enable_tuple_broaden && GraphUtils::IsTupleCanBroaden(obj));
}

bool CanbeMutable(const py::object &arg) {
  if (GraphUtils::IsConst(arg)) {
    return false;
  }
  // not necessary
  if (GraphUtils::IsMutable(arg)) {
    return false;
  }
  if (py::isinstance<py::dict>(arg) || py::isinstance<py::list>(arg) || py::isinstance<py::tuple>(arg)) {
    py::object o = python_adapter::CallPyFn("mindspore.graph.mutable", "_check_element_type", arg);
    return py::isinstance<py::bool_>(o) && py::bool_(o);
  }
  return false;
}

void MarkArgumentMutable(const py::tuple &args) {
  for (size_t idx = 0; idx < args.size(); idx++) {
    if (CanbeMutable(args[idx])) {
      MS_LOG(DEBUG) << "Make argument mutable, arg index: " << idx << ", arg object: " << py::str(args[idx]);
      args[idx] = python_adapter::CallPyFn("mindspore.graph.mutable", "mutable", args[idx]);
    }
  }
}

void MarkArgumentMutableWithParams(const py::tuple &args, const AnfNodePtrList &params) {
  MS_LOG(DEBUG) << "Number of input arguments: " << args.size();
  for (auto param : params) {
    auto abstract = param->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    if (!abstract->isa<abstract::AbstractTensor>() && abstract->BuildValue() == kValueAny) {
      if (!abstract->has_user_data(pipeline::kActualArgumentIndex)) {  // Might be a bug!
        MS_LOG(INFO) << "Cannot find index of param: " << param->DebugString() << ", " << abstract->ToString();
        continue;
      }
      std::shared_ptr<size_t> index_ptr = abstract->user_data<size_t>(pipeline::kActualArgumentIndex);
      MS_EXCEPTION_IF_NULL(index_ptr);
      auto index = *index_ptr;
      MS_LOG(DEBUG) << "Param: " << param->DebugString() << ", index: " << index;
      MS_EXCEPTION_IF_CHECK_FAIL(index < args.size(), "Arg index out of range.");
      auto arg = args[index];
      if (GraphUtils::IsMutable(arg)) {
        continue;
      }
      py::object o = python_adapter::CallPyFn("mindspore.graph.mutable", "_check_element_type", arg);
      if (py::isinstance<py::bool_>(o) && py::bool_(o)) {
        args[index] = python_adapter::CallPyFn("mindspore.graph.mutable", "mutable", arg);
        MS_LOG(DEBUG) << "Add mutable to object";
      } else {
        MS_LOG(INFO) << "Failed to make argument mutable, arg object: " << py::str(arg);
      }
    }
  }
}

py::tuple MakeNewArgsTuple(const py::tuple &args) {
  py::tuple new_args = py::reinterpret_steal<py::tuple>(PyTuple_New(args.size()));
  for (size_t idx = 0; idx < args.size(); idx++) {
    new_args[idx] = args[idx];
  }
  return new_args;
}

py::tuple EliminateSelf(const py::tuple &args, const std::string &name) {
  if (!args.empty() && !GraphUtils::IsTensor(args[0]) && py::hasattr(args[0], common::SafeCStr(name))) {
    return py::reinterpret_steal<py::tuple>(PyTuple_GetSlice(args.ptr(), 1, args.size()));
  }
  return args;
}

py::tuple EliminateInvalidArgs(const py::tuple &args, int co_flags, bool enable_tuple_broaden) {
  py::list new_args;
  for (size_t idx = 0; idx < args.size(); idx++) {
    if (IsValidRunArg(args[idx], enable_tuple_broaden)) {
      if ((idx < (args.size() - 1) || (IntToSize(co_flags) & CO_VARKEYWORDS) == 0) &&
          py::isinstance<py::dict>(args[idx])) {
        new_args.append(py::reinterpret_steal<py::tuple>(PyDict_Values(args[idx].ptr())));
      } else {
        new_args.append(args[idx]);
      }
    } else {
      MS_LOG(INFO) << "Eliminate invalid argument at index " << idx << ", arg object: " << py::str(args[idx]);
    }
  }
  return py::cast<py::tuple>(new_args);
}

PyObject *RunGraph(const std::string &phase, const py::tuple &args, const std::string &name, int co_flags,
                   bool enable_tuple_broaden) {
  auto graph_executor = pipeline::GetExecutor();
  MS_EXCEPTION_IF_NULL(graph_executor);
  py::tuple args_tuple = EliminateSelf(args, name);
  args_tuple = MakeNewArgsTuple(args_tuple);
  auto origin_fg = graph_executor->GetFuncGraph(phase);
  MS_EXCEPTION_IF_NULL(origin_fg);
  const auto &params = origin_fg->parameters();
  MarkArgumentMutableWithParams(args_tuple, params);
  MarkArgumentMutable(args_tuple);
  args_tuple = EliminateInvalidArgs(args_tuple, co_flags, enable_tuple_broaden);
  MS_LOG(INFO) << "Args for run: " << std::string(py::str(args_tuple));
  py::object ret;
  if (pynative::GradState::Get().grad_flag()) {
    MS_LOG(INFO) << "Do GradJit";
    JitSyntaxLevelScope jit_syntax_level_scope;
    pynative::PyNativeAdapter::SetGraphPhase(phase);
    ret = pynative::PyNativeAdapter::GradJit(args_tuple);
  } else {
    ret = graph_executor->Run(args_tuple, py::str(phase));
  }
  FuncGraphPtr ms_func_graph = graph_executor->GetFuncGraph(phase);
  MS_EXCEPTION_IF_NULL(ms_func_graph);
  ret = python_adapter::CallPyFn("mindspore.graph.api", "_convert_python_data", ret);
  ret.inc_ref();
  return ret.ptr();
}

py::tuple MergeArgsKwargs(PyObject *args, PyObject *kwargs) {
  if (kwargs == nullptr) {
    return py::cast<py::tuple>(args);
  }
  py::list new_args;
  for (const auto &value : py::cast<py::tuple>(args)) {
    new_args.append(value);
  }
  // Graph mode will convert dict input to tuple with values.
  py::list converted_kwargs;
  for (const auto &[key, value] : py::cast<py::dict>(kwargs)) {
    (void)key;
    converted_kwargs.append(value);
  }
  if (py::len(converted_kwargs) != 0) {
    new_args.append(py::cast<py::tuple>(converted_kwargs));
  }
  return py::cast<py::tuple>(new_args);
}
}  // namespace

CallableGraph GraphCompiler::Compile(const FuncGraphPtr &func_graph, const py::tuple &args, const py::dict &kwargs,
                                     const std::string &phase, const CompileInfo &compile_info) {
  MS_EXCEPTION_IF_CHECK_FAIL(!phase.empty(),
                             "Phase name should not be empty for function " + compile_info.co_name_ + ".");
  CallableGraph callable = [compile_info, phase](PyObject *args, PyObject *kwargs) -> PyObject * {
    MS_EXCEPTION_IF_CHECK_FAIL(PyTuple_Check(args), "Excepted a Tuple Object for run args.");
    MS_EXCEPTION_IF_CHECK_FAIL(((kwargs == nullptr) || PyDict_Check(kwargs)),
                               "Excepted nullptr or a Dict Object for run kwargs.");

    py::tuple tuple = MergeArgsKwargs(args, kwargs);
    return RunGraph(phase, tuple, compile_info.co_name_, compile_info.co_flags_, false);  // need adapt for optimizer
  };

  auto jit_executor = pipeline::GetExecutor();
  if (jit_executor->HasCompiled(phase)) {
    return callable;
  }

  if (func_graph == nullptr) {
    return nullptr;
  }
  py::tuple new_arg = MakeNewArgsTuple(args);
  const auto &parameters = func_graph->parameters();
  auto args_cnt = parameters.size() - func_graph->fv_param_count();
  if (new_arg.size() > args_cnt) {
    new_arg = EliminateSelf(new_arg, compile_info.co_name_);
  }
  MarkArgumentMutable(new_arg);
  if (MsContext::GetInstance()->CanDump(kIntroductory)) {
    DumpIR("graph_before_compile.ir", func_graph);
  }
  MS_LOG(INFO) << "Args for compile: " << std::string(py::str(new_arg));

  auto origin_num = compile_info.origin_top_input_num_;

  const auto &params = func_graph->parameters();
  if (origin_num != params.size()) {
    MS_LOG(INFO) << "Reorder top function graph inputs.";
    AnfNodePtrList new_params;
    (void)std::copy(params.begin(), params.begin() + origin_num, std::back_inserter(new_params));
    (void)std::copy_if(params.begin() + origin_num, params.end(), std::back_inserter(new_params),
                       [](const AnfNodePtr &param) { return !param->abstract()->isa<abstract::AbstractRefTensor>(); });
    (void)std::copy_if(params.begin() + origin_num, params.end(), std::back_inserter(new_params),
                       [](const AnfNodePtr &param) { return param->abstract()->isa<abstract::AbstractRefTensor>(); });
    func_graph->set_parameters(new_params);
  }

  // Clone a func_graph as root graph in order to clear and drop the temporary func_graphs in the manager.
  // If not, the old context of graph will be reused in abstract_analyze phase.
  auto fg_to_compile = BasicClone(func_graph);
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  mng->KeepRoots({fg_to_compile});

  (void)jit_executor->CompileInner(fg_to_compile, new_arg, kwargs, phase, true);

  return callable;
}

std::pair<std::string, CallableGraph> GraphCompiler::Compile(const FuncGraphPtr &func_graph,
                                                             const CompileInfo &compile_info) {
  MS_LOG(INFO) << "Start FuncGraph compile";
  if (func_graph == nullptr) {
    MS_LOG(INFO) << "FuncGraph is NULL!";
    return std::make_pair("", nullptr);
  }
  std::string phase =
    compile_info.co_filename_ + "_" + std::to_string(compile_info.co_firstlineno_) + "_" + compile_info.co_name_;
  const auto &parameters = func_graph->parameters();
  py::tuple args(parameters.size() - func_graph->fv_param_count());
  size_t cur_fv_param_count = 0;
  for (size_t i = 0; i < parameters.size(); ++i) {
    auto para = parameters[i]->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    if (para->has_default()) {
      cur_fv_param_count++;
      continue;
    }
    auto para_abstract = para->abstract();
    MS_EXCEPTION_IF_NULL(para_abstract);
    phase += "_" + para_abstract->ToString();
    auto input_obj = para->user_data<py::object>("pi_jit_py_obj");
    MS_EXCEPTION_IF_NULL(input_obj);
    args[i - cur_fv_param_count] = *input_obj;
  }
  phase += ".pi_jit";
  CallableGraph callable = GraphCompiler::Compile(func_graph, args, py::dict(), phase, compile_info);
  MS_LOG(INFO) << "End FuncGraph compile";
  return std::make_pair(phase, callable);
}
}  // namespace pijit
}  // namespace mindspore
