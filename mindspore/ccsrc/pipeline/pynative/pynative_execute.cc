/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "pipeline/pynative/pynative_execute.h"
#include "pipeline/pynative/pynative_utils.h"
#include "pipeline/pynative/predict_out_type_map.h"
#include "pipeline/jit/debug/trace.h"
#include "pybind_api/pybind_patch.h"
#include "include/common/utils/config_manager.h"
#include "include/common/pybind_api/api_register.h"
#include "frontend/optimizer/ad/grad.h"
#include "pipeline/jit/pass.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/pynative/op_compiler.h"
#include "pipeline/jit/parse/data_converter.h"
#include "ir/cell.h"
#include "abstract/utils.h"
#include "include/common/utils/stub_tensor.h"
#include "include/common/utils/python_utils.h"

namespace mindspore::pynative {
std::shared_ptr<PyNativeExecutor> PyNativeExecutor::executor_ = nullptr;
ForwardExecutorPtr PyNativeExecutor::forward_executor_ = nullptr;
GradExecutorPtr PyNativeExecutor::grad_executor_ = nullptr;
std::mutex PyNativeExecutor::instance_lock_;

namespace {
enum class AsyncRunOpArgsEnum : size_t { PY_PRIM = 0, PY_INPUTS, PY_ARGS_NUM };
template <typename T, typename... Args>
T PyNativeExecutorTry(const std::function<T(const Args &...)> &method, const Args &... args) {
  const auto &inst = PyNativeExecutor::GetInstance();
  MS_EXCEPTION_IF_NULL(inst);
  MS_EXCEPTION_IF_NULL(method);
  auto already_set_error_handler = [&inst]() {
    // Print function call stack info before release.
    std::ostringstream oss;
    trace::TraceGraphEval();
    trace::GetEvalStackInfo(oss);
    // Call py::print to output function call stack to STDOUT, in case of output the log to file, the user can see
    // these info from screen, no need to open log file to find these info.
    py::print(oss.str());
    MS_LOG(ERROR) << oss.str();
    inst->ClearRes();
  };

  if constexpr (std::is_same_v<T, void>) {
    HandleExceptionRethrow([&method, &args...]() { method(args...); }, already_set_error_handler,
                           [&inst]() { inst->ClearRes(); }, [&inst]() { inst->ClearRes(); });
  } else {
    T res;
    HandleExceptionRethrow([&res, &method, &args...]() { res = method(args...); }, already_set_error_handler,
                           [&inst]() { inst->ClearRes(); }, [&inst]() { inst->ClearRes(); });
    return res;
  }
}

TypePtr PredictOutTypeByName(const std::string &op_name) {
  static PredictOutTypeMap ops_map;
  const auto iter = ops_map.find(op_name);
  if (iter != ops_map.end()) {
    return iter->second;
  }
  static auto operator_fns = ops::OperatorRegister::GetInstance().GetOperatorMap();
  if (operator_fns.find(op_name) == operator_fns.end()) {
    return ops_map[op_name] = kAnyType;
  }
  const auto pre_iter = out_type_prediction.find(op_name);
  auto type = pre_iter == out_type_prediction.end() ? kTensorType : pre_iter->second;
  return ops_map[op_name] = type;
}
}  // namespace

py::object PyNativeExecutor::RunOpAsync(const py::args &args) const {
  if (args.size() != static_cast<size_t>(AsyncRunOpArgsEnum::PY_ARGS_NUM)) {
    MS_LOG(EXCEPTION) << "Two args are needed by RunOp";
  }
  auto prim = args[static_cast<size_t>(AsyncRunOpArgsEnum::PY_PRIM)];
  auto input_args = args[static_cast<size_t>(AsyncRunOpArgsEnum::PY_INPUTS)];
  const auto &adapter = prim.cast<PrimitivePyAdapterPtr>();
  auto run_args = py::make_tuple(prim, adapter->name(), input_args);
  FrontendOpRunInfoPtr op_run_info = forward_executor()->GenerateOpRunInfo(run_args);
  PyNativeExecutorTry(forward_executor()->RunOpS, op_run_info);
  // 1. get top_type from Primitive::PredictOutputType
  auto top_type = PredictOutTypeByName(adapter->name());
  // 2. if predict failed(kAnyType), return after infer(half-asynchronous) or run(synchronous mode)
  if (top_type == kAnyType) {
    return PyNativeAlgo::DataConvert::ValueToPyObj(op_run_info->out_value);
  }
  // 3. create top stub node
  auto node = stub::MakeTopNode(top_type);
  // 4. set abstract and value in asynchronous thread after infer and run
  stub::StubNodePtr stub = node.second;
  stub->SetAbstract(op_run_info->base_op_run_info.abstract);
  stub->SetValue(op_run_info->out_value);
  // 5. return stub node
  return node.first;
}

py::object PyNativeExecutor::RealRunOp(const py::args &args) const {
  FrontendOpRunInfoPtr op_run_info = forward_executor()->GenerateOpRunInfo(args);
  PyNativeExecutorTry(forward_executor()->RunOpS, op_run_info);
  if (PyGILState_Check() == 0) {
    py::gil_scoped_acquire acquire;
    return PyNativeAlgo::DataConvert::ValueToPyObj(op_run_info->out_value);
  } else {
    return PyNativeAlgo::DataConvert::ValueToPyObj(op_run_info->out_value);
  }
}

py::object PyNativeExecutor::CallConstantFolding(const py::args &args) const {
  return forward_executor()->infer_operation()->CallConstantFolding(args);
}

void PyNativeExecutor::set_py_exe_path(const py::object &py_exe_path) const {
  if (!py::isinstance<py::str>(py_exe_path)) {
    MS_LOG(EXCEPTION) << "Failed, py_exe_path input is not a str";
  }
  const auto &py_exe_path_s = py_exe_path.cast<std::string>();
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<std::string>(MS_CTX_PYTHON_EXE_PATH, py_exe_path_s);
}

void PyNativeExecutor::set_kernel_build_server_dir(const py::object &kernel_build_server_dir) const {
  if (!py::isinstance<py::str>(kernel_build_server_dir)) {
    MS_LOG(EXCEPTION) << "Failed, kernel_build_server_dir input is not a str";
  }
  const auto &kernel_build_server_dir_s = kernel_build_server_dir.cast<std::string>();
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<std::string>(MS_CTX_KERNEL_BUILD_SERVER_DIR, kernel_build_server_dir_s);
}

void PyNativeExecutor::ClearRes() const {
  runtime::OpExecutor::GetInstance().Reset();
  pynative::OpCompiler::GetInstance().ClearAllCache();
  pynative::autograd::ClearPyNativeAutoGradStaticRes();

  // Maybe exit in runop step
  auto ms_context = MsContext::GetInstance();
  if (ms_context != nullptr) {
    ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, false);
  }
  ConfigManager::GetInstance().ResetIterNum();
  if (forward_executor_ != nullptr) {
    forward_executor_->ClearRes();
  }
  if (grad_executor_ != nullptr) {
    grad_executor_->ClearRes();
  }
  ad::CleanRes();
  pipeline::ReclaimOptimizer();
  MS_LOG(DEBUG) << "Clear all res";
}

void PyNativeExecutor::Init() {
  MS_LOG(DEBUG) << "Init PyNativeExecutor";
  forward_executor_ = std::make_shared<ForwardExecutor>();
  forward_executor_->Init();
  grad_executor_ = std::make_shared<GradExecutor>(forward_executor_);
  grad_executor_->Init();
  forward_executor_->set_grad_executor(grad_executor_);
}

void PyNativeExecutor::Sync() const { forward_executor()->Sync(); }

void PyNativeExecutor::SetHookChanged(const py::object &cell) const {
  if (!py::isinstance<Cell>(cell)) {
    MS_LOG(EXCEPTION) << "The 'set_hook_changed' function is only supported on Cell object!";
  }
  grad_executor()->SetHookChanged(cell);
}

bool PyNativeExecutor::grad_flag() const { return grad_executor()->grad_flag(); }

void PyNativeExecutor::set_grad_flag(bool flag) const { grad_executor()->set_grad_flag(flag); }

py::object PyNativeExecutor::CheckAlreadyRun(const prim::GradOperationPtr &grad, const py::object &obj,
                                             const py::object &grad_hash_id, const py::args &args) const {
  return grad_executor()->CheckAlreadyRun(grad, obj, grad_hash_id, args);
}

void PyNativeExecutor::NewGraph(const py::object &obj, const py::args &args) const {
  forward_executor()->ProcessBeforeNewGraph(obj, args);

  if (!grad_flag()) {
    MS_LOG(DEBUG) << "Grad flag is false";
    return;
  }
  PyNativeExecutorTry(grad_executor()->InitGraph, obj, args);
  forward_executor()->ProcessBeforeNewGraph(obj);
}

void PyNativeExecutor::EndGraph(const py::object &obj, const py::object &out, const py::args &args) const {
  bool is_cell = py::isinstance<Cell>(obj);
  forward_executor()->ProcessBeforeEndGraph(obj, is_cell);

  if (!grad_flag()) {
    MS_LOG(DEBUG) << "Grad flag is false";
    return;
  }
  PyNativeExecutorTry(grad_executor()->LinkGraph, obj, out, args);
  forward_executor()->ProcessAfterEndGraph(obj, is_cell);
}

py::object PyNativeExecutor::Run() const {
  const auto &ret = PyNativeExecutorTry(grad_executor()->RunGraph);
  return ret;
}

void PyNativeExecutor::GradNet(const prim::GradOperationPtr &grad, const py::object &cell, const py::object &weights,
                               const py::object &grad_position, const py::args &args) const {
  PyNativeExecutorTry(grad_executor()->GradGraph, grad, cell, weights, grad_position, args);
}

py::object PyNativeExecutor::GradMsFunction(const py::object &out, const py::args &args) const {
  const auto &ret = grad_executor()->ms_function()->GradMsFunction(out, args);
  return ret;
}

void PyNativeExecutor::SetLazyBuild(bool enable) const { forward_executor()->set_lazy_build(enable); }

bool PyNativeExecutor::IsFirstCell() const { return forward_executor()->IsFirstCell(); }

void PyNativeExecutor::SetMsFunctionCompileStatus(bool is_compiling, const std::string &phase) const {
  forward_executor()->set_is_ms_function_compiling(is_compiling);
  grad_executor()->ms_function()->set_graph_phase(phase);
}

void PyNativeExecutor::SetDynamicInput(const py::object &cell, const py::args &args) const {
  grad_executor()->SaveDynamicInputsCells(cell);
  MS_LOG(DEBUG) << "Set dynamic shape by set inputs";
}

void RegPyNativeExecutor(const py::module *m) {
  stub::RegStubNodes(m);

  (void)py::class_<PyNativeExecutor, std::shared_ptr<PyNativeExecutor>>(*m, "PyNativeExecutor_")
    .def_static("get_instance", &PyNativeExecutor::GetInstance, "PyNativeExecutor get_instance.")
    .def("is_first_cell", &PyNativeExecutor::IsFirstCell, "check if the first cell.")
    .def("new_graph", &PyNativeExecutor::NewGraph, "pynative new a graph.")
    .def("end_graph", &PyNativeExecutor::EndGraph, "pynative end a graph.")
    .def("check_run", &PyNativeExecutor::CheckAlreadyRun, "pynative check graph run before.")
    .def("grad_ms_function", &PyNativeExecutor::GradMsFunction, "pynative grad for ms_function.")
    .def("grad_net", &PyNativeExecutor::GradNet, "pynative grad graph.")
    .def("clear_res", &PyNativeExecutor::ClearRes, "pynative clear exception res.")
    .def("sync", &PyNativeExecutor::Sync, "pynative sync stream.")
    .def("set_lazy_build", &PyNativeExecutor::SetLazyBuild, "pynative build kernel async")
    .def("__call__", &PyNativeExecutor::Run, "pynative executor run grad graph.")
    .def("grad_flag", &PyNativeExecutor::grad_flag, "pynative grad flag")
    .def("set_hook_changed", &PyNativeExecutor::SetHookChanged, "set pynative hook changed")
    .def("set_grad_flag", &PyNativeExecutor::set_grad_flag, py::arg("flag") = py::bool_(false),
         "Executor set grad flag.")
    .def("set_dynamic_input", &PyNativeExecutor::SetDynamicInput, "set dynamic input")
    .def("set_py_exe_path", &PyNativeExecutor::set_py_exe_path, py::arg("py_exe_path") = py::str(""),
         "set python executable path.")
    .def("set_kernel_build_server_dir", &PyNativeExecutor::set_kernel_build_server_dir,
         py::arg("kernel_build_server_dir") = py::str(""), "set kernel build server directory path.")
    .def("set_ms_function_compile_status", &PyNativeExecutor::SetMsFunctionCompileStatus,
         "set ms_funciton compile status.")
    .def("real_run_op", &PyNativeExecutor::RealRunOp, "Run op pynatively.")
    .def("run_op_async", &PyNativeExecutor::RunOpAsync, "run op asynchronously")
    .def("constant_folding", &PyNativeExecutor::CallConstantFolding, "Call Constant Folding Primitive");
}
}  // namespace mindspore::pynative
