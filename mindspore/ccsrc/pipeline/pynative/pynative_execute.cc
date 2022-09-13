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
#include "pipeline/jit/debug/trace.h"
#include "pybind_api/pybind_patch.h"
#include "include/common/utils/config_manager.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/pybind_api/api_register.h"
#include "frontend/optimizer/ad/grad.h"
#include "pipeline/jit/pass.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/pynative/op_compiler.h"
#include "ir/cell.h"

namespace mindspore::pynative {
PyNativeExecutorPtr PyNativeExecutor::executor_ = nullptr;
ForwardExecutorPtr PyNativeExecutor::forward_executor_ = nullptr;
GradExecutorPtr PyNativeExecutor::grad_executor_ = nullptr;
std::mutex PyNativeExecutor::instance_lock_;

namespace {
template <typename T, typename... Args>
void PyNativeExecutorTry(const std::function<void(T *ret, const Args &...)> &method, T *ret, const Args &... args) {
  const auto inst = PyNativeExecutor::GetInstance();
  MS_EXCEPTION_IF_NULL(inst);
  MS_EXCEPTION_IF_NULL(method);
  try {
    method(ret, args...);
  } catch (const py::error_already_set &ex) {
    // print function call stack info before release
    std::ostringstream oss;
    trace::TraceGraphEval();
    trace::GetEvalStackInfo(oss);
    // call py::print to output function call stack to STDOUT, in case of output the log to file, the user can see
    // these info from screen, no need to open log file to find these info
    py::print(oss.str());
    MS_LOG(ERROR) << oss.str();
    inst->ClearRes();
    // re-throw this exception to Python interpreter to handle it
    throw(py::error_already_set(ex));
  } catch (const py::index_error &ex) {
    inst->ClearRes();
    throw py::index_error(ex);
  } catch (const py::value_error &ex) {
    inst->ClearRes();
    throw py::value_error(ex);
  } catch (const py::type_error &ex) {
    inst->ClearRes();
    throw py::type_error(ex);
  } catch (const py::name_error &ex) {
    inst->ClearRes();
    throw py::name_error(ex);
  } catch (const std::exception &ex) {
    inst->ClearRes();
    // re-throw this exception to Python interpreter to handle it
    throw(std::runtime_error(ex.what()));
  } catch (...) {
    inst->ClearRes();
#ifndef _MSC_VER
    auto exception_type = abi::__cxa_current_exception_type();
    MS_EXCEPTION_IF_NULL(exception_type);
    std::string ex_name(exception_type->name());
    MS_LOG(EXCEPTION) << "Error occurred when compile graph. Exception name: " << ex_name;
#else
    MS_LOG(EXCEPTION) << "Error occurred when compile graph.";
#endif
  }
}
}  // namespace

py::object RealRunOp(const py::args &args) {
  const auto &executor = PyNativeExecutor::GetInstance();
  MS_EXCEPTION_IF_NULL(executor);
  FrontendOpRunInfoPtr op_run_info = executor->forward_executor()->GenerateOpRunInfo(args);
  py::object ret = py::none();
  PyNativeExecutorTry(executor->forward_executor()->RunOpS, &ret, op_run_info);
  return ret;
}

py::object GetDynShape(const py::args &args) {
  const auto &executor = PyNativeExecutor::GetInstance();
  MS_EXCEPTION_IF_NULL(executor);
  return executor->forward_executor()->dynamic_shape()->GetDynShape(args);
}

py::object CallConstantFolding(const py::args &args) {
  const auto &prim_arg = args[0];
  const auto &adapter = py::cast<PrimitivePyAdapterPtr>(prim_arg);
  MS_EXCEPTION_IF_NULL(adapter);
  auto prim = adapter->attached_primitive();
  if (prim == nullptr) {
    prim = std::make_shared<PrimitivePy>(prim_arg, adapter);
    adapter->set_attached_primitive(prim);
  }
  if (!prim->HasPyObj()) {
    MS_LOG(EXCEPTION) << "Pyobj is empty";
  }
  const auto &v = PyNativeAlgo::DataConvert::PyObjToValue(args[1]);
  std::vector<AbstractBasePtr> input_abs;
  input_abs.push_back(v->ToAbstract());
  prim->BeginRecordAddAttr();
  auto eval_ret = EvalOnePrim(prim, input_abs);
  MS_EXCEPTION_IF_NULL(eval_ret);
  AbstractBasePtr infer_res = eval_ret->abstract();
  MS_EXCEPTION_IF_NULL(infer_res);
  prim->EndRecordAddAttr();
  auto value_ptr = PyNativeAlgo::DataConvert::PyObjToValue(ConvertAbstractToPython(infer_res)[ATTR_VALUE]);
  return ValueToPyData(value_ptr);
}

void PyNativeExecutor::set_py_exe_path(const py::object &py_exe_path) const {
  if (!py::isinstance<py::str>(py_exe_path)) {
    MS_LOG(EXCEPTION) << "Failed, py_exe_path input is not a str";
  }
  const auto &py_exe_path_s = py::cast<std::string>(py_exe_path);
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<std::string>(MS_CTX_PYTHON_EXE_PATH, py_exe_path_s);
}

void PyNativeExecutor::set_kernel_build_server_dir(const py::object &kernel_build_server_dir) const {
  if (!py::isinstance<py::str>(kernel_build_server_dir)) {
    MS_LOG(EXCEPTION) << "Failed, kernel_build_server_dir input is not a str";
  }
  const auto &kernel_build_server_dir_s = py::cast<std::string>(kernel_build_server_dir);
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<std::string>(MS_CTX_KERNEL_BUILD_SERVER_DIR, kernel_build_server_dir_s);
}

void PyNativeExecutor::ClearRes() {
  MS_LOG(DEBUG) << "Clear all res";
  runtime::OpExecutor::GetInstance().Reset();
  pynative::OpCompiler::GetInstance().ClearAllCache();

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
}

void PyNativeExecutor::Init() {
  MS_LOG(DEBUG) << "Init PyNativeExecutor";
  forward_executor_ = std::make_shared<ForwardExecutor>();
  grad_executor_ = std::make_shared<GradExecutor>(forward_executor_, forward_executor_->dynamic_shape());
  forward_executor_->set_grad_executor(grad_executor_);
}

void PyNativeExecutor::Sync() const { forward_executor()->Sync(); }

void PyNativeExecutor::SetHookChanged(const py::object &cell) const {
  if (!py::isinstance<Cell>(cell)) {
    MS_LOG(EXCEPTION) << "The 'set_hook_changed' function is only supported on Cell object!";
  }
  grad_executor()->SetHookChanged(cell);
}

void PyNativeExecutor::set_graph_phase(const std::string &graph_phase) const {
  grad_executor()->ms_function()->set_graph_phase(graph_phase);
}

bool PyNativeExecutor::grad_flag() const { return grad_executor()->grad_flag(); }

void PyNativeExecutor::set_grad_flag(bool flag) const { grad_executor()->set_grad_flag(flag); }

void PyNativeExecutor::SetDynamicInput(const py::object &cell, const py::args &args) const {
  MS_LOG(DEBUG) << "Set dynamic input for feed mode from cell id " << PyNativeAlgo::PyParser::GetIdByPyObj(cell);
  forward_executor()->dynamic_shape()->SetDynamicInput(cell, args);
  // After set input, check previous top cell can be make to dynamic shape
  forward_executor()->dynamic_shape()->CheckPreviousTopCellCanBeDynamicShape(cell, args);
}

py::object PyNativeExecutor::GetDynamicInput(const py::object &actual_input) const {
  return forward_executor()->dynamic_shape()->GetDynamicInput(actual_input);
}

py::object PyNativeExecutor::CheckAlreadyRun(const prim::GradOperationPtr &grad, const py::object &cell,
                                             const py::object &grad_hash_id, const py::args &args) const {
  return grad_executor()->CheckAlreadyRun(grad, cell, grad_hash_id, args);
}

py::object PyNativeExecutor::Run(const py::object &cell, const py::object &sens_param, const py::tuple &args) const {
  py::object ret;
  PyNativeExecutorTry(grad_executor()->RunGraph, &ret, cell, sens_param, args);
  return ret;
}

void PyNativeExecutor::ClearCell(const py::object &cell) const {
  const auto &cell_id = PyNativeAlgo::PyParser::GetIdByPyObj(cell);
  MS_LOG(DEBUG) << "Clear cell res, cell id " << cell_id;
  grad_executor()->ClearCellRes(cell_id);
}

void PyNativeExecutor::ClearGrad(const py::object &cell, const py::args &args) const {
  MS_LOG(DEBUG) << "Clear grad";
  return grad_executor()->ClearGrad(cell, args);
}

void PyNativeExecutor::NewGraph(const py::object &cell, const py::args &args) const {
  forward_executor()->ProcessBeforeNewGraph(cell, args);

  if (!grad_flag()) {
    MS_LOG(DEBUG) << "Grad flag is false";
    return;
  }
  const py::object ret;
  PyNativeExecutorTry(grad_executor()->InitGraph, &ret, cell, args);
}

void PyNativeExecutor::EndGraph(const py::object &cell, const py::object &out, const py::args &args) const {
  forward_executor()->ProcessBeforeEndGraph(cell, args);

  if (!grad_flag()) {
    MS_LOG(DEBUG) << "Grad flag is false";
    return;
  }
  const py::object ret;
  PyNativeExecutorTry(grad_executor()->LinkGraph, &ret, cell, out, args);
  forward_executor()->ProcessAfterEndGraph();
}

void PyNativeExecutor::GradNet(const prim::GradOperationPtr &grad, const py::object &cell, const py::object &weights,
                               const py::object &grad_position, const py::args &args) const {
  const py::object ret;
  PyNativeExecutorTry(grad_executor()->GradGraph, &ret, grad, cell, weights, grad_position, args);
}

py::object PyNativeExecutor::GradMsFunction(const py::object &out, const py::args &args) const {
  return grad_executor()->ms_function()->GradMsFunction(out, args);
}

void PyNativeExecutor::SetLazyBuild(bool enable) const { forward_executor()->set_lazy_build(enable); }

bool PyNativeExecutor::IsFirstCell() const { return forward_executor()->IsFirstCell(); }

void PyNativeExecutor::SetMsFunctionCompileStatus(bool is_compiling) {
  forward_executor()->set_is_ms_function_compiling(is_compiling);
}

void RegPynativeExecutor(py::module *m) {
  (void)py::class_<PyNativeExecutor, std::shared_ptr<PyNativeExecutor>>(*m, "PynativeExecutor_")
    .def_static("get_instance", &PyNativeExecutor::GetInstance, "PyNativeExecutor get_instance.")
    .def("is_first_cell", &PyNativeExecutor::IsFirstCell, "check if the first cell.")
    .def("new_graph", &PyNativeExecutor::NewGraph, "pynative new a graph.")
    .def("end_graph", &PyNativeExecutor::EndGraph, "pynative end a graph.")
    .def("check_run", &PyNativeExecutor::CheckAlreadyRun, "pynative check graph run before.")
    .def("grad_ms_function", &PyNativeExecutor::GradMsFunction, "pynative grad for ms_function.")
    .def("grad_net", &PyNativeExecutor::GradNet, "pynative grad graph.")
    .def("clear_cell", &PyNativeExecutor::ClearCell, "pynative clear status.")
    .def("clear_res", &PyNativeExecutor::ClearRes, "pynative clear exception res.")
    .def("clear_grad", &PyNativeExecutor::ClearGrad, "pynative clear grad status.")
    .def("sync", &PyNativeExecutor::Sync, "pynative sync stream.")
    .def("set_lazy_build", &PyNativeExecutor::SetLazyBuild, "pynative build kernel async")
    .def("__call__", &PyNativeExecutor::Run, "pynative executor run grad graph.")
    .def("set_graph_phase", &PyNativeExecutor::set_graph_phase, "pynative set graph phase")
    .def("grad_flag", &PyNativeExecutor::grad_flag, "pynative grad flag")
    .def("set_hook_changed", &PyNativeExecutor::SetHookChanged, "set pynative hook changed")
    .def("set_grad_flag", &PyNativeExecutor::set_grad_flag, py::arg("flag") = py::bool_(false),
         "Executor set grad flag.")
    .def("set_dynamic_input", &PyNativeExecutor::SetDynamicInput, "set dynamic input")
    .def("get_dynamic_input", &PyNativeExecutor::GetDynamicInput, "get dynamic input")
    .def("set_py_exe_path", &PyNativeExecutor::set_py_exe_path, py::arg("py_exe_path") = py::str(""),
         "set python executable path.")
    .def("set_kernel_build_server_dir", &PyNativeExecutor::set_kernel_build_server_dir,
         py::arg("kernel_build_server_dir") = py::str(""), "set kernel build server directory path.")
    .def("set_ms_function_compile_status", &PyNativeExecutor::SetMsFunctionCompileStatus,
         "set ms_funciton compile status.");
}
}  // namespace mindspore::pynative
