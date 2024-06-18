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
#include "pipeline/pynative/grad/ir/ir_bprop.h"
#include "pipeline/pynative/predict_out_type_map.h"
#include "pipeline/jit/ps/debug/trace.h"
#include "pybind_api/pybind_patch.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "pybind_api/ir/hook_py.h"
#include "include/common/utils/config_manager.h"
#include "include/common/pybind_api/api_register.h"
#include "frontend/optimizer/ad/grad.h"
#include "pipeline/jit/ps/pass.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/pynative/op_compiler.h"
#include "runtime/pynative/op_runner.h"
#include "include/common/profiler.h"
#include "ir/cell.h"
#include "include/common/utils/stub_tensor.h"
#include "include/common/utils/python_utils.h"
#include "frontend/operator/ops_front_infer_function.h"
#include "kernel/kernel_mod_cache.h"
#include "runtime/pipeline/pipeline.h"

namespace mindspore::pynative {
std::shared_ptr<PyNativeExecutor> PyNativeExecutor::executor_ = nullptr;
ForwardExecutorPtr PyNativeExecutor::forward_executor_ = nullptr;
GradExecutorPtr PyNativeExecutor::grad_executor_ = nullptr;
std::mutex PyNativeExecutor::instance_lock_;

namespace {
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

// Tensor may be used before the execution of the asynchronous task.
void SetCallbackForInputTensor(const std::vector<ValuePtr> &input_values) {
  for (auto &input : input_values) {
    MS_EXCEPTION_IF_NULL(input);
    if (input->isa<tensor::BaseTensor>()) {
      auto tensor = input->cast<tensor::BaseTensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      tensor->set_need_pipeline_sync(true);
    }
  }
}
}  // namespace

void PyNativeExecutor::StoreAsyncStatus(const FrontendOpRunInfoPtr &op_run_info) const {
  // Pure function running or cell not set mix precision
  op_run_info->async_status.disable_mix_precision = forward_executor()->CellNotSetMixedPrecision(op_run_info);
  op_run_info->async_status.is_jit_compiling = forward_executor()->is_jit_compiling();
  op_run_info->async_status.custom_bprop_cell_count = grad_executor()->custom_bprop_cell_count();
}

py::object PyNativeExecutor::RunOpStub(const py::args &args) const {
  FrontendOpRunInfoPtr op_run_info = forward_executor()->GenerateOpRunInfo(args, true);
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunOp,
                                     op_run_info->base_op_run_info.op_name, false, true);
  SetCallbackForInputTensor(op_run_info->op_grad_info->input_value);

  StoreAsyncStatus(op_run_info);
  const auto &op_name = op_run_info->base_op_run_info.op_name;
  // 1. get top_type from Primitive::PredictOutputType
  auto top_type = PredictOutType(op_run_info);
  // 2. if disable PyTraceAsync, return after infer(half-asynchronous) or run(synchronous mode)
  if (!forward_executor()->EnablePipeline(op_name)) {
    // Wait for async task finish
    forward_executor()->WaitForwardTask();
    // RunOp sync
    PyNativeExecutorTry(forward_executor()->RunOpS, op_run_info);
    return PyNativeAlgo::DataConvert::ValueToPyObj(op_run_info->real_out);
  }
  // 3. create top stub node
  auto node = stub::MakeTopNode(top_type);
  // The task in the AsyncQueue may need to acquire gil.
  GilReleaseWithCheck release_gil;
  // 4. set abstract and value in asynchronous thread after infer and run
  op_run_info->stub_output = node.second;
  forward_executor()->DispatchFrontendTask(op_run_info);
  // 5. return stub node
  return node.first;
}

py::object PyNativeExecutor::RunSliceOpStub(const std::vector<ValuePtr> &input_values,
                                            const std::vector<SliceOpInfoPtr> &slice_op_infos) const {
  runtime::ProfilerStageRecorder recorder(runtime::ProfilerStage::kRunOp);
  SetCallbackForInputTensor(input_values);
  auto requires_grad = grad_executor()->RequiresGrad();
  if (!forward_executor()->EnablePipeline("")) {
    forward_executor()->WaitForwardTask();
    auto ret = forward_executor()->RunSliceOpFrontend(input_values, slice_op_infos, requires_grad, nullptr);
    return PyNativeAlgo::DataConvert::ValueToPyObj(ret);
  }
  auto top_type = kTensorType;
  auto node = stub::MakeTopNode(top_type);
  GilReleaseWithCheck release_gil;
  forward_executor()->DispatchSilceOpFrontendTask(input_values, slice_op_infos, requires_grad, node.second);
  return node.first;
}

py::object PyNativeExecutor::RealRunOp(const py::args &args) const {
  FrontendOpRunInfoPtr op_run_info = forward_executor()->GenerateOpRunInfo(args);
  StoreAsyncStatus(op_run_info);
  PyNativeExecutorTry(forward_executor()->RunOpS, op_run_info);
  if (PyGILState_Check() == 0) {
    py::gil_scoped_acquire acquire;
    return PyNativeAlgo::DataConvert::ValueToPyObj(op_run_info->real_out);
  } else {
    return PyNativeAlgo::DataConvert::ValueToPyObj(op_run_info->real_out);
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
  runtime::Pipeline::Get().WaitAll();
  // Clear forward tasks before clear op graphs cache.
  pynative::OpCompiler::GetInstance().ClearAllCache();
  kernel::KernelModCache::GetInstance().ClearAllCache();
  pynative::autograd::ClearAutoGradCache();
  tensor::RegisterHook::ClearHookMap();

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
  forward_executor_->RefreshForwardCallback();
  runtime::ProfilerAnalyzer::GetInstance().SetThreadIdToName(std::this_thread::get_id(), "Python");
}

void PyNativeExecutor::Sync() const {
  forward_executor()->Sync();
  runtime::ProfilerAnalyzer::GetInstance().EndStep();
  runtime::ProfilerAnalyzer::GetInstance().StartStep();
}

void PyNativeExecutor::SetHookCellId(const py::object &cell) const {
  grad_executor()->set_hook_cell_id(PyNativeAlgo::PyParser::GetIdByPyObj(cell));
}

bool PyNativeExecutor::grad_flag() const { return grad_executor()->grad_flag(); }

void PyNativeExecutor::set_grad_flag(bool flag) const { grad_executor()->set_grad_flag(flag); }

bool PyNativeExecutor::enable_grad() const { return grad_executor()->enable_grad(); }

void PyNativeExecutor::set_enable_grad(bool enable_grad) const { grad_executor()->set_enable_grad(enable_grad); }

py::object PyNativeExecutor::CheckAlreadyRun(const prim::GradOperationPtr &grad, const py::object &obj,
                                             const py::object &weights, const py::object &grad_hash_id,
                                             const py::args &args) const {
  return grad_executor()->CheckAlreadyRun(grad, obj, weights, grad_hash_id, args);
}

void PyNativeExecutor::NewGraph(const py::object &obj, const py::args &args) const {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeNewGraph,
                                     runtime::ProfilerRecorder::kNoName, false);
  PyNativeExecutorTry(grad_executor()->InitGraph, obj, args);
}

void PyNativeExecutor::EndGraph(const py::object &obj, const py::object &out, const py::args &args) const {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeEndGraph,
                                     runtime::ProfilerRecorder::kNoName, false);
  PyNativeExecutorTry(grad_executor()->LinkGraph, obj, out, args);
}

py::object PyNativeExecutor::RunGrad(const prim::GradOperationPtr &grad, const py::object &cell,
                                     const py::object &weights, const py::object &grad_position,
                                     const py::args &args) const {
  runtime::ProfilerStageRecorder recorder(runtime::ProfilerStage::kRunGrad);
  return PyNativeExecutorTry(grad_executor()->Run, grad, cell, weights, grad_position, args);
}

py::object PyNativeExecutor::GradJit(const py::object &out, const py::args &args) const {
  const auto &ret = grad_executor()->jit()->GradJit(out, args);
  return ret;
}

void PyNativeExecutor::SetMixedPrecisionType(const MixedPrecisionType mix_type, bool is_push) const {
  return forward_executor()->set_mix_precision_type(mix_type, is_push);
}

void PyNativeExecutor::WorkerJoin() {
  GilReleaseWithCheck release_gil;
  runtime::Pipeline::Get().frontend_stage()->WorkerJoin();
}

void PyNativeExecutor::SetJitCompileStatus(bool is_compiling, const std::string &phase) const {
  forward_executor()->set_is_jit_compiling(is_compiling);
  grad_executor()->jit()->set_graph_phase(phase);
}

void PyNativeExecutor::SetIsRunRecompute(bool is_runing_recompute) const {
  grad_executor()->set_is_run_recompute(is_runing_recompute);
}

void PyNativeExecutor::set_forward_use_dynamic_shape_process(bool flag) const {
  grad_executor()->set_forward_use_dynamic_shape_process(flag);
#ifndef ENABLE_SECURITY
  if (flag) {
    profiler::ProfilerManager::GetInstance()->SetNetDynamicShapeStatus();
  }
#endif
}

void PyNativeExecutor::SetDynamicInput(const py::object &obj, const py::args &args) const {
  grad_executor()->SaveDynamicInputsCells(obj, args);
  if (grad_executor()->dynamic_shape()->enable_unknown_shape()) {
    grad_executor()->dynamic_shape()->SetDynamicInput(obj, args);
  }
}

py::object PyNativeExecutor::GetDynamicInput(const py::object &actual_input) const {
  if (grad_executor()->dynamic_shape()->enable_unknown_shape()) {
    MS_LOG(DEBUG) << "Get dynamic shape for jit";
    return grad_executor()->dynamic_shape()->GetDynamicInput(actual_input);
  }
  return actual_input;
}

void PyNativeExecutor::ParentBeforeFork() {
  MS_LOG(DEBUG) << "PyNativeExecutor prepare before fork.";
  runtime::Pipeline::Get().WaitAll();
  MS_LOG(DEBUG) << "PyNativeExecutor prepare before fork done.";
}

void PyNativeExecutor::ChildAfterFork() {
  MS_LOG(DEBUG) << "PyNativeExecutor reinitialize after fork.";
  MS_LOG(DEBUG) << "Clear OpCompiler Cache.";
  pynative::OpCompiler::GetInstance().ClearAllCache();
  if (forward_executor_ != nullptr) {
    MS_LOG(DEBUG) << "Clear forward_executor_ resources.";
    forward_executor_->ClearRes();
    // Call ForwardExecutor::ReInit() to update device_target_
    forward_executor_->ReInit();
    MS_LOG(DEBUG) << "Reinitialize forward_executor_.";
    forward_executor_->ChildAfterFork();
  }
  // Reset PyNativeExecutor resources
  if (grad_executor_ != nullptr) {
    MS_LOG(DEBUG) << "Clear grad_executor_ resources.";
    grad_executor_->ClearRes();
    MS_LOG(DEBUG) << "Reinitialize grad_executor_.";
    grad_executor_->ChildAfterFork();
  }
  runtime::OpRunner::ChildAfterFork();
  MS_LOG(DEBUG) << "PyNativeExecutor reinitialize after fork done.";
}

void PyNativeExecutor::SetAsyncForGraph(bool flag) const {
  runtime::OpExecutor::GetInstance().set_async_for_graph(flag);
}

void RegPyNativeExecutor(const py::module *m) {
  stub::RegStubNodes(m);

  (void)py::class_<PyNativeExecutor, std::shared_ptr<PyNativeExecutor>>(*m, "PyNativeExecutor_")
    .def_static("get_instance", &PyNativeExecutor::GetInstance, "PyNativeExecutor get_instance.")
    .def("set_mixed_precision_type", &PyNativeExecutor::SetMixedPrecisionType, "set cell mixed precision type.")
    .def("new_graph", &PyNativeExecutor::NewGraph, "pynative new a graph.")
    .def("end_graph", &PyNativeExecutor::EndGraph, "pynative end a graph.")
    .def("check_run", &PyNativeExecutor::CheckAlreadyRun, "pynative check graph run before.")
    .def("grad_jit", &PyNativeExecutor::GradJit, "pynative grad for jit.")
    .def("clear_res", &PyNativeExecutor::ClearRes, "pynative clear exception res.")
    .def("sync", &PyNativeExecutor::Sync, "pynative sync stream.")
    .def("grad", &PyNativeExecutor::RunGrad, "pynative executor run grad.")
    .def("grad_flag", &PyNativeExecutor::grad_flag, "pynative grad flag")
    .def("enable_grad", &PyNativeExecutor::enable_grad, "pynative enable grad, used for with no_grad")
    .def("set_hook_id", &PyNativeExecutor::SetHookCellId, "set pynative hook changed")
    .def("set_grad_flag", &PyNativeExecutor::set_grad_flag, py::arg("flag") = py::bool_(false),
         "Executor set grad flag.")
    .def("set_enable_grad", &PyNativeExecutor::set_enable_grad, py::arg("enable_grad") = py::bool_(true),
         "pynative set enable grad")
    .def("set_eval_use_dynamic_shape_process", &PyNativeExecutor::set_forward_use_dynamic_shape_process,
         "set eval use dynamic shape process.")
    .def("set_dynamic_input", &PyNativeExecutor::SetDynamicInput, "set dynamic input")
    .def("get_dynamic_input", &PyNativeExecutor::GetDynamicInput, "get dynamic input")
    .def("set_py_exe_path", &PyNativeExecutor::set_py_exe_path, py::arg("py_exe_path") = py::str(""),
         "set python executable path.")
    .def("set_kernel_build_server_dir", &PyNativeExecutor::set_kernel_build_server_dir,
         py::arg("kernel_build_server_dir") = py::str(""), "set kernel build server directory path.")
    .def("set_jit_compile_status", &PyNativeExecutor::SetJitCompileStatus, "set jit compile status.")
    .def("set_is_run_recompute", &PyNativeExecutor::SetIsRunRecompute, "set grad is in recompile status.")
    .def("real_run_op", &PyNativeExecutor::RealRunOp, "run op synchronously")
    .def("run_op_async", &PyNativeExecutor::RunOpStub, "run op asynchronously")
    .def("set_async_for_graph", &PyNativeExecutor::SetAsyncForGraph, py::arg("flag") = py::bool_(false),
         "Executor set async flag.")
    .def("constant_folding", &PyNativeExecutor::CallConstantFolding, "Call Constant Folding Primitive");
}
}  // namespace mindspore::pynative
