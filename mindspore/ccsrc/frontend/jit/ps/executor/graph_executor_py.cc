/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#include "include/frontend/jit/ps/executor/graph_executor_py.h"

#include <memory>
#include <map>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <unordered_map>
#include <functional>

#include "include/backend/backend_manager/backend_base.h"
#include "include/backend/backend_manager/backend_manager.h"

#include "tools/profiler/profiling.h"

#include "frontend/optimizer/ad/dfunctor.h"
#include "frontend/optimizer/ad/prim_bprop_optimizer.h"

#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/dynamic_shape/dynamic_shape.h"
#include "frontend/parallel/graph_util/flops_collection.h"
#include "frontend/parallel/graph_util/get_parallel_info.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/strategy.h"

#include "mindspore/ccsrc/utils/ir_dump/dump_proto.h"
#include "mindspore/ccsrc/utils/symbol_engine/utils.h"
#include "include/utils/compile_cache_context.h"
#include "include/utils/tensor_py.h"
#include "include/utils/tensor_py_wrapper.h"
#include "include/utils/parallel_context.h"
#include "utils/config_manager.h"
#include "include/utils/pipeline/pipeline_utils.h"
#include "frontend/jit/ps/event_message_print.h"
#include "frontend/jit/ps/pass_config.h"
#include "frontend/jit/ps/pipeline.h"
#include "include/frontend/jit/ps/pass_interface.h"
#include "include/frontend/jit/ps/pipeline_interface.h"
#include "include/frontend/jit/ps/parse/py_data_convert.h"
#include "include/frontend/optimizer/ad/grad_interface.h"

#include "pybind_api/pybind_patch.h"
#include "pybind11/pybind11.h"

#include "utils/phase.h"
#include "ir/tensor_new.h"

namespace mindspore {
// namespace to support intermediate representation definition
namespace pipeline {
using Tensor = mindspore::tensor::Tensor;
using MetaTensor = mindspore::tensor::MetaTensor;

GraphExecutorPyPtr GraphExecutorPy::executor_ = nullptr;
std::mutex GraphExecutorPy::instance_lock_;

void GraphExecutorPy::ParentBeforeFork() {
  MS_LOG(DEBUG) << "GraphExecutorPy prepare before fork.";
  MS_LOG(DEBUG) << "Stop AnalysisSchedule tasks.";
  abstract::AnalysisSchedule::GetInstance().Stop();
  MS_LOG(DEBUG) << "GraphExecutorPy prepare before fork done.";
}

void GraphExecutorPy::ParentAfterFork() {
  MS_LOG(DEBUG) << "GraphExecutorPy in parent process reinitialize after fork.";
  MS_LOG(DEBUG) << "Restart AnalysisSchedule tasks.";
  abstract::AnalysisSchedule::GetInstance().Start();
  MS_LOG(DEBUG) << "GraphExecutorPy in parent process reinitialize after fork done.";
}

void GraphExecutorPy::ChildAfterFork() {
  MS_LOG(DEBUG) << "GraphExecutorPy in child process reinitialize after fork.";
  MS_LOG(DEBUG) << "Restart AnalysisSchedule tasks.";
  abstract::AnalysisSchedule::GetInstance().Start();
  MS_LOG(DEBUG) << "GraphExecutorPy in child process reinitialize after fork done.";
}

py::bytes GraphExecutorPy::GetOptimizeGraphProto(const std::string &phase) {
  if (info_.count(phase) == 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "No phase in executor: " << phase;
  }
  FuncGraphPtr fg_ptr = info_[phase]->resource->optimize_graph();
  if (fg_ptr == nullptr) {
    MS_LOG(WARNING) << "Can not find optimize graph.";
    return "";
  }
  std::string proto_str = GetFuncGraphProtoString(fg_ptr);
  if (proto_str.empty()) {
    MS_LOG(EXCEPTION) << "Export optimize graph proto string failed.";
  }
  return proto_str;
}

py::dict GraphExecutorPy::GetParallelGraphInfo(const std::string &phase) {
  MS_LOG(DEBUG) << "GetParallelGraphInfo!";
  std::string parallel_phase = phase + kStepParallelGraph;
  auto graph = GetFuncGraph(parallel_phase);
  if (graph == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Can not access FuncGraph according to phase: " << parallel_phase;
  }

  return mindspore::parallel::GetParallelCNodeInfoFromGraph(graph);
}

py::dict GraphExecutorPy::GetParameterLayout(const std::string &phase) {
  MS_LOG(DEBUG) << "GetParameterLayout!";
  std::string layout_graph = phase + kStepParallelGraph;
  auto graph = GetFuncGraph(layout_graph);
  if (graph == nullptr) {
    if (info_.find(phase) == info_.end()) {
      MS_LOG(INFO) << "Not found in GraphExecutor info for phase: " << phase;
      return {};
    }
    auto resource = info_[phase]->resource;
    return mindspore::parallel::GetParameterLayoutFromResource(resource);
  }
  return mindspore::parallel::GetParameterLayoutFromGraph(graph);
}

py::tuple GraphExecutorPy::FlopsCollection(const std::string &phase) {
  auto graph = GetFuncGraph(phase);
  return mindspore::parallel::FlopsCollection(graph);
}

py::dict GraphExecutorPy::GetCNodeStrategy(const std::string &phase) {
  MS_LOG(DEBUG) << "GetCNodeStrategy!";
  return stra_dict_[phase];
}

py::list GraphExecutorPy::GetParallelParameterNameList(const std::string &phase) {
  std::string param_graph = phase + kStepParallelGraph;
  auto graph = GetFuncGraph(param_graph);
  if (graph == nullptr) {
    auto resource = info_[phase]->resource;
    return mindspore::parallel::GetParallelParameterNameListFromResource(resource);
  }
  return mindspore::parallel::GetParallelParameterNameListFromGraph(graph);
}

void GraphExecutorPy::SetCNodeStrategy(const std::string &name, const parallel::Strategies &strategy) {
  MS_LOG(DEBUG) << "SetCNodeStrategy!";
  stra_dict_[phase_][py::str(name)] = strategy;
}

size_t GraphExecutorPy::GetNumOpsInfo(const std::string &phase) {
  MS_LOG(DEBUG) << "GetNumOpsInfo!";
  return phase_to_num_op_info_[phase];
}

void GraphExecutorPy::SetNumOpsInfo(size_t num_ops) {
  MS_LOG(DEBUG) << "SetNumOpsInfo!";
  phase_to_num_op_info_[phase_] = num_ops;
}

py::dict GraphExecutorPy::GetAllreduceFusion(const std::string &phase) {
  MS_LOG(INFO) << "GetAllreduceFusion!";
  auto graph = GetFuncGraph(phase);
  return mindspore::parallel::GetAllreduceFusion(graph);
}

void GraphExecutorPy::DelOneNetRes(const py::handle &py_phase) {
  if (!pybind11::isinstance<py::str>(py_phase)) {
    MS_LOG(ERROR) << "Expect string phase, but got " << py::str(py_phase);
    return;
  }
  auto phase = pybind11::cast<std::string>(py_phase);
  MS_LOG(INFO) << "Delete one net resource start, phase: " << phase;
  auto iter = info_.find(phase);
  auto clear = false;
  if (iter != info_.end()) {
    clear = true;
    auto res = iter->second->resource;
    if (res->HasResult(kStepParallelGraph)) {
      std::string layout_graph = phase + kStepParallelGraph;
      (void)info_.erase(layout_graph);
    }
    (void)info_.erase(phase);
    MS_LOG(DEBUG) << "Delete phase: " << phase << ", info size: " << info_.size();
  }
  if (clear) {
    // Do clear here to avoid any pointer for resource.
    FuncGraphLoopBreaker::Inst().ClearCellGraphs(phase);
    FuncGraphLoopBreaker::Inst().CleanUnusedFuncGraphs(phase);
  }
  MS_LOG(INFO) << "Delete one net resource end. " << clear;
}

void GraphExecutorPy::ClearRes() {
  MS_LOG(INFO) << "Clean Graph executor resource!";
  executor_ = nullptr;
}

GraphExecutorPy::~GraphExecutorPy() {
  MS_LOG(INFO) << "Release Executor!";
  ConfigManager::GetInstance().ResetConfig();
}

void GraphExecutorPy::SaveCompiledGraph(const std::string &phase) {
  // save the graph to GraphExecutorPy
  FuncGraphPtr func_graph = info_[phase]->resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(INFO) << "Save compiled func graph(" << func_graph->ToString() << ") phase(" << phase << ")!";
  info_[phase]->func_graph = func_graph;
  func_graph->set_attr("phase", MakeValue(GetPhasePrefix(phase)));

  if ((func_graph != nullptr) && parallel::IsAutoParallelCareGraph(func_graph)) {
    MS_LOG(DEBUG) << "Save model parallel parameter layout graph!";
    auto res = info_[phase]->resource;
    // When using frontend compile cache, model parallel parameter layout graph is not saved.
    if (res->HasResult(kStepParallelGraph)) {
      func_graph = res->GetResult(kStepParallelGraph).cast<FuncGraphPtr>();
      ExecutorInfoPtr executor_info = std::make_shared<ExecutorInfo>();
      std::string layout_graph = phase + kStepParallelGraph;
      executor_info->func_graph = func_graph;
      info_[layout_graph] = executor_info;
    }
  } else {
    MS_LOG(DEBUG) << "Save model parallel parameter layout graph null!";
  }
  MS_LOG(INFO) << "End save compiled func graph!";
}

void GraphExecutorPy::ParallelPostProcess(const std::string &phase, bool use_compile_cache) {
  // Slice Python parameter obj
  auto layout_graph = phase + kStepParallelGraph;
  // only Parallel graph has tensor_layout
  auto root = GetFuncGraph(layout_graph);
  bool after_shard = false;
  if (phase.find("after_shard") != std::string::npos) {
    after_shard = true;
  }
  // Use compile cache
  if (use_compile_cache) {
    parallel::InitCompileCacheParams(info_[phase]->resource);
    return;
  }
  // Initialize parameters for graph which auto-parallel not care.
  if (root == nullptr && !after_shard) {
    auto graph = info_[phase]->resource->func_graph();
    MS_EXCEPTION_IF_NULL(graph);
    parallel::InitPynativeNoShardParams(graph);
    return;
  }
  MS_EXCEPTION_IF_NULL(root);
  parallel::AutoParallelPostProcess(root);
}

// Clean all resource not used in the future and cache generated during compiling.
void GraphExecutorPy::CleanCompileRes(const ResourcePtr &resource) {
  MS_LOG(INFO) << "Clean compile resource start";
  parallel::ParallelContext::GetInstance()->set_dynamic_shape_parallel_flag(false);
  ProcessStatus::GetInstance().RecordStart(kPipelineClean);
  uint64_t start_time = profiler::GetClockSyscnt();
  abstract::AnalysisContext::ClearContext();
  ClearCompileArgumentsResource();
  ad::ClearPrimBpropOptimizer();
  ad::ClearKPrim();
  ad::DFunctor::Clear();
  ReclaimOptimizer();
  resource->Clean();
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  if (parallel_context->hccl_test_available()) {
    parallel::g_device_manager = nullptr;
  }
  FuncGraphLoopBreaker::Inst().CleanMetaFuncGraphs();
  (void)profiler::CollectHostInfo(kCompiler, kPipelineClean, kPipelineClean, start_time, profiler::GetClockSyscnt(), 0);
  ProcessStatus::GetInstance().RecordEnd();
  CompileCacheContext::GetInstance().Clear();
  parse::Parser::CleanParserResource();
  MS_LOG(INFO) << "Clean compile resource end";
}

namespace {
std::vector<ActionItem> GetActions(const ResourcePtr &resource, const std::string &phase, bool trace_flag = false,
                                   bool erase_parse = false) {
  MS_EXCEPTION_IF_NULL(resource);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_LOG(DEBUG) << "Enable mindRT.";
  context_ptr->set_param<bool>(MS_CTX_ENABLE_MINDRT, true);
  return VmPipeline(resource, trace_flag, erase_parse);
}
}  // namespace

bool GraphExecutorPy::CompileInner(const FuncGraphPtr &graph, const py::tuple &args, const py::dict &kwargs,
                                   const std::string &phase, bool trace_flag) {
  GraphCompilingScope jit_compiling_scope;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->SetCellReuseLevel(CellReuseLevel::kNoCellReuse);
  PhaseManager::GetInstance().set_phase(phase);
  phase_ = phase;

  ExecutorInfoPtr executor_info = std::make_shared<ExecutorInfo>();
  ResourcePtr resource = std::make_shared<Resource>();
  resource->set_func_graph(graph);
  resource->set_pipeline_level(pipeline::kLevelGraph);
  if (CompileCacheEnable()) {
    MS_LOG(EXCEPTION) << "Compile cache is not enabled in PIJit.";
  }

  bool erase_parse = true;
  auto actions = GetActions(resource, phase, trace_flag, erase_parse);
  std::shared_ptr<Pipeline> pip = std::make_shared<Pipeline>(resource, actions);

  // Get the parameters items and add the value to args_abs.
  abstract::AbstractBasePtrList args_abs;
  std::vector<ValuePtr> arguments;
  MS_EXCEPTION_IF_NULL(parallel::ParallelContext::GetInstance());
  bool is_auto_parallel = (parallel::ParallelContext::GetInstance()->parallel_mode() == parallel::kSemiAutoParallel ||
                           parallel::ParallelContext::GetInstance()->parallel_mode() == parallel::kAutoParallel);
  ConvertArgs(args, kwargs, resource, is_auto_parallel, &args_abs, &arguments);
  ConvertSymbolicShape(args, &args_abs);
  bool init_null = resource->func_graph() == nullptr;
  if (!resource->EnableCompileCache() || init_null) {
    AddManagerForFuncGraphArgs(resource, arguments);
  }
  resource->set_arguments(arguments);
  resource->set_args_abs(args_abs);
  resource->set_real_arguments(real_arguments());
  executor_info->arg_list_size = args.size() + kwargs.size();
  executor_info->resource = resource;
  info_[phase] = executor_info;
  pip->Run();

  // Save the compiled graph to MsPipeLine.
  SaveCompiledGraph(phase);
  if (is_auto_parallel) {
    ParallelPostProcess(phase, CompileCacheContext::GetInstance().UseCompileCache());
  }
  CleanCompileRes(resource);
  PhaseManager::GetInstance().ClearPhase();
  PhaseManager::GetInstance().ClearJitConfig();
  MS_LOG(INFO) << "Finish compiling.";
  return true;
}

bool GraphExecutorPy::CompileInner(const py::object &source, const py::tuple &args, const py::dict &kwargs,
                                   const py::object &phase) {
  GraphCompilingScope jit_compiling_scope;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->SetCellReuseLevel(CellReuseLevel::kNoCellReuse);
  // Check if the phase is valid.
  if ((!py::isinstance<py::str>(phase))) {
    MS_LOG(ERROR) << "The `phase` must be string.";
    return false;
  }
  // Check if the function or net is valid.
  if (py::isinstance<py::none>(source)) {
    MS_LOG(ERROR) << "The source object to compile should not be None.";
    return false;
  }
  // Check if the args of function or net is valid.
  CheckArgsValid(source, args);

  source_ = py::cast<std::string>(py::str(source));
  phase_ = py::cast<std::string>(phase);
  PhaseManager::GetInstance().set_phase(phase_);
  obj_desc_ = GetObjDesc(source);
  MS_LOG(INFO) << "Start compiling, phase: " << phase_;

  auto root_func_name = obj_desc_;
  std::replace(root_func_name.begin(), root_func_name.end(), '.', '_');
  std::replace(root_func_name.begin(), root_func_name.end(), '\'', '_');
  opt::LoadPassesConfig(root_func_name);

  PROF_START(compile_graph);
  MS_LOG(DEBUG) << "source: {" << source_ << "}\nargs: " << py::str(const_cast<py::tuple &>(args))
                << "\nkwargs: " << py::str(const_cast<py::dict &>(kwargs));
  EventMessage::PrintCompileStartMsg(phase_, obj_desc_);

  ExecutorInfoPtr executor_info = std::make_shared<ExecutorInfo>();
  ResourcePtr resource = std::make_shared<Resource>(source);
  resource->set_pipeline_level(pipeline::kLevelGraph);
  InitCompileCacheResource(resource, phase_);

  auto actions = GetActions(resource, phase_, false, false);
  std::shared_ptr<Pipeline> pip = std::make_shared<Pipeline>(resource, actions);

  uint64_t start_time = profiler::GetClockSyscnt();
  (void)profiler::CollectHostInfo(kCompiler, kCreateBackend, kCreateBackend, start_time, profiler::GetClockSyscnt(), 0);

  // Get the parameters items and add the value to args_abs.
  abstract::AbstractBasePtrList args_abs;
  std::vector<ValuePtr> arguments;
  MS_EXCEPTION_IF_NULL(parallel::ParallelContext::GetInstance());
  bool is_parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode() == parallel::kSemiAutoParallel ||
                          parallel::ParallelContext::GetInstance()->parallel_mode() == parallel::kAutoParallel;
  bool is_auto_parallel = is_parallel_mode && !py::hasattr(source, parallel::kSkipAutoParallelCompile) &&
                          !py::hasattr(source, parallel::kKeepInputUnchanged);
  ConvertArgs(args, kwargs, resource, is_auto_parallel, &args_abs, &arguments);
  ConvertSymbolicShape(args, &args_abs);
  AddManagerForFuncGraphArgs(resource, arguments);
  resource->set_arguments(arguments);
  resource->set_args_abs(args_abs);
  resource->set_real_arguments(real_arguments());
  executor_info->arg_list_size = args.size() + kwargs.size();
  executor_info->resource = resource;
  info_[phase_] = executor_info;
  pip->Run();

  opt::SavePassesConfig(root_func_name);
  // Save the compiled graph to MsPipeLine.
  SaveCompiledGraph(phase_);
  PROF_START(ParallelPostProcess);
  if (is_parallel_mode) {
    ParallelPostProcess(phase_, CompileCacheContext::GetInstance().UseCompileCache());
  }
  PROF_END(ParallelPostProcess);
  PROF_START(CleanCompileRes);
  CleanCompileRes(resource);
  EventMessage::PrintCompileEndMsg(phase_, obj_desc_);
  PhaseManager::GetInstance().ClearPhase();
  PhaseManager::GetInstance().ClearJitConfig();
  PROF_END(CleanCompileRes);
  MS_LOG(INFO) << "Finish compiling.";
  PROF_END(compile_graph);
  return true;
}

void GraphExecutorPy::ConvertArgs(const py::tuple &args, const py::dict &kwargs, const ResourcePtr &resource,
                                  bool is_auto_parallel, abstract::AbstractBasePtrList *args_abs,
                                  std::vector<ValuePtr> *arguments) {
  MS_EXCEPTION_IF_NULL(args_abs);
  MS_EXCEPTION_IF_NULL(arguments);
  for (std::size_t i = 0; i < args.size(); i++) {
    // In some parallel mode need full_tensor which cause the args of GenerateArgumentsKey not same to compile,
    // So can't use cur_convert_input_ directly.
    auto iter = cur_convert_input_.find(args[i].ptr());
    if (iter != cur_convert_input_.end()) {
      (void)arguments->emplace_back(iter->second.first);
      if (is_auto_parallel) {
        auto abs_item = iter->second.second->Clone();
        (void)parallel::ExtendInputArgsAbstractShape(abs_item, i);
        (void)args_abs->emplace_back(abs_item);
        continue;
      }
      (void)args_abs->emplace_back(iter->second.second);
      SetHookForArgAbstract(resource, args[i], iter->second.second);
      continue;
    }
    ValuePtr converted = nullptr;
    bool success = parse::ConvertData(args[i], &converted);
    if (!success) {
      MS_LOG(INTERNAL_EXCEPTION) << "Fail to convert the " << i << "th argument, args[" << i
                                 << "]: " << py::str(args[i]);
    }
    (void)arguments->emplace_back(converted);
    auto args_abstract_item = ArgsToAbstract(args[i], converted, enable_tuple_broaden_);
    if (is_auto_parallel) {
      (void)parallel::ExtendInputArgsAbstractShape(args_abstract_item, i);
    }
    args_abstract_item->set_user_data<size_t>(kActualArgumentIndex, std::make_shared<size_t>(i));
    (void)args_abs->emplace_back(args_abstract_item);
    SetHookForArgAbstract(resource, args[i], args_abstract_item);
  }
  for (const auto &item : kwargs) {
    auto iter = cur_convert_input_.find(item.first.ptr());
    if (iter != cur_convert_input_.end()) {
      (void)arguments->emplace_back(iter->second.first);
      (void)args_abs->emplace_back(iter->second.second);
      auto keyword_arg_abs = iter->second.second->cast<abstract::AbstractKeywordArgPtr>();
      MS_EXCEPTION_IF_NULL(keyword_arg_abs);
      SetHookForArgAbstract(resource, py::cast<py::object>(item.second), keyword_arg_abs->get_arg());
      continue;
    }
    ValuePtr key = nullptr;
    ValuePtr value = nullptr;
    bool success = parse::ConvertData(py::cast<py::object>(item.first), &key) &&
                   parse::ConvertData(py::cast<py::object>(item.second), &value);
    if (!success) {
      MS_LOG(INTERNAL_EXCEPTION) << "Fail to convert the argument (" << py::str(item.first) << ": "
                                 << py::str(item.second) << ").";
    }
    AbstractBasePtr value_abs = ArgsToAbstract(py::cast<py::object>(item.second), value, enable_tuple_broaden_);
    auto keyword_arg_abs = std::make_shared<abstract::AbstractKeywordArg>(GetValue<std::string>(key), value_abs);
    (void)arguments->emplace_back(value);
    (void)args_abs->emplace_back(keyword_arg_abs);
    SetHookForArgAbstract(resource, py::cast<py::object>(item.second), value_abs);
  }
}

py::object GraphExecutorPy::RunInner(const py::tuple &args, const py::object &phase_obj) {
  JitRunningScope jit_running_scope;
  if (common::GetEnv(kSimulationLevel) == kSimulationLevelCompileGraph) {
    py::int_ ret = 0;
    return ret;
  }
  if (!py::isinstance<py::str>(phase_obj)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Run failed, phase input is not a str";
  }
  auto phase = py::cast<std::string>(phase_obj);
  auto phase_prefix = GetPhasePrefix(phase);
  PhaseManager::GetInstance().set_phase(phase_prefix);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  static const bool enable_infer_boost = ms_context->IsEnableInferBoost();
  if (enable_infer_boost) {
    PhaseManager::GetInstance().set_phase(phase);
  }

  auto ret_val = std::make_shared<py::object>();
  if (info_.count(phase) != 0 && info_[phase]->func_graph != nullptr) {
    if (IsGraphOutputValueNodeOrParameter(info_[phase]->func_graph->output(), args, ret_val)) {
      return *ret_val;
    }
  }
#ifndef WITH_BACKEND
  if (ms_context->backend_policy() == "ge") {
    // Virtual output constructed for test cases.
    if (!args.empty()) {
      return args[0];
    }
    return args;
  }
#endif
  auto iter = info_.find(phase);
  if (iter == info_.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "No executor info. found for phase: " << phase;
  }
  auto &execute_info = iter->second;
  MS_EXCEPTION_IF_NULL(execute_info);
  if (args.size() > execute_info->arg_list_size) {
    MS_LOG(WARNING) << "The args size: " << args.size() << ", full_arg_size: " << execute_info->arg_list_size;
  }
  ProcessVmArg(args, phase, &execute_info->arg_list);
  // Start to run phase.
  ResourcePtr resource = GetResource(phase);
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->HasResult(kNoBackend)) {
    MS_LOG(INFO) << "No backend.";
    return py::none();
  }
  auto run = GetVmEvalFunc(phase);
  if (run == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Can't find run graph func for " << phase;
  }

  MS_LOG(DEBUG) << "Eval run " << ms_context->backend_policy();
  const auto &output = execute_info->func_graph->output();
  MS_EXCEPTION_IF_NULL(output);
  const auto &output_abs = output->abstract();
  MS_EXCEPTION_IF_NULL(output_abs);
  BaseRef value = (*run)(execute_info->arg_list);

  py::object res = BaseRefToPyDataWithUserData(value, output_abs);
  ClearRunArgumentsResource(args.size(), &execute_info->arg_list);
  PhaseManager::GetInstance().ClearPhase();
  PhaseManager::GetInstance().ClearJitConfig();
  MS_LOG(DEBUG) << "Run end";
  return res;
}

void GraphExecutorPy::BuildGraph(const py::dict &init_params, const std::string &phase) const {
  GraphCompilingScope jit_compiling_scope;
  MS_LOG(INFO) << "Start building df graph, phase = " << phase;
  if (info_.count(phase) == 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "No phase in executor: " << GetPhasePrefix(phase);
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (target != kAscendDevice) {
    MS_LOG(INFO) << "Only Support ascend.";
    return;
  }

  auto iter = info_.find(phase);
  if (iter == info_.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Phase " << phase << " must compile.";
  }

  std::map<std::string, std::shared_ptr<Tensor>> init_tensors{};
  ConvertObjectToTensors(init_params, &init_tensors, info_.at(phase)->func_graph);
  backend::BackendManager::GetInstance().ConvertIR(info_.at(phase)->func_graph, init_tensors, backend::IRFormat::kAir);
  MS_LOG(INFO) << "End building df graph, phase = " << phase;
}

void GraphExecutorPy::ConvertObjectToTensors(const py::dict &dict,
                                             std::map<std::string, std::shared_ptr<Tensor>> *const tensors,
                                             const FuncGraphPtr &anf_graph) const {
  for (auto item : dict) {
    if ((!py::isinstance<py::str>(item.first))) {
      MS_LOG(WARNING) << "Type of key of py_dict is not string, ignore it.";
      continue;
    }
    std::shared_ptr<Tensor> tensor;
    std::string name = py::cast<std::string>(item.first);

    if (py::isinstance<py::float_>(item.second.attr("data"))) {
      // convert float to tensor with shape([1])
      tensor = tensor::from_spec(kNumberTypeFloat32, std::vector<int64_t>({1}), device::DeviceType::kCPU);
      *(static_cast<float *>(tensor->data_c())) = py::cast<float>(item.second.attr("data"));
    } else if (py::isinstance<py::int_>(item.second.attr("data"))) {
      // convert int64_t to tensor with shape([1])
      tensor = tensor::from_spec(kNumberTypeInt32, std::vector<int64_t>({1}), device::DeviceType::kCPU);
      *(static_cast<float *>(tensor->data_c())) = py::cast<float>(item.second.attr("data"));
    } else if (tensor::IsTensorPy(item.second.attr("data"))) {
      // cast tensor
      tensor = tensor::ConvertToTensor(item.second.attr("data"));
    }

    if (tensor == nullptr) {
      MS_LOG(EXCEPTION) << "Get default value for " << name << " failed";
    }
    (void)tensors->emplace(name, tensor);
  }
}

void GraphExecutorPy::UpdataParamNodeDefaultInput(const std::string &phase,
                                                  const std::unordered_map<std::string, py::object> &params_value) {
  FuncGraphPtr func_graph = info_[phase]->resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(DEBUG) << "UpdataParamNodeDefaultInput for func graph(" << func_graph->ToString() << ") phase(" << phase
                << ")!";
  auto &params = func_graph->parameters();
  for (const auto &param : params) {
    MS_EXCEPTION_IF_NULL(param);
    auto param_cast = param->cast_ptr<Parameter>();
    MS_EXCEPTION_IF_NULL(param_cast);
    auto iter = params_value.find(param_cast->name());
    if (iter != params_value.end()) {
      auto value_ptr = tensor::ConvertToTensorPyWrapper(iter->second);
      param_cast->set_default_param(value_ptr);
    }
  }
}

py::bytes GraphExecutorPy::GetRandomStatus(const std::string &phase) const {
  auto iter = info_.find(phase);
  if (iter == info_.end()) {
    MS_LOG(ERROR) << "Phase " << phase << " must compile.";
    return "";
  }
  std::string random_status = "";
  return py::bytes(random_status.c_str(), random_status.size());
}

void GraphExecutorPy::PyExePath(const py::object &py_exe_path) const {
  if (!py::isinstance<py::str>(py_exe_path)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Failed, py_exe_path input is not a str";
  }
  auto py_exe_path_s = py::cast<std::string>(py_exe_path);
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<std::string>(MS_CTX_PYTHON_EXE_PATH, py_exe_path_s);
}

void GraphExecutorPy::KernelBuildServerDir(const py::object &kernel_build_server_dir) const {
  if (!py::isinstance<py::str>(kernel_build_server_dir)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Failed, kernel_build_server_dir input is not a str";
  }
  auto kernel_build_server_dir_s = py::cast<std::string>(kernel_build_server_dir);
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<std::string>(MS_CTX_KERNEL_BUILD_SERVER_DIR, kernel_build_server_dir_s);
}
void GraphExecutorPy::SetOptimizeConfig(const py::list &optimize_cfg) {
  opt::PassConfigure::Instance().SetOptimizeConfig(optimize_cfg);
}
std::string GraphExecutorPy::GetOptimizeConfig() { return opt::PassConfigure::Instance().GetOptimizeConfig(); }
void GraphExecutorPy::SetConfigPasses(const py::list &passes) {
  opt::PassConfigure::Instance().SetConfigPasses(passes);
}
py::list GraphExecutorPy::GetRunningPasses() { return opt::PassConfigure::Instance().GetRunningPasses(); }

void GraphExecutorPy::ExportGraph(const std::string &file_name, const std::string &phase, const py::object encrypt,
                                  char *key) {
  GraphCompilingScope jit_compiling_scope;
  MS_LOG(INFO) << "Start exporting df graph, phase = " << phase;
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (target != kAscendDevice) {
    MS_EXCEPTION(ValueError) << "Only support export file in 'AIR' format with Ascend backend.";
  }

  bool is_save_to_file = true;
  if (key != nullptr) {
    if (py::isinstance<py::none()>(encrypt)) {
      MS_LOG(ERROR) << "ERROR: encrypt is not a function";
      return;
    }
    is_save_to_file = false;
  }
  auto iter = info_.find(phase);
  if (iter == info_.end()) {
    MS_LOG(ERROR) << "Phase " << phase << " must compile.";
    return;
  }
  FuncGraphPtr func_graph = info_[phase]->func_graph;
  MS_EXCEPTION_IF_NULL(func_graph);

  string save_str =
    backend::BackendManager::GetInstance().ExportIR(func_graph, file_name, is_save_to_file, backend::IRFormat::kAir);

  if (is_save_to_file) {
    MS_LOG(INFO) << "End exporting df graph, phase = " << phase;
    return;
  }
  // save_to_mem in GE & save to file use encrypt
  py::bytes model_bytes(save_str);
  py::bytes key_bytes(key);

  // call python encrypt func
  py::bytes encrypted_model_stream = encrypt(model_bytes, key_bytes);
  if (encrypted_model_stream == py::none()) {
    MS_LOG(ERROR) << "ERROR: Model encrypt fail";
    return;
  }
  // save to file
  std::ofstream ofs(file_name);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "ERROR: Open File '" << file_name << "' failed!";
    return;
  }
  ofs << std::string(encrypted_model_stream);
  ofs.close();
  MS_LOG(INFO) << "End exporting df graph, phase = " << phase;
}

}  // namespace pipeline
}  // namespace mindspore
