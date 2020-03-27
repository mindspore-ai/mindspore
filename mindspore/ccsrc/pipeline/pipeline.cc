/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "pipeline/pipeline.h"

#include <sstream>
#include <map>
#include <unordered_map>
#include <cstdlib>
#include <algorithm>

#include "pipeline/pass.h"
#include "pipeline/parse/data_converter.h"
#include "optimizer/ad/dfunctor.h"
#include "ir/meta_tensor.h"
#include "transform/convert.h"
#include "transform/df_graph_manager.h"
#include "transform/graph_builder.h"
#include "transform/graph_runner.h"
#include "debug/anf_ir_dump.h"
#include "debug/anf_ir_utils.h"
#include "utils/config_manager.h"
#include "utils/convert_utils.h"
#include "utils/utils.h"
#include "utils/base_ref.h"
#include "vm/segment_runner.h"
#include "parallel/context.h"
#include "parallel/graph_util/get_parallel_info.h"
#include "device/kernel_runtime_manager.h"
#include "debug/trace.h"

namespace mindspore {
// namespace to support intermediate representation definition
namespace pipeline {
using Tensor = mindspore::tensor::Tensor;
using MetaTensor = mindspore::tensor::MetaTensor;
using TensorOrderMap = std::map<std::string, std::shared_ptr<Tensor>>;
using mindspore::abstract::AbstractTensor;
using mindspore::abstract::AbstractTensorPtr;
using mindspore::abstract::AbstractTuple;
using mindspore::abstract::AbstractTuplePtr;
using mindspore::transform::DfGraphConvertor;
using mindspore::transform::DfGraphManager;
using mindspore::transform::GeTensorPtr;
using mindspore::transform::MeTensorPtr;
using mindspore::transform::Status;
using mindspore::transform::TransformUtil;

const char IR_TYPE_ANF[] = "anf_ir";
const char IR_TYPE_ONNX[] = "onnx_ir";

ExecutorPyPtr ExecutorPy::executor_ = nullptr;
std::mutex ExecutorPy::instance_lock_;

std::unordered_map<abstract::AbstractBasePtrList, int, abstract::AbstractBasePtrListHasher,
                   abstract::AbstractBasePtrListEqual>
  g_args_cache;

namespace {
std::string GetBaseNameForIR(int stage_idx, const std::string& action_name) {
  std::ostringstream oss;
  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    MS_LOG(EXCEPTION) << "ms_context is nullptr";
  }
  auto save_graphs_path = ms_context->save_graphs_path();
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }
  oss << save_graphs_path << "/" << stage_idx << "_" << action_name;
  return oss.str();
}

std::string GetFilePathName(const std::string& file_name) {
  std::ostringstream oss;
  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    MS_LOG(EXCEPTION) << "ms_context is nullptr";
  }
  auto save_graphs_path = ms_context->save_graphs_path();
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }
  oss << save_graphs_path << "/" << file_name;
  return oss.str();
}
}  // namespace

// We will not execute graph when output is constant or just input itself.
static bool IsGraphOutputValueNodeOrParameter(const AnfNodePtr& output, const py::tuple& args,
                                              const std::shared_ptr<py::object>& ret_val) {
  if (output->isa<ValueNode>()) {
    MS_LOG(INFO) << "Graph's output is a constant. No need to execute.";
    ValuePtr value = GetValueNode(output);
    *ret_val = ValuePtrToPyData(value);
    return true;
  }

  // Adapter will transform values in __init__() and construct() to parameters, this could cause
  // inputs (a.k.a args in current function) size less than parameters'.
  if (output->isa<Parameter>()) {
    MS_LOG(INFO) << "Graph's output is a parameter. If all params are inputs, no need to execute.";
    if (args.empty()) {
      MS_LOG(EXCEPTION) << "Inputs size is 0, let graph to be executed.";
    }
    // Find the right parameter as ret_val.
    auto func_graph = output->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    auto params = func_graph->parameters();
    if (params.empty()) {
      MS_EXCEPTION(UnknownError) << "Graph's parameters size is 0";
    }
    if (args.size() != params.size()) {
      MS_LOG(EXCEPTION) << "Input size " << args.size() << " not equal to params size " << params.size()
                        << ", let graph to be executed.";
    }

    auto it = std::find(params.begin(), params.end(), output);
    if (it == params.end()) {
      MS_EXCEPTION(UnknownError) << "When graph output is Parameter,  it should be found in graph parameters";
    }
    size_t index = it - params.cbegin();
    if (index >= args.size()) {
      MS_EXCEPTION(UnknownError) << "Index " << index << " equal or larger than args size " << args.size() << ".";
    }
    *ret_val = args[index];
    return true;
  }
  return false;
}

py::tuple GenerateKey(const std::string& name, const std::unordered_map<std::string, py::object>& defaults) {
  MS_LOG(DEBUG) << "GenerateKey args size:" << defaults.size();
  abstract::AbstractBasePtrList args_spec;

  for (auto arg : defaults) {
    if (py::isinstance<py::module>(arg.second)) {
      MS_LOG(EXCEPTION) << "GenerateKey failed, argument input should not be py::module";
    }
    ValuePtr converted = nullptr;
    if (!parse::ConvertData(arg.second, &converted)) {
      MS_LOG(EXCEPTION) << "GenerateKey convert arg failed";
    }
    args_spec.push_back(abstract::FromValue(converted, true));
  }
  if (g_args_cache.count(args_spec) == 0) {
    static int key = 0;
    MS_LOG(INFO) << "start new args and compile key:" << key;
    g_args_cache[args_spec] = key++;
  }
  py::tuple argSpec = py::tuple(2);
  argSpec[0] = name;
  argSpec[1] = g_args_cache[args_spec];
  return argSpec;
}

py::bool_ VerifyInputSignature(const py::list input_signature, const py::tuple inputs) {
  MS_LOG(DEBUG) << "Verify args size:" << inputs.size();
  if (inputs.size() != input_signature.size()) {
    MS_LOG(ERROR) << "signature size not equal to args size";
    return false;
  }

  size_t count = 0;
  for (auto arg_obj : inputs) {
    if (py::hasattr(arg_obj, PYTHON_TENSOR_FLAG)) {
      MS_LOG(DEBUG) << "Verify Tensor";
      std::shared_ptr<Tensor> m_tensor = arg_obj.cast<std::shared_ptr<Tensor>>();
      if (m_tensor == nullptr) {
        MS_LOG(ERROR) << "Verify Tensor error, get ptr is null";
        return false;
      }
      std::shared_ptr<MetaTensor> sig = input_signature[count].cast<std::shared_ptr<MetaTensor>>();
      std::vector<int> sig_shape = sig->shape();
      TypePtr sig_type = sig->Dtype();

      std::vector<int> tensor_shape = m_tensor->shape_c();
      if (tensor_shape != sig_shape) {
        MS_LOG(ERROR) << "Python input shape is incompatible with input_signature";
        return false;
      }

      if (*m_tensor->Dtype() != *sig_type) {
        MS_LOG(ERROR) << "Python input type(" << m_tensor->Dtype()->ToString() << ") incompatible with input_signature("
                      << sig_type->ToString() << ")";
        return false;
      }
    }
    count++;
  }

  return true;
}

ExecutorPy::ExecutorPy() {
  // because Ge only support one Session exist at the same time ,so we delete the old one
  DfGraphManager::GetInstance().DeleteGraphRunner();
  DfGraphManager::GetInstance().DeleteGeSession();
}

ResourcePtr ExecutorPy::GetResource(const std::string& phase) {
  MS_LOG(DEBUG) << "phase size:" << info_.size();
  if (info_.count(phase) == 0) {
    return nullptr;
  }
  return info_[phase]->resource;
}

std::string GetPhasePrefix(const std::string& phase) {
  auto pos = phase.find('.');
  if (pos == std::string::npos) {
    MS_LOG(EXCEPTION) << "phase has no . for prefix" << phase;
  }
  return phase.substr(0, pos);
}

FuncGraphPtr ExecutorPy::GetFuncGraph(const std::string& phase) {
  if (info_.count(phase) == 0) {
    MS_LOG(EXCEPTION) << "no phase in executor:" << GetPhasePrefix(phase);
  }
  return info_[phase]->func_graph;
}

std::size_t ExecutorPy::ArgListSize(const std::string& phase) {
  if (info_.count(phase) == 0) {
    MS_LOG(EXCEPTION) << "no phase in executor:" << GetPhasePrefix(phase);
  }
  return info_[phase]->arg_list_size;
}

compile::VmEvalFuncPtr ExecutorPy::GetVmEvalFunc(const std::string& phase) {
  ResourcePtr res = GetResource(phase);
  MS_EXCEPTION_IF_NULL(res);
  if (res->results().find(kOutput) != res->results().end() && res->results()[kOutput].is<compile::VmEvalFuncPtr>()) {
    return res->results()[kOutput].cast<compile::VmEvalFuncPtr>();
  }
  MS_LOG(ERROR) << "GetVmEvalFunc vm model can't find kOutput:" << kOutput;
  return nullptr;
}

bool ExecutorPy::HasCompiled(const std::string& phase) const {
  if (info_.count(phase) == 0) {
    return false;
  }
  return true;
}

py::bytes ExecutorPy::GetFuncGraphProto(const std::string& phase, const std::string& ir_type) {
  FuncGraphPtr fg_ptr = GetFuncGraph(phase);
  if (fg_ptr == nullptr) {
    for (auto& item : info_) {
      MS_LOG(DEBUG) << "Phase key is: " << item.first;
    }
    MS_LOG(EXCEPTION) << "Can not find func graph " << phase;
  }

  if (ir_type == IR_TYPE_ANF) {
    std::string proto_str = GetFuncGraphProtoString(fg_ptr);
    if (proto_str.empty()) {
      MS_LOG(EXCEPTION) << "Graph proto is empty.";
    }
    return proto_str;
  }

  if (ir_type == IR_TYPE_ONNX) {
    std::string proto_str = GetOnnxProtoString(fg_ptr);
    if (proto_str.empty()) {
      MS_LOG(EXCEPTION) << "Graph proto is empty.";
    }
    return proto_str;
  }

  MS_LOG(EXCEPTION) << "Unknown ir type: " << ir_type;
}

py::dict ExecutorPy::GetParameterLayout(const std::string& phase) {
  MS_LOG(DEBUG) << "GetParameterLayout!";
  std::string layout_graph = phase + kStepParallelGraph;
  auto graph = GetFuncGraph(layout_graph);
  return mindspore::parallel::GetParameterLayout(graph);
}

py::dict ExecutorPy::GetCNodeStrategy(const std::string& phase) {
  MS_LOG(DEBUG) << "GetCNodeStrategy!";
  std::string layout_graph = phase + kStepParallelGraph;
  auto graph = GetFuncGraph(layout_graph);
  return mindspore::parallel::GetCNodeStrategy(graph);
}

py::dict ExecutorPy::GetAllreduceFusion(const std::string& phase) {
  MS_LOG(INFO) << "GetAllreduceFusion!";
  auto graph = GetFuncGraph(phase);
  return mindspore::parallel::GetAllreduceFusion(graph);
}

void ExecutorPy::DelNetRes(const std::string& id) {
#ifdef ENABLE_GE
  FinalizeGe();
#endif
  if (executor_ != nullptr) {
    bool flag = false;
    auto tmp_info = info_;
    for (auto& item : tmp_info) {
      if (item.first.find(id) != string::npos) {
        MS_LOG(INFO) << "delete network res:" << item.first;
        (void)info_.erase(item.first);
        flag = true;
      }
    }

    if (flag && info_.size() == 0) {
      DfGraphManager::GetInstance().DeleteGraphRunner();
      DfGraphManager::GetInstance().EraseAnfGraph();
      DfGraphManager::GetInstance().DeleteGeSession();
    }
  }
}

void ExecutorPy::ClearRes() {
  MS_LOG(INFO) << "clean executor Resrouce!";
  executor_ = nullptr;
}

ExecutorPy::~ExecutorPy() {
  MS_LOG(INFO) << "Release Executor!";
  ConfigManager::GetInstance().ResetConfig();
}

void ExecutorPy::SaveCompiledGraph(const std::string& phase_s) {
  // save the graph to ExecutorPy
  FuncGraphPtr func_graph = info_[phase_s]->resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(parallel::ParallelContext::GetInstance());
  std::string parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();

  MS_LOG(INFO) << "save compiled func graph(" << func_graph->ToString() << ") phase(" << phase_s << ")!";
  info_[phase_s]->func_graph = func_graph;
  if ((func_graph != nullptr) &&
      ((parallel_mode == parallel::AUTO_PARALLEL) || (parallel_mode == parallel::SEMI_AUTO_PARALLEL))) {
    MS_LOG(DEBUG) << "save model parallel parameter layout graph!";
    func_graph = info_[phase_s]->resource->results()[kStepParallelGraph].cast<FuncGraphPtr>();
    ExecutorInfoPtr excutor_info = std::make_shared<ExecutorInfo>();
    std::string layout_graph = phase_s + kStepParallelGraph;
    excutor_info->func_graph = func_graph;
    info_[layout_graph] = excutor_info;
  } else {
    MS_LOG(DEBUG) << "save model parallel parameter layout graph null!";
  }
  MS_LOG(INFO) << "end save compiled func graph!";
}

bool ExecutorPy::ChangeExportGeirUseVmFlag(bool use_vm, const std::string& phase_s) const {
  std::string phase_prefix = GetPhasePrefix(phase_s);

  if (use_vm && phase_prefix == "export") {
    MS_LOG(INFO) << "use ge backend to export geir";
    use_vm = false;
  }
  return use_vm;
}

void ExecutorPy::GetGeBackendPolicy() const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string backend = ms_context->backend_policy();
  if (backend != "ge") {
    MS_LOG(EXCEPTION) << "" << backend << " backend policy is not supported under ge backend!";
  }
}

bool ExecutorPy::CompileInner(const py::object& obj, const py::tuple& args, const py::object& phase, bool use_vm) {
  MS_LOG(DEBUG) << "Start ExecutorPy compile!";
  if ((!py::isinstance<py::str>(phase))) {
    MS_LOG(ERROR) << "arg phase must be string.";
    return false;
  }
  // check the arg valid?
  if (py::isinstance<py::none>(obj)) {
    MS_LOG(ERROR) << "Find error: parse obj is None.";
    return false;
  }
#ifdef ENABLE_GE
  GetGeBackendPolicy();
#endif
  ExecutorInfoPtr excutor_info = std::make_shared<ExecutorInfo>();
  std::string phase_s = py::cast<std::string>(phase);
  MS_LOG(INFO) << "ExecutorPy compile phase:" << phase_s << "!";
  ResourcePtr resource = std::make_shared<Resource>(obj);
  std::vector<ActionItem> p_actions;

  use_vm = ChangeExportGeirUseVmFlag(use_vm, phase_s);

  if (use_vm) {
    // Create backend and session
    resource->results()[kBackend] = compile::CreateBackend();
    p_actions = VmPipeline();
  } else {
    p_actions = GePipeline();
  }

  std::shared_ptr<Pipeline> pip = std::make_shared<Pipeline>(resource, p_actions);

  // get the parameters items and add the value to args_spec
  abstract::AbstractBasePtrList args_spec;
  std::size_t size = args.size();
  for (std::size_t i = 0; i < size; i++) {
    ValuePtr converted = nullptr;
    bool succ = parse::ConvertData(args[i], &converted);
    if (!succ) {
      MS_LOG(EXCEPTION) << "args convert error";
    }
    bool broaden = true;
    args_spec.push_back(abstract::FromValue(converted, broaden));
  }

  resource->set_args_spec(args_spec);
  excutor_info->arg_list_size = size;
  excutor_info->resource = resource;
  info_[phase_s] = excutor_info;
  pip->Run();

  // save the run graph func to MsPipeLine
  SaveCompiledGraph(phase_s);

  resource->Clean();
  // Reclaim all resource used by optimizer;
  ReclaimOptimizer();

  MS_LOG(INFO) << "End ExecutorPy compile!";
  return true;
}

void ExecutorPy::ReleaseResource(const py::object& phase) {
  ResourcePtr res = GetResource(py::cast<std::string>(phase));
  if (res != nullptr) {
    res->Clean();
  }
  // Reclaim all resource used by optimizer;
  ReclaimOptimizer();
}

static std::string PrintArgs(const py::tuple& args) {
  py::print(args);
  return "";
}

bool ExecutorPy::Compile(const py::object& obj, const py::tuple& args, const py::object& phase, bool use_vm) {
  bool ret_value = false;

  try {
    MS_LOG(DEBUG) << PrintArgs(args);
    ret_value = CompileInner(obj, args, phase, use_vm);
  } catch (const py::error_already_set& ex) {
    // print function call stack info before release
    std::ostringstream oss;
    trace::TraceGraphInfer();
    trace::GetInferStackInfo(oss);
    // call py::print to output function call stack to STDOUT, in case of output the log to file, the user can see
    // these info from screen, no need to open log file to find these info
    py::print(oss.str());
    MS_LOG(ERROR) << oss.str();
    ReleaseResource(phase);

    // re-throw this exception to Python interpreter to handle it
    throw(py::error_already_set(ex));
  } catch (const py::type_error& ex) {
    ReleaseResource(phase);
    throw py::type_error(ex);
  } catch (const py::value_error& ex) {
    ReleaseResource(phase);
    throw py::value_error(ex);
  } catch (const std::exception& ex) {
    ReleaseResource(phase);
    // re-throw this exception to Python interpreter to handle it
    throw(std::runtime_error(ex.what()));
  } catch (...) {
    ReleaseResource(phase);
    std::string exName(abi::__cxa_current_exception_type()->name());
    MS_LOG(EXCEPTION) << "Error occurred when compile graph. Exception name: " << exName;
  }

  return ret_value;
}

void SetGeOption(const std::map<std::string, std::string>& options) {
  ConfigManager::GetInstance().set_ge_initialize_options(options);
}

bool InitDistribute(const std::map<std::string, std::string>& options) {
  ConfigManager::GetInstance().set_parallel_strategy(ParallelStrategy::DISTRIBUTION);
  MS_LOG(INFO) << "ME run in DISTRIBUTION strategy mode";

  SetGeOption(options);
#ifdef ENABLE_GE
  auto ge_options = ConfigManager::GetInstance().ge_initialize_options();
  {
    // Release GIL before calling into (potentially long-running) C++ code
    py::gil_scoped_release release;
    if (ge::GEInitialize(ge_options) != ge::GRAPH_SUCCESS) {
      MS_LOG(ERROR) << "Initialize GE failed!";
      return false;
    }
  }
#endif
  MS_LOG(DEBUG) << "Initialize Ge success";
  return true;
}

#ifdef ENABLE_LOAD_ANF_IR
// get MindSpore Intermediate Representation File
std::string GetMsIrFile(void) {
  std::string file;
  const char* path = getenv("MS_IR_FILE");
  if (path == nullptr) {
    return file;
  }

  char real_path[PATH_MAX] = {0};
  if (realpath(path, real_path) == nullptr) {
    MS_LOG(ERROR) << "MS IR Path error, " << path;
    return file;
  }
  file = real_path;
  return file;
}

void RunPipelineAction(const ActionItem& action, pipeline::ResourcePtr resource, bool* result) {
  MS_EXCEPTION_IF_NULL(resource);
  MS_EXCEPTION_IF_NULL(result);

  std::string ir_file = GetMsIrFile();
  (void)parse::python_adapter::set_python_scoped();
  if (ir_file.empty()) {
    *result = action.second(resource);
    return;
  }

  // when in loading anf ir mode, action `parse` do nothing
  if (action.first == "parse") {
    parse::PythonAdapter::SetPythonEnvFlag(true);
    return;
  }

  // load MindSpore IR from file
  if (action.first == "symbol_resolve") {
    MS_LOG(DEBUG) << "" << action.first << " read ir file: " << ir_file;
    std::vector<FuncGraphPtr> graphs = ImportIR(ir_file);
    if (graphs.size() == 0) {
      MS_LOG(EXCEPTION) << "" << action.first << " read ir file " << ir_file << " failed as no graph found";
    }
    auto manager = resource->manager();
    MS_EXCEPTION_IF_NULL(manager);
    for (auto& graph : graphs) {
      manager->AddFuncGraph(graph);
    }
    resource->set_func_graph(graphs[0]);
    return;
  }

  // do normal action when not in `parse` and `symbol_resolve` stage
  *result = action.second(resource);
}
#endif

void Pipeline::Run() {
  MS_LOG(INFO) << "pipeline run";
  MS_EXCEPTION_IF_NULL(resource_);
  FuncGraphPtr user_graph = nullptr;

  WITH(MsProfile::GetProfile())[&user_graph, this]() {
    int i = 0;
    for (auto& action : actions_) {
#ifdef ENABLE_TIMELINE
      DumpTime& dump_time = DumpTime::GetInstance();
      dump_time.Record(action.first, GetTime(), true);
#endif
      bool result = true;
      WITH(MsProfile::GetProfile()->Step(action.first))[&result, &action, this]() {
        MS_LOG(DEBUG) << "Action " << action.first << " start ...";
#ifdef ENABLE_LOAD_ANF_IR
        RunPipelineAction(action, resource_, &result);
#else
        result = action.second(resource_);
#endif
        MS_LOG(DEBUG) << "Action " << action.first << " end.";
      };
      if (!result) {
        MS_LOG(EXCEPTION) << "pipeline running to end, failed in step:" << action.first;
      }
      if (MsContext::GetInstance()->save_graphs_flag() && resource_->func_graph() != nullptr) {
        auto graph = resource_->func_graph();
        if (graph != nullptr) {
          user_graph = graph;
          std::string base_name = GetBaseNameForIR(i, action.first);

          // generate IR file in dot format, which can be converted to svg file using graphviz dot command
          draw::Draw(base_name + ".dot", graph);
          // generate IR file in human readable format
          DumpIR(base_name + ".ir", graph);
          // generate IR file in a heavily commented format, which can also be reloaded
          if (action.first != "parse") {
            ExportIR(base_name + ".dat", std::to_string(i), graph);
          }
        }
#ifdef MS_DEBUG
        // Dump graph cnode list
        MS_LOG(INFO) << "Show CNode list after " << action.first;
        graph->DumpCNodeList();
#endif
      }
      if (resource_->func_graph() != nullptr) {
        auto func_graph = resource_->func_graph();
        if (func_graph->has_flag(GRAPH_FLAG_HAS_EFFECT)) {
          func_graph->EraseUnusedNodeInOrder();
          func_graph->CheckOrder();
          for (auto fg : func_graph->func_graphs_used_total()) {
            MS_LOG(DEBUG) << "Check order graph " << fg->ToString() << ".";
            fg->EraseUnusedNodeInOrder();
            fg->CheckOrder();
          }
        }
      }
      i++;
#ifdef ENABLE_TIMELINE
      dump_time.Record(action.first, GetTime(), false);
#endif
    }
  };
#ifdef ENABLE_PROFILE
  MsProfile::Print();
  MsProfile::Reset();
#endif

  if (MsContext::GetInstance()->save_graphs_flag() && (user_graph != nullptr)) {
    std::string user_graph_file = GetFilePathName("ModelDigraph.dot");
    MS_LOG(DEBUG) << "save user graph to: " << user_graph_file;
    draw::DrawUserFuncGraph(user_graph_file, user_graph);

#ifdef ENABLE_DUMP_IR
    std::string filename = GetFilePathName("ms_output.pb");
    ChangeFileMode(filename, S_IRWXU);
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
      MS_LOG(ERROR) << "Open file '" << filename << "' failed!";
      return;
    }
    ofs << GetFuncGraphProtoString(user_graph);
    ofs.close();
    // set file mode to read only by user
    ChangeFileMode(filename, S_IRUSR);
#endif
  }
  MS_LOG(INFO) << "end";
}

void ExecutorPy::ProcessVmArg(const py::tuple& args, const std::string& phase, VectorRef* arg_list) {
  std::size_t size = args.size();

  for (std::size_t i = 0; i < size; i++) {
    py::object arg = args[i];
    auto ms_context = MsContext::GetInstance();
    if (ms_context->backend_policy() == kMsConvert && py::isinstance<py::array>(arg)) {
      MS_LOG(EXCEPTION) << "args[" << i << "] is numpy array, not tensor";
    }
    (*arg_list).push_back(arg);
  }

  ResourcePtr res = GetResource(phase);
  MS_EXCEPTION_IF_NULL(res);
  auto graph = res->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> graph_params = graph->parameters();
  std::size_t graph_params_size = graph_params.size();
  if ((*arg_list).size() != graph_params_size) {
    // maybe some default parameter
    for (std::size_t i = (*arg_list).size(); i < graph_params_size; i++) {
      MS_EXCEPTION_IF_NULL(graph_params[i]);
      py::object obj = dyn_cast<Parameter>(graph_params[i])->default_param();
      py::object p_value = py::cast<py::object>(parse::python_adapter::GetPyObjAttr(obj, "default_input"));
      (*arg_list).push_back(p_value);
    }
  }
}

py::object ExecutorPy::Run(const py::tuple& args, const py::object& phase) {
  std::size_t size = args.size();
  if (!py::isinstance<py::str>(phase)) {
    MS_LOG(EXCEPTION) << "Run failed, phase input is not a str";
  }
  auto phase_s = py::cast<std::string>(phase);
  std::string backend = MsContext::GetInstance()->backend_policy();
  if (backend == "ge") {
    return ExecDFGraph(args, phase_s);
  }
  std::size_t full_arg_size = ArgListSize(phase_s);
  if (size > full_arg_size) {
    MS_LOG(WARNING) << "The arg num : size = " << size << ". full_arg_size = " << full_arg_size;
  }
  VectorRef arg_list;
  ProcessVmArg(args, phase_s, &arg_list);

  compile::VmEvalFuncPtr run = GetVmEvalFunc(phase_s);
  if (run == nullptr) {
    MS_LOG(EXCEPTION) << "Can't find run graph func for " << phase_s;
  }

  MS_LOG(DEBUG) << "eval run";
  BaseRef value = (*run)(arg_list);
  MS_LOG(DEBUG) << "run end";
  return BaseRefToPyData(value);
}

py::object StructureOutput(const AbstractBasePtr& output, const py::tuple& data, size_t* count) {
  MS_EXCEPTION_IF_NULL(output);

  if (!output->isa<AbstractTuple>()) {
    ValuePtr value = output->BuildValue();
    if (value != kAnyValue) {
      return ValuePtrToPyData(value);
    }
    if (!output->isa<AbstractTensor>()) {
      MS_LOG(EXCEPTION) << "Output can only be tensor except for constants, but got "
                        << output->BuildValue()->ToString() << ".";
    }
    if (*count >= data.size()) {
      MS_LOG(EXCEPTION) << "The number of elements in the outputs : " << data.size()
                        << " less than the number of elements required. ";
    }
    auto shape = output->BuildShape();
    auto shape_act = shape->cast<abstract::ShapePtr>()->shape();
    Tensor tensor_exp = py::cast<Tensor>(data[*count]);
    if (shape_act != tensor_exp.shape()) {
      MS_LOG(EXCEPTION) << "The shape of the tensor returned from GE is not the same as "
                           "the shape of the tensor derived from ME.";
    }
    return data[(*count)++];
  }

  auto tuple_output = output->cast<AbstractTuplePtr>();
  AbstractBasePtrList elements = tuple_output->elements();
  size_t size = elements.size();
  py::tuple tp = py::tuple(size);
  for (size_t i = 0; i < size; i++) {
    tp[i] = StructureOutput(elements[i], data, count);
  }
  return std::move(tp);
}

std::shared_ptr<py::object> DoExecGraph(const FuncGraphPtr& graph, const std::vector<MeTensorPtr>& inputs,
                                        const std::string& phase) {
  std::vector<GeTensorPtr> ge_tensors = TransformUtil::ConvertInputTensors(inputs, kOpFormat_NCHW);
  if (ge_tensors.size() != inputs.size()) {
    MS_LOG(ERROR) << "args convert to ge tensor error";
    return nullptr;
  }

  std::vector<GeTensorPtr> ge_outputs;
  transform::RunOptions run_options;

  run_options.name = phase;

  auto graph_runner = DfGraphManager::GetInstance().GetGraphRunner();

  if (graph_runner == nullptr) {
    MS_LOG(ERROR) << "Can not found GraphRunner";
    return nullptr;
  }

  {
    // Release GIL before calling into (potentially long-running) C++ code
    py::gil_scoped_release release;
    MS_LOG(DEBUG) << "Run graph begin, inputs size is: " << inputs.size();
    Status ret = graph_runner->RunGraph(run_options, ge_tensors, &ge_outputs);
    MS_LOG(DEBUG) << "Run graph finish, outputs size is: " << ge_outputs.size();
    if (ret != Status::SUCCESS) {
      MS_LOG(ERROR) << "Exec graph failed";
      return nullptr;
    }
  }

  std::vector<MeTensorPtr> me_outputs = TransformUtil::ConvertGeTensors(ge_outputs);
  if (me_outputs.size() != ge_outputs.size()) {
    MS_LOG(ERROR) << "Convert output Ge tensor to Me tensor failed";
  }

  py::tuple outputs(me_outputs.size());
  for (std::size_t i = 0; i < outputs.size(); i++) {
    outputs[i] = *me_outputs[i];
  }

  std::shared_ptr<py::object> ret = nullptr;

#ifdef ENABLE_GE
  AnfNodePtr root = graph->get_return();
  MS_EXCEPTION_IF_NULL(root);
  AbstractBasePtr output = root->abstract();
  size_t count = 0;
  py::object oj = StructureOutput(output, outputs, &count);
  ret = std::make_shared<py::object>(oj);
#else
  if (outputs.size() == 1) {
    ret = std::make_shared<py::object>(outputs[0]);
  } else {
    ret = std::make_shared<py::object>(outputs);
  }
#endif

  return ret;
}

void DoExecNonInputGraph(const std::string& phase) {
  std::vector<GeTensorPtr> ge_tensors;
  std::vector<GeTensorPtr> ge_outputs;
  transform::RunOptions run_options;
  run_options.name = phase;
  auto graph_runner = DfGraphManager::GetInstance().GetGraphRunner();

  if (graph_runner == nullptr) {
    MS_LOG(ERROR) << "Can not found GraphRunner";
    return;
  }
  {
    // Release GIL before calling into (potentially long-running) C++ code
    py::gil_scoped_release release;
    Status ret = graph_runner->RunGraph(run_options, ge_tensors, &ge_outputs);
    if (ret != Status::SUCCESS) {
      MS_LOG(ERROR) << "Exec graph:" << run_options.name << " failed";
      return;
    }
  }
}

void ExecutorPy::ProcessGeArg(const py::tuple& args, const std::string& phase, std::vector<tensor::TensorPtr>* inputs) {
  // check the arg and use the ExecutorPy args
  std::size_t size = args.size();
  if (size != ArgListSize(phase)) {
    MS_LOG(EXCEPTION) << "The real arg num : size = " << size << ". graph_arg_size = " << ArgListSize(phase);
  }

  // process the first args of tensor
  // only in Dataset Feed Mode, fp_bp graph need input tensors
  if (ConfigManager::GetInstance().dataset_mode() == DS_FEED_MODE) {
    for (std::size_t i = 0; i < size; i++) {
      ValuePtr converted = nullptr;
      bool succ = parse::ConvertData(args[i], &converted);
      if (!succ) {
        MS_LOG(EXCEPTION) << "args convert error";
      }
      if (converted->isa<tensor::Tensor>()) {
        (*inputs).push_back(converted->cast<tensor::TensorPtr>());
      } else {
        MS_LOG(EXCEPTION) << "args, " << converted->ToString() << " is not tensor";
      }
    }
  }
}

py::object ExecutorPy::ExecDFGraph(const py::tuple& args, const std::string& phase) {
  std::string phase_prefix = GetPhasePrefix(phase);

  if (phase_prefix == "save") {
    DoExecNonInputGraph(phase);
    ConfigManager::GetInstance().ResetConfig();
    return py::none();
  }

  if (info_.count(phase) == 0) {
    MS_LOG(EXCEPTION) << "has no phase:" << phase;
  }

#if (!defined ENABLE_GE) || (defined ENABLE_INFER)
  // Now don't use the graph because the exec ge function don't take effect
  MS_EXCEPTION_IF_NULL(info_[phase]->func_graph);
  if (ENABLE_TRAIN != info_[phase]->func_graph->flags()["training"]) {
    MS_LOG(ERROR) << "Graph training mode mismatch mode of libraries";
    ConfigManager::GetInstance().ResetConfig();
    return py::none();
  }
#endif

  std::shared_ptr<py::object> ret_val = std::make_shared<py::object>();
  if (IsGraphOutputValueNodeOrParameter(info_[phase]->func_graph->output(), args, ret_val)) {
    ConfigManager::GetInstance().ResetConfig();
    return *ret_val;
  }

  std::vector<tensor::TensorPtr> inputs;
  ProcessGeArg(args, phase, &inputs);

  std::shared_ptr<py::object> ret = DoExecGraph(GetFuncGraph(phase), inputs, phase);
  ConfigManager::GetInstance().ResetConfig();
  if (ret != nullptr) {
    return *ret;
  } else {
    MS_LOG(EXCEPTION) << "exec graph failed";
  }
}

void ExecutorPy::RunInitGraph(const py::dict& init_params, const std::string& phase) {
  MS_LOG(DEBUG) << "ExecInitGraph start.";
  TensorOrderMap inputs_with_name{};
  ConvertObjectToTensors(init_params, &inputs_with_name);
  std::vector<tensor::TensorPtr> inputs;
  (void)std::transform(inputs_with_name.begin(), inputs_with_name.end(), std::back_inserter(inputs),
                       [](const std::pair<std::string, tensor::TensorPtr>& item) { return item.second; });

  std::vector<GeTensorPtr> ge_tensors = TransformUtil::ConvertInputTensors(inputs, kOpFormat_NCHW);
  if (ge_tensors.size() != inputs.size()) {
    MS_LOG(ERROR) << "Args convert to ge tensor error.";
    return;
  }
  MS_LOG(DEBUG) << "Run graph begin, inputs size is: " << inputs.size() << ".";

  std::vector<GeTensorPtr> ge_outputs;
  transform::RunOptions run_options;

  run_options.name = phase;
  if (DfGraphManager::GetInstance().GetGraphByName(phase) == nullptr) {
    MS_LOG(WARNING) << "Can not find " << phase << " sub graph, don't need data init subgraph in INFER mode.";
    return;
  }
  auto graph_runner = DfGraphManager::GetInstance().GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
  }
  {
    // Release GIL before calling into (potentially long-running) C++ code
    py::gil_scoped_release release;
    Status ret = graph_runner->RunGraph(run_options, ge_tensors, &ge_outputs);
    if (ret != Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "Exec " << phase << " graph failed.";
    }

    MS_LOG(INFO) << "Exec " << phase << " graph success.";

    if ((ConfigManager::GetInstance().parallel_strategy() == ParallelStrategy::DISTRIBUTION) &&
        (DfGraphManager::GetInstance().GetGraphByName(BROADCAST_GRAPH_NAME) != nullptr)) {
      run_options.name = BROADCAST_GRAPH_NAME;
      ret = graph_runner->RunGraph(run_options, ge_tensors, &ge_outputs);
      if (ret != Status::SUCCESS) {
        MS_LOG(EXCEPTION) << "Exec BROADCAST_GRAPH_NAME failed.";
      }
      MS_LOG(INFO) << "Exec broadcast graph success.";
    }
  }
}

Status CreateSessionAndGraphRunner(bool is_training = true) {
  std::shared_ptr<ge::Session> sess = DfGraphManager::GetInstance().GetGeSession();
  if (sess == nullptr) {
    transform::SessionOptions options;
    if (is_training) {
      options["ge.trainFlag"] = "1";
      options["ge.streamNum"] = "100";
      options["ge.enabledLocalFmkop"] = "1";
      options["ge.hcomParallel"] = "1";
    } else {
      options["ge.trainFlag"] = "0";
    }

    options["ge.enablePrintOpPass"] = "0";
    sess = transform::GraphRunner::NewSession(options);
    if (sess == nullptr) {
      MS_LOG(ERROR) << "Init data graph failed, because of create Ge session failed";
      return Status::FAILED;
    } else {
      DfGraphManager::GetInstance().SetGeSession(sess);
    }
  }

  transform::GraphRunnerOptions options;
  options.sess_ptr = sess;
  auto graph_runner = std::make_shared<transform::GraphRunner>(options);
  if (graph_runner == nullptr) {
    MS_LOG(ERROR) << "Create new graph runner failed";
    return Status::FAILED;
  } else {
    DfGraphManager::GetInstance().SetGraphRunner(graph_runner);
  }

  return Status::SUCCESS;
}

void ExecutorPy::ConvertObjectToTensors(const py::dict& dict, TensorOrderMap* const tensors) {
  for (auto item : dict) {
    if ((!py::isinstance<py::str>(item.first))) {
      MS_LOG(WARNING) << "Type of key of py_dict is not string, ignore it.";
      continue;
    }
    std::shared_ptr<Tensor> tensor;
    std::string name = py::cast<std::string>(item.first);
    if (py::isinstance<py::float_>(item.second.attr("default_input"))) {
      // convert float to tensor with shape([1])
      tensor = std::make_shared<Tensor>(kNumberTypeFloat32, std::vector<int>({1}));
      *(static_cast<float*>(tensor->data_c(true))) = py::cast<float>(item.second.attr("default_input"));
    } else if (py::isinstance<py::int_>(item.second.attr("default_input"))) {
      // convert int to tensor with shape([1])
      tensor = std::make_shared<Tensor>(kNumberTypeInt32, std::vector<int>({1}));
      *(static_cast<float*>(tensor->data_c(true))) = py::cast<float>(item.second.attr("default_input"));
    } else if (py::hasattr(item.second.attr("default_input"), PYTHON_TENSOR_FLAG)) {
      // cast tensor
      tensor = py::cast<std::shared_ptr<Tensor>>(item.second.attr("default_input"));
    }

    if (tensor == nullptr) {
      MS_LOG(EXCEPTION) << "Get default value for " << name << " failed";
    }
    (void)tensors->emplace(name, tensor);
  }
}

bool ExecutorPy::AddDFGraph(const py::dict& init_params, const std::string& phase, const py::object& broadcast_params) {
  FuncGraphPtr anf_graph = info_[phase]->func_graph;
  DfGraphConvertor convertor(anf_graph);

  size_t pos = phase.find('.');
  std::string net_id = ((pos == std::string::npos || pos == phase.size() - 1) ? phase : phase.substr(pos + 1));
  std::string phase_prefix = phase.substr(0, pos);

  if (phase_prefix == "export") {
    MS_LOG(INFO) << "Set DfGraphConvertor training : false";
    convertor.set_training(false);
  }

  TensorOrderMap init_tensors{};
  ConvertObjectToTensors(init_params, &init_tensors);
  (void)convertor.ConvertAllNode().InitParam(init_tensors).BuildGraph();

  if (broadcast_params != py::none()) {
    if (!py::isinstance<py::dict>(broadcast_params)) {
      MS_LOG(ERROR) << "Invalid broadcast params, it must be py::dict type";
      return false;
    }
    py::dict broadcast = broadcast_params.cast<py::dict>();
    if (broadcast.empty()) {
      (void)convertor.GenerateBroadcastGraph(init_tensors);
    } else {
      TensorOrderMap broadcast_tensors{};
      ConvertObjectToTensors(broadcast, &broadcast_tensors);
      (void)convertor.GenerateBroadcastGraph(broadcast_tensors);
    }
    MS_LOG(INFO) << "Generate broadcast graph with params and broadcast_empty is " << broadcast.empty();
  }

  (void)convertor.GenerateCheckpointGraph();
  if (convertor.ErrCode() != 0) {
    DfGraphManager::GetInstance().ClearGraph();
    MS_LOG(ERROR) << "convert df graph failed, err:" << convertor.ErrCode();
    return false;
  }

  if (MsContext::GetInstance()->save_graphs_flag()) {
    convertor.DrawComputeGraph(GetFilePathName("ge_graph.dot"));                      // for debug
    convertor.DrawInitGraph(GetFilePathName("init_graph.dot"));                       // for debug
    convertor.DrawSaveCheckpointGraph(GetFilePathName("save_checkpoint_graph.dot"));  // for debug
  }
  std::string init_graph = "init_subgraph." + net_id;
  std::string checkpoint_name = "save." + net_id;
  if (phase == "train") {
    (void)DfGraphManager::GetInstance().AddGraph(phase, convertor.GetComputeGraph(), {{"ge.exec.variable_acc", "1"}});
  } else {
    (void)DfGraphManager::GetInstance().AddGraph(phase, convertor.GetComputeGraph());
  }
  (void)DfGraphManager::GetInstance().AddGraph(init_graph, convertor.GetInitGraph());
  (void)DfGraphManager::GetInstance().AddGraph(checkpoint_name, convertor.GetSaveCheckpointGraph());
  (void)DfGraphManager::GetInstance().AddGraph(BROADCAST_GRAPH_NAME, convertor.GetBroadcastGraph());

  DfGraphManager::GetInstance().SetAnfGraph(checkpoint_name, anf_graph);

  return true;
}

FuncGraphPtr ExecutorPy::BuildDFGraph(const py::dict& init_params, const std::string& phase,
                                      const py::object& broadcast_params) {
  if (info_.count(phase) == 0) {
    MS_LOG(EXCEPTION) << "no phase in executor:" << GetPhasePrefix(phase);
  }
  FuncGraphPtr anf_graph = info_[phase]->func_graph;

  if (MsContext::GetInstance()->save_graphs_flag()) {
    draw::Draw(GetFilePathName("anf_graph.dot"), anf_graph);  // for debug
    DumpIR(GetFilePathName("anf_graph.ir"), anf_graph, true);
  }

  if (!AddDFGraph(init_params, phase, broadcast_params)) {
    MS_LOG(ERROR) << "GenConvertor failed";
    return nullptr;
  }

#if ENABLE_TRAIN
  (void)setenv("GE_TRAIN", "1", 1);
#else
  (void)setenv("GE_TRAIN", "0", 1);
#endif

  if (CreateSessionAndGraphRunner(static_cast<bool>(ENABLE_TRAIN)) != Status::SUCCESS) {
    MS_LOG(ERROR) << "Create GE Session or GraphRunner failed.";
    return nullptr;
  }

  return anf_graph;
}

bool InitExecDataset(const std::string& queue_name, int64_t iter_num, int64_t batch_size,
                     const std::vector<TypePtr>& types, const std::vector<std::vector<int64_t>>& shapes,
                     const std::vector<int64_t>& input_indexes, const std::string& phase) {
  std::string name = MsContext::GetInstance()->backend_policy();
  if (name == kMsConvert || name == kMsVm) {
    return InitExecDatasetVm(queue_name, iter_num, batch_size, types, shapes, input_indexes);
  } else {
    return InitExecDatasetGe(queue_name, iter_num, batch_size, types, shapes, input_indexes, phase);
  }
}

bool InitExecDatasetGe(const std::string& queue_name, int64_t size, int64_t batch_size,
                       const std::vector<TypePtr>& types, const std::vector<std::vector<int64_t>>& shapes,
                       const std::vector<int64_t>& input_indexes, const std::string& phase) {
  // Convert types to GE types and TF types
  std::vector<int64_t> ge_types;
  (void)std::transform(types.begin(), types.end(), std::back_inserter(ge_types), [](const TypePtr& i) -> int64_t {
    return transform::TransformUtil::ConvertDataType(i->type_id());
  });

  ConfigManager::GetInstance().set_dataset_mode(DatasetMode::DS_GRAPH_MODE);
  ConfigManager::GetInstance().set_iter_num(size);
  ConfigManager::GetInstance().set_dataset_phase(phase);

  DatasetGraphParam param(queue_name, size, batch_size, ge_types, shapes, input_indexes);
  ConfigManager::GetInstance().set_dataset_param(param);

  if (transform::BuildDatasetGraph(param, phase) != transform::SUCCESS) {
    MS_LOG(ERROR) << "Build dateset graph failed.";
    return false;
  }

#if ENABLE_TRAIN
  (void)setenv("GE_TRAIN", "1", 1);
#else
  (void)setenv("GE_TRAIN", "0", 1);
#endif

  if (CreateSessionAndGraphRunner(static_cast<bool>(ENABLE_TRAIN)) != Status::SUCCESS) {
    MS_LOG(ERROR) << "Create GE Session or GraphRunner failed.";
    return false;
  }

  MS_LOG(INFO) << "DoExecNonInputGraph:" << phase;
  DoExecNonInputGraph(phase);

  return true;
}

bool InitExecDatasetVm(const std::string& queue_name, int64_t size, int64_t batch_size,
                       const std::vector<TypePtr>& types, const std::vector<std::vector<int64_t>>& shapes,
                       const std::vector<int64_t>& input_indexes) {
  MS_LOG(INFO) << "Start InitDataSet Entry";
  std::vector<int> int_input_indexes;
  (void)std::transform(input_indexes.begin(), input_indexes.end(), std::back_inserter(int_input_indexes),
                       [](int64_t item) { return static_cast<int>(item); });
  std::vector<std::vector<int>> int_shapes;
  (void)std::transform(shapes.begin(), shapes.end(), std::back_inserter(int_shapes),
                       [](const std::vector<int64_t>& item) {
                         std::vector<int> vector_item;
                         (void)std::transform(item.begin(), item.end(), std::back_inserter(vector_item),
                                              [](int64_t inner_item) { return static_cast<int>(inner_item); });
                         return vector_item;
                       });
  auto p_init = std::make_shared<Primitive>("InitDataSetQueue");
  p_init->set_attr("queue_name", MakeValue(queue_name));
  p_init->set_attr("size", MakeValue(static_cast<int>(size)));
  p_init->set_attr("batch_size", MakeValue(static_cast<int>(batch_size)));
  p_init->set_attr("types", MakeValue(types));
  p_init->set_attr("shapes", MakeValue(int_shapes));
  p_init->set_attr("input_indexes", MakeValue(int_input_indexes));

  const std::vector<std::string> emply_str_list;
  p_init->set_attr("input_names", MakeValue(emply_str_list));
  p_init->set_attr("output_names", MakeValue(emply_str_list));

  FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
  auto app_init = std::make_shared<CNode>(AnfNodePtrList{NewValueNode(p_init)}, func_graph);
  func_graph->set_output(app_init);
  auto manager = MakeManager();
  manager->AddFuncGraph(func_graph);

  // AbstractNone indicates there is no output for this apply node.
  auto abstract_none = std::make_shared<abstract::AbstractNone>();
  app_init->set_abstract(abstract_none);

  auto backend = compile::CreateBackend();
  MS_EXCEPTION_IF_NULL(backend);
  auto convert_fn = backend->convert_fn();
  MS_EXCEPTION_IF_NULL(convert_fn);
  // Convert CNodeList to LinConvertResult.
  ConfigManager::GetInstance().set_iter_num(1);
  auto runner = convert_fn({app_init});
  backend->Link(runner.graph_id);
  ConfigManager::GetInstance().set_iter_num(size);

  if (!(*runner.run)) {
    // empty function
    MS_LOG(EXCEPTION) << "Backend " << backend->name() << " unsupports tdt dataset.";
  }

  // launch init dataset runner without inputs and outputs
  VectorRef args;
  auto fn = runner.run;
  (void)(*fn)(args);
  MS_LOG(DEBUG) << "InitDataSetVm End.";
  return true;
}

void InitGe() {
  // set python env flag
  mindspore::parse::python_adapter::set_python_env_flag(true);
  // open tsd before ge initialize
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->OpenTsd()) {
    MS_LOG(EXCEPTION) << "open tsd failed";
  }
  (void)ms_context->InitGe();
}

void FinalizeGe() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  (void)context_ptr->FinalizeGe();
  (void)context_ptr->CloseTsd();
}

void ResetOpId() { mindspore::id_generator::reset_id(); }

void InitHccl() {
#ifdef ENABLE_GE
  (void)InitGe();
#else
  mindspore::parse::python_adapter::set_python_env_flag(true);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  (void)ms_context->OpenTsd();
  uint32_t device_id = ms_context->device_id();
  std::string device_name = ms_context->device_target();

  if (ms_context->backend_policy() == "ms" && ms_context->device_target() == kAscendDevice) {
    auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(device_name, device_id);
    MS_EXCEPTION_IF_NULL(runtime_instance);
    if (!runtime_instance->Init()) {
      MS_LOG(ERROR) << "kernel runtime init error.";
      return;
    }
  }
#endif
}

void FinalizeHccl() {
#ifdef ENABLE_GE
  (void)FinalizeGe();
#else
  device::KernelRuntimeManager::Instance().ClearRuntimeResource();
#endif
}
void ExportDFGraph(const std::string& file_name, const std::string&, const std::string& phase) {
  MS_LOG(DEBUG) << "ExportGraph Begin";
  transform::DfGraphWrapperPtr wrap_ptr = DfGraphManager::GetInstance().GetGraphByName(phase);
  if (wrap_ptr == nullptr) {
    MS_LOG(ERROR) << "Get graph form DfGraphManager failed!";
    return;
  }

  transform::DfGraphPtr ge_graph = wrap_ptr->graph_ptr_;
  if (nullptr == ge_graph) {
    MS_LOG(ERROR) << "The export graph is null";
    return;
  }

  (void)ge_graph->SaveToFile(file_name);

  MS_LOG(DEBUG) << "ExportGraph End";
}

}  // namespace pipeline
}  // namespace mindspore
